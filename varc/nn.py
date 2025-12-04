from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp


def _rotate_half(x: jax.Array) -> jax.Array:
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)


class Embedding(eqx.Module):
    freqs_cos: jax.Array
    freqs_sin: jax.Array
    task_rope_tokens: int = eqx.field(static=True)

    def __init__(self, dim: int, pt_seq_len: int, task_rope_tokens: int = 0):
        freqs = 1.0 / (10000 ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / float(dim)))
        t = (
            jnp.arange(pt_seq_len, dtype=jnp.float32)
            / float(pt_seq_len)
            * float(pt_seq_len)
        )
        freqs = jnp.einsum("i,f->if", t, freqs)  # (pt_seq_len, dim/2)
        freqs = jnp.repeat(freqs, 2, axis=-1)  # (pt_seq_len, dim)
        freqs = jnp.concatenate(
            [
                jnp.broadcast_to(
                    freqs[:, None, :], (pt_seq_len, pt_seq_len, freqs.shape[-1])
                ),
                jnp.broadcast_to(
                    freqs[None, :, :], (pt_seq_len, pt_seq_len, freqs.shape[-1])
                ),
            ],
            axis=-1,
        )  # (pt_seq_len, pt_seq_len, 2 * dim)
        freqs = freqs.reshape(pt_seq_len * pt_seq_len, -1)
        self.freqs_cos = jnp.cos(freqs)
        self.freqs_sin = jnp.sin(freqs)
        self.task_rope_tokens = task_rope_tokens

    def __call__(self, t: jax.Array) -> jax.Array:
        total_seq = t.shape[1]
        usable = total_seq - self.task_rope_tokens
        head_dim = t.shape[2]

        cos = self.freqs_cos[:usable, :head_dim]
        sin = self.freqs_sin[:usable, :head_dim]

        rotated_part = t[:, self.task_rope_tokens :, :head_dim] * cos + _rotate_half(
            t[:, self.task_rope_tokens :, :head_dim]
        ) * sin
        if self.task_rope_tokens == 0:
            return rotated_part
        return jnp.concatenate(
            [t[:, : self.task_rope_tokens, :head_dim], rotated_part], axis=1
        )


class PatchEmbed(eqx.Module):
    conv: eqx.nn.Conv2d
    grid: int = eqx.field(static=True)
    embed_dim: int = eqx.field(static=True)

    def __init__(
        self, image_size: int, patch_size: int, embed_dim: int, *, key: jax.Array
    ):
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")
        self.grid = image_size // patch_size
        self.embed_dim = embed_dim
        self.conv = eqx.nn.Conv2d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            use_bias=True,
            padding=0,
            key=key,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.conv(x)  # (embed_dim, grid, grid)
        x = jnp.transpose(x, (1, 2, 0)).reshape(self.grid * self.grid, self.embed_dim)
        return x


class Attention(eqx.Module):
    qkv: eqx.nn.Linear
    proj: eqx.nn.Linear
    attn_dropout: eqx.nn.Dropout
    proj_dropout: eqx.nn.Dropout
    rotary: Embedding
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    scale: float = eqx.field(static=True)

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        token_grid: int,
        task_rope_tokens: int,
        *,
        key: jax.Array,
    ):
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("RoPE requires even head_dim.")
        self.scale = self.head_dim**-0.5
        qkv_key, proj_key = jax.random.split(key)
        self.qkv = eqx.nn.Linear(embed_dim, 3 * embed_dim, key=qkv_key)
        self.proj = eqx.nn.Linear(embed_dim, embed_dim, key=proj_key)
        self.attn_dropout = eqx.nn.Dropout(dropout)
        self.proj_dropout = eqx.nn.Dropout(dropout)
        self.rotary = Embedding(
            dim=self.head_dim // 2, pt_seq_len=token_grid, task_rope_tokens=task_rope_tokens
        )

    def __call__(
        self,
        x: jax.Array,
        *,
        mask: Optional[jax.Array],
        key: Optional[jax.Array],
        inference: bool,
    ) -> jax.Array:
        seq_len = x.shape[0]
        attn_key, proj_key = (None, None) if key is None else jax.random.split(key)
        dropout_inference = inference or key is None

        qkv = jax.vmap(self.qkv)(x)  # (seq, 3*embed_dim)
        qkv = qkv.reshape(seq_len, 3, self.num_heads, self.head_dim)
        qkv = jnp.transpose(qkv, (1, 2, 0, 3))
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.rotary(q)
        k = self.rotary(k)

        attn_scores = jnp.einsum("hld,hmd->hlm", q, k) * self.scale
        if mask is not None:
            if mask.shape[-1] != seq_len:
                raise ValueError(
                    f"Attention mask must have length {seq_len}, got {mask.shape}."
                )
            attn_scores = attn_scores + (jnp.logical_not(mask)[None, None, :]) * -1e9

        attn_weights = jax.nn.softmax(attn_scores, axis=-1)
        attn_weights = self.attn_dropout(
            attn_weights, key=attn_key, inference=dropout_inference
        )
        context = jnp.einsum("hij,hjd->hid", attn_weights, v)
        context = jnp.transpose(context, (1, 0, 2)).reshape(seq_len, -1)
        context = jax.vmap(self.proj)(context)
        context = self.proj_dropout(context, key=proj_key, inference=dropout_inference)
        return context


class Block(eqx.Module):
    attn: Attention
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    dropout1: eqx.nn.Dropout
    dropout2: eqx.nn.Dropout
    dropout3: eqx.nn.Dropout

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float,
        token_grid: int,
        task_rope_tokens: int,
        *,
        key: jax.Array,
    ):
        attn_key, linear1_key, linear2_key = jax.random.split(key, 3)
        self.attn = Attention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            token_grid=token_grid,
            task_rope_tokens=task_rope_tokens,
            key=attn_key,
        )
        self.norm1 = eqx.nn.LayerNorm(embed_dim, use_weight=True, use_bias=True)
        self.norm2 = eqx.nn.LayerNorm(embed_dim, use_weight=True, use_bias=True)
        self.linear1 = eqx.nn.Linear(embed_dim, mlp_dim, key=linear1_key)
        self.linear2 = eqx.nn.Linear(mlp_dim, embed_dim, key=linear2_key)
        self.dropout1 = eqx.nn.Dropout(dropout)
        self.dropout2 = eqx.nn.Dropout(dropout)
        self.dropout3 = eqx.nn.Dropout(dropout)

    def __call__(
        self,
        x: jax.Array,
        *,
        mask: Optional[jax.Array],
        key: Optional[jax.Array],
        inference: bool,
    ) -> jax.Array:
        keys = (None, None, None, None) if key is None else jax.random.split(key, 4)
        dropout_inference = inference or key is None

        norm1 = jax.vmap(self.norm1)
        norm2 = jax.vmap(self.norm2)

        attn_out = self.attn(norm1(x), mask=mask, key=keys[0], inference=inference)
        attn_out = self.dropout1(attn_out, key=keys[1], inference=dropout_inference)
        x = x + attn_out

        mlp = jax.vmap(self.linear1)(norm2(x))
        mlp = jax.nn.gelu(mlp)
        mlp = self.dropout2(mlp, key=keys[2], inference=dropout_inference)
        mlp = jax.vmap(self.linear2)(mlp)
        mlp = self.dropout3(mlp, key=keys[3], inference=dropout_inference)
        return x + mlp


class Transformer(eqx.Module):
    layers: tuple

    def __init__(
        self,
        depth: int,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float,
        token_grid: int,
        task_rope_tokens: int,
        *,
        key: jax.Array,
    ):
        keys = jax.random.split(key, depth)
        self.layers = tuple(
            Block(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
                token_grid=token_grid,
                task_rope_tokens=task_rope_tokens,
                key=layer_key,
            )
            for layer_key in keys
        )

    def __call__(
        self,
        x: jax.Array,
        *,
        mask: Optional[jax.Array],
        key: Optional[jax.Array],
        inference: bool,
    ) -> jax.Array:
        layer_keys = (
            [None] * len(self.layers)
            if key is None
            else jax.random.split(key, len(self.layers))
        )
        for layer, layer_key in zip(self.layers, layer_keys):
            x = layer(x, mask=mask, key=layer_key, inference=inference)
        return x
