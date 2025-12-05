from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp

import equinox as eqx
import jax
import jax.numpy as jnp


import equinox as eqx
import jax
import jax.numpy as jnp

import equinox as eqx
import jax
import jax.numpy as jnp


def _rotate_half(x: jax.Array) -> jax.Array:
    original_shape = x.shape
    x = x.reshape(original_shape[:-1] + (-1, 2))
    x1, x2 = x[..., 0], x[..., 1]
    return jnp.stack([-x2, x1], axis=-1).reshape(original_shape)


class RotaryEmbedding(eqx.Module):
    freqs_cos: jax.Array
    freqs_sin: jax.Array
    rope_skip_dim: int = eqx.field(static=True)

    def __init__(self, head_dim: int, pt_seq_len: int, rope_skip_dim: int = 0):
        self.rope_skip_dim = rope_skip_dim

        dim = head_dim // 2
        inv_freq = 1.0 / (10000.0 ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))

        t = jnp.arange(pt_seq_len, dtype=jnp.float32)

        freqs_1d = jnp.einsum("i,f->if", t, inv_freq)
        freqs_1d = jnp.repeat(freqs_1d, 2, axis=-1)
        freq_dim = freqs_1d.shape[-1]

        freqs_h = freqs_1d[:, None, :]
        freqs_w = freqs_1d[None, :, :]
        freqs_h = jnp.broadcast_to(freqs_h, (pt_seq_len, pt_seq_len, freq_dim))
        freqs_w = jnp.broadcast_to(freqs_w, (pt_seq_len, pt_seq_len, freq_dim))

        freqs_2d = jnp.concatenate([freqs_h, freqs_w], axis=-1)
        freqs_flat = freqs_2d.reshape(pt_seq_len * pt_seq_len, -1)

        self.freqs_cos = jnp.cos(freqs_flat)
        self.freqs_sin = jnp.sin(freqs_flat)

    def __call__(self, t: jax.Array) -> jax.Array:
        prefix = t[:, : self.rope_skip_dim, :]
        x = t[:, self.rope_skip_dim :, :]

        seq_len = x.shape[1]
        cos = jax.lax.stop_gradient(self.freqs_cos[:seq_len, :])
        sin = jax.lax.stop_gradient(self.freqs_sin[:seq_len, :])

        cos = cos[None, :, :]
        sin = sin[None, :, :]
        x_rot = x * cos + _rotate_half(x) * sin

        if self.rope_skip_dim == 0:
            return x_rot

        return jnp.concatenate([prefix, x_rot], axis=1)


class PatchEmbed(eqx.Module):
    conv: eqx.nn.Conv2d
    grid: int = eqx.field(static=True)
    embed_dim: int = eqx.field(static=True)
    patch_size: int = eqx.field(static=True)

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        embed_dim: int,
        *,
        key: jax.Array,
    ):
        self.grid = image_size // patch_size
        self.embed_dim = embed_dim
        self.patch_size = patch_size

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
        x = self.conv(x)
        x = jnp.transpose(x, (1, 2, 0))
        x = x.reshape(self.grid * self.grid, self.embed_dim)
        return x


class Attention(eqx.Module):
    qkv: eqx.nn.Linear
    proj: eqx.nn.Linear
    attn_dropout: eqx.nn.Dropout
    proj_dropout: eqx.nn.Dropout
    rotary: RotaryEmbedding

    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    scale: float = eqx.field(static=True)

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        token_grid: int,
        rope_skip_dim: int,
        *,
        key: jax.Array,
    ):
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        qkv_key, proj_key = jax.random.split(key)
        self.qkv = eqx.nn.Linear(embed_dim, 3 * embed_dim, key=qkv_key)
        self.proj = eqx.nn.Linear(embed_dim, embed_dim, key=proj_key)

        self.attn_dropout = eqx.nn.Dropout(dropout)
        self.proj_dropout = eqx.nn.Dropout(dropout)

        self.rotary = RotaryEmbedding(
            head_dim=self.head_dim,
            pt_seq_len=token_grid,
            rope_skip_dim=rope_skip_dim,
        )

    def __call__(
        self,
        x: jax.Array,
        *,
        attention_mask: Optional[jax.Array],
        key: Optional[jax.Array],
        inference: bool,
    ) -> jax.Array:
        seq_len = x.shape[0]
        attn_key, proj_key = (None, None) if key is None else jax.random.split(key)
        dropout_inference = inference or key is None

        qkv = jax.vmap(self.qkv)(x)
        qkv = qkv.reshape(seq_len, 3, self.num_heads, self.head_dim)

        qkv = jnp.transpose(qkv, (1, 2, 0, 3))
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.rotary(q)
        k = self.rotary(k)

        attn_scores = jnp.einsum("hld,hmd->hlm", q, k) * self.scale

        if attention_mask is not None:
            attn_scores = jnp.where(
                attention_mask[None, None, :],
                attn_scores,
                jnp.finfo(attn_scores.dtype).min,
            )

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
        rope_skip_dim: int,
        *,
        key: jax.Array,
    ):
        attn_key, linear1_key, linear2_key = jax.random.split(key, 3)

        self.attn = Attention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            token_grid=token_grid,
            rope_skip_dim=rope_skip_dim,
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
        attention_mask: Optional[jax.Array],
        key: Optional[jax.Array],
        inference: bool,
    ) -> jax.Array:
        keys = (None, None, None, None) if key is None else jax.random.split(key, 4)
        dropout_inference = inference or key is None

        residual = x
        attn_out = self.attn(
            x, attention_mask=attention_mask, key=keys[0], inference=inference
        )
        attn_out = self.dropout1(attn_out, key=keys[1], inference=dropout_inference)
        x = residual + attn_out

        norm1 = jax.vmap(self.norm1)
        x = norm1(x)

        residual = x
        mlp = jax.vmap(self.linear1)(x)
        mlp = jax.nn.gelu(mlp)
        mlp = self.dropout2(mlp, key=keys[2], inference=dropout_inference)
        mlp = jax.vmap(self.linear2)(mlp)
        mlp = self.dropout3(mlp, key=keys[3], inference=dropout_inference)
        x = residual + mlp

        norm2 = jax.vmap(self.norm2)
        x = norm2(x)
        return x


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
        rope_skip_dim: int,
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
                rope_skip_dim=rope_skip_dim,
                key=layer_key,
            )
            for layer_key in keys
        )

    def __call__(
        self,
        x: jax.Array,
        *,
        attention_mask: Optional[jax.Array],
        key: Optional[jax.Array],
        inference: bool,
    ) -> jax.Array:
        layer_keys = (
            [None] * len(self.layers)
            if key is None
            else jax.random.split(key, len(self.layers))
        )
        for layer, layer_key in zip(self.layers, layer_keys):
            x = layer(
                x, attention_mask=attention_mask, key=layer_key, inference=inference
            )
        return x
