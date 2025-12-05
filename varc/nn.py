from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float


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

        freqs_h = jnp.broadcast_to(
            freqs_1d[:, None, :], (pt_seq_len, pt_seq_len, freq_dim)
        )
        freqs_w = jnp.broadcast_to(
            freqs_1d[None, :, :], (pt_seq_len, pt_seq_len, freq_dim)
        )

        freqs_2d = jnp.concatenate([freqs_h, freqs_w], axis=-1)
        freqs_flat = freqs_2d.reshape(pt_seq_len * pt_seq_len, -1)

        self.freqs_cos = jnp.cos(freqs_flat)
        self.freqs_sin = jnp.sin(freqs_flat)

    def __call__(self, x: Float[Array, "B S H D"]) -> Float[Array, "B S H D"]:
        orig_dtype = x.dtype
        prefix = x[:, : self.rope_skip_dim, :, :]
        main = x[:, self.rope_skip_dim :, :, :]

        seq_len = main.shape[1]
        cos = jax.lax.stop_gradient(self.freqs_cos[:seq_len, :])
        sin = jax.lax.stop_gradient(self.freqs_sin[:seq_len, :])

        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
        rotated = main * cos + _rotate_half(main) * sin
        rotated = rotated.astype(orig_dtype)

        if self.rope_skip_dim == 0:
            return rotated

        return jnp.concatenate([prefix, rotated], axis=1)


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
        x = x.astype(self.conv.weight.dtype)
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
    dropout_rate: float = eqx.field(static=True)
    dtype: jnp.dtype = eqx.field(static=True)

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        token_grid: int,
        rope_skip_dim: int,
        dtype: jnp.dtype,
        *,
        key: jax.Array,
    ):
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout_rate = dropout
        self.dtype = dtype

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
        x: Float[Array, "B S E"],
        *,
        attention_mask: Optional[Bool[Array, "B S"]],
        key: Optional[jax.Array],
        inference: bool,
    ) -> Float[Array, "B S E"]:
        batch, seq_len, _ = x.shape
        attn_key, proj_key = (None, None) if key is None else jax.random.split(key)
        inference = inference or key is None

        qkv = jax.vmap(jax.vmap(self.qkv))(x)
        qkv = qkv.reshape(batch, seq_len, 3, self.num_heads, self.head_dim)

        q = qkv[:, :, 0, :, :].astype(self.dtype)
        k = qkv[:, :, 1, :, :].astype(self.dtype)
        v = qkv[:, :, 2, :, :].astype(self.dtype)

        q = self.rotary(q)
        k = self.rotary(k)

        if attention_mask is not None:
            valid = attention_mask[:, :, None, None].astype(self.dtype)
            q = q * valid
            k = k * valid
            v = v * valid

        attn_out = _flash_attention(q, k, v, attention_mask=attention_mask)

        attn_out = self.attn_dropout(attn_out, key=attn_key, inference=inference)
        attn_out = attn_out.reshape(batch, seq_len, -1)
        attn_out = jax.vmap(jax.vmap(self.proj))(attn_out)
        attn_out = self.proj_dropout(attn_out, key=proj_key, inference=inference)
        attn_out = attn_out.astype(self.dtype)
        return attn_out


class FeedForward(eqx.Module):
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    dropout_mid: eqx.nn.Dropout
    dropout_out: eqx.nn.Dropout
    dtype: jnp.dtype = eqx.field(static=True)

    def __init__(
        self,
        embed_dim: int,
        mlp_dim: int,
        dropout: float,
        dtype: jnp.dtype,
        *,
        key: jax.Array,
    ):
        k1, k2 = jax.random.split(key)
        self.linear1 = eqx.nn.Linear(embed_dim, mlp_dim, key=k1)
        self.linear2 = eqx.nn.Linear(mlp_dim, embed_dim, key=k2)
        self.dropout_mid = eqx.nn.Dropout(dropout)
        self.dropout_out = eqx.nn.Dropout(dropout)
        self.dtype = dtype

    def __call__(
        self, x: Float[Array, "b s e"], *, key: Optional[jax.Array], inference: bool
    ) -> Float[Array, "b s e"]:
        k_mid, k_out = (None, None) if key is None else jax.random.split(key)
        inference = inference or key is None

        hidden = jax.vmap(jax.vmap(self.linear1))(x)
        hidden = jax.nn.gelu(hidden)
        hidden = self.dropout_mid(hidden, key=k_mid, inference=inference)
        hidden = jax.vmap(jax.vmap(self.linear2))(hidden)
        hidden = self.dropout_out(hidden, key=k_out, inference=inference)
        return hidden.astype(self.dtype)


class Block(eqx.Module):
    attn: Attention
    norm1: eqx.nn.LayerNorm
    ff: FeedForward
    dropout1: eqx.nn.Dropout
    norm2: eqx.nn.LayerNorm

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float,
        token_grid: int,
        rope_skip_dim: int,
        dtype: jnp.dtype,
        *,
        key: jax.Array,
    ):
        attn_key, ff_key = jax.random.split(key, 2)

        self.attn = Attention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            token_grid=token_grid,
            rope_skip_dim=rope_skip_dim,
            dtype=dtype,
            key=attn_key,
        )

        self.ff = FeedForward(
            embed_dim=embed_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            dtype=dtype,
            key=ff_key,
        )

        self.norm1 = eqx.nn.LayerNorm(embed_dim, use_weight=True, use_bias=True)
        self.dropout1 = eqx.nn.Dropout(dropout)
        self.norm2 = eqx.nn.LayerNorm(embed_dim, use_weight=True, use_bias=True)

    def __call__(
        self,
        x: Float[Array, "B S E"],
        *,
        attention_mask: Optional[Bool[Array, "B S"]],
        key: Optional[jax.Array],
        inference: bool,
    ) -> Float[Array, "B S E"]:
        keys = (None, None, None) if key is None else jax.random.split(key, 3)
        inference = inference or key is None

        residual = x
        attn_out = self.attn(
            x, attention_mask=attention_mask, key=keys[0], inference=inference
        )
        attn_out = self.dropout1(attn_out, key=keys[1], inference=inference)
        x = residual + attn_out

        norm1 = jax.vmap(jax.vmap(self.norm1))
        x = norm1(x)

        x = x + self.ff(x, key=keys[2], inference=inference)

        norm2 = jax.vmap(jax.vmap(self.norm2))
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
        dtype: jnp.dtype,
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
                dtype=dtype,
                key=layer_key,
            )
            for layer_key in keys
        )

    def __call__(
        self,
        x: Float[Array, "B S E"],
        *,
        attention_mask: Optional[Bool[Array, "B S"]],
        key: Optional[jax.Array],
        inference: bool,
    ) -> Float[Array, "B S E"]:
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


def _flash_attention(
    q: Float[Array, "B S H D"],
    k: Float[Array, "B S H D"],
    v: Float[Array, "B S H D"],
    *,
    attention_mask: Optional[Bool[Array, "B S"]],
) -> Float[Array, "B S H D"]:
    _, q_len, _, _ = q.shape
    s_len = k.shape[1]

    def _pad(x: jax.Array, pad_len: int) -> jax.Array:
        return jnp.pad(x, ((0, 0), (0, pad_len), (0, 0), (0, 0)))

    padded_q_len = ((q_len + 3) // 4) * 4
    padded_s_len = ((s_len + 3) // 4) * 4

    pad_q = padded_q_len - q_len
    pad_s = padded_s_len - s_len

    q_p = _pad(q, pad_q)
    k_p = _pad(k, pad_s)
    v_p = _pad(v, pad_s)

    base_mask = jnp.ones((1, 1, padded_q_len, padded_s_len), dtype=jnp.bool_)
    base_mask = base_mask.at[:, :, q_len:, :].set(False)
    base_mask = base_mask.at[:, :, :, s_len:].set(False)

    if attention_mask is not None:
        key_mask = attention_mask.astype(jnp.bool_)
        if pad_s:
            key_mask = jnp.pad(key_mask, ((0, 0), (0, pad_s)), constant_values=False)
        base_mask = base_mask & key_mask[:, None, None, :]

    attn_out = jax.nn.dot_product_attention(
        query=q_p,
        key=k_p,
        value=v_p,
        mask=base_mask,
        bias=None,
        implementation="cudnn",
        is_causal=False,
    )

    return attn_out[:, :q_len, :, :]


def _rotate_half(x: jax.Array) -> jax.Array:
    original_shape = x.shape
    x = x.reshape(original_shape[:-1] + (-1, 2))
    x1, x2 = x[..., 0], x[..., 1]
    return jnp.stack([-x2, x1], axis=-1).reshape(original_shape)
