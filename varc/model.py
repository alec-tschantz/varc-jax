from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from .nn import PatchEmbed, Transformer


class ARCViT(eqx.Module):
    color_embed: eqx.nn.Embedding
    task_token_embed: eqx.nn.Embedding
    patch_embed: PatchEmbed
    positional_embed: Float[Array, "S E"]
    encoder: Transformer
    norm: eqx.nn.LayerNorm
    head: eqx.nn.Linear
    dropout: eqx.nn.Dropout

    image_size: int = eqx.field(static=True)
    patch_size: int = eqx.field(static=True)
    num_task_tokens: int = eqx.field(static=True)
    seq_length: int = eqx.field(static=True)
    token_grid: int = eqx.field(static=True)
    num_colors: int = eqx.field(static=True)
    embed_dim: int = eqx.field(static=True)
    dtype: jnp.dtype = eqx.field(static=True)

    def __init__(
        self,
        *,
        num_tasks: int,
        image_size: int,
        num_colors: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float,
        num_task_tokens: int,
        patch_size: int,
        dtype: jnp.dtype,
        key: jax.Array,
    ):
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_task_tokens = num_task_tokens
        self.num_colors = num_colors
        self.embed_dim = embed_dim
        self.dtype = dtype

        self.token_grid = image_size // patch_size
        self.seq_length = self.token_grid * self.token_grid

        (
            color_key,
            task_key,
            patch_key,
            pos_key,
            enc_key,
            head_key,
        ) = jax.random.split(key, 6)

        self.color_embed = eqx.nn.Embedding(num_colors, embed_dim, key=color_key)
        w_color = self._trunc_normal(color_key, self.color_embed.weight.shape)
        self.color_embed = eqx.tree_at(lambda m: m.weight, self.color_embed, w_color)

        self.task_token_embed = eqx.nn.Embedding(
            num_tasks, embed_dim * num_task_tokens, key=task_key
        )
        w_task = self._trunc_normal(task_key, self.task_token_embed.weight.shape)
        self.task_token_embed = eqx.tree_at(
            lambda m: m.weight, self.task_token_embed, w_task
        )

        self.patch_embed = PatchEmbed(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            key=patch_key,
        )

        self.positional_embed = self._trunc_normal(
            pos_key, (self.seq_length, embed_dim)
        )

        self.encoder = Transformer(
            depth=depth,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            token_grid=self.token_grid,
            rope_skip_dim=num_task_tokens,
            dtype=dtype,
            key=enc_key,
        )

        self.norm = eqx.nn.LayerNorm(embed_dim, use_weight=True, use_bias=True)

        self.head = eqx.nn.Linear(embed_dim, num_colors * (patch_size**2), key=head_key)
        b_head = jnp.zeros_like(self.head.bias)
        self.head = eqx.tree_at(lambda m: m.bias, self.head, b_head)

        self.dropout = eqx.nn.Dropout(dropout)

    def __call__(
        self,
        pixels: Int[Array, "B H W"],
        task_id: Int[Array, "B"],
        *,
        attention_mask: Optional[Bool[Array, "B H W"]] = None,
        key: Optional[jax.Array] = None,
        inference: bool = False,
    ) -> Float[Array, "B C H W"]:
        """
        Shapes:
            B: batch size
            H, W: image height/width
            E: embedding dim
            T: number of task tokens
            S: number of input tokens (grid size x grid size)
            C: number of output colors
            P: patch_size
            G: grid_size = height // patch_size
        """

        drop_key, enc_key = (None, None) if key is None else jax.random.split(key)
        inference = inference or key is None

        batch_size = pixels.shape[0]

        # (B, H, W, E)
        color_lookup = jax.vmap(jax.vmap(jax.vmap(self.color_embed)))
        embedded = color_lookup(pixels.astype(jnp.int32))
        embedded = embedded.astype(self.dtype)

        # (B, E, H, W)
        embedded = jnp.transpose(embedded, (0, 3, 1, 2))

        # (B, S, E)
        patch_tokens = jax.vmap(self.patch_embed)(embedded)
        patch_tokens = patch_tokens.astype(self.dtype)

        # (1, S, E)
        pos_embed = self.positional_embed[None, :, :].astype(self.dtype)

        # (B, S, E)
        patch_tokens = patch_tokens + pos_embed

        # (B, T*E)
        task_tokens = jax.vmap(self.task_token_embed)(task_id.astype(jnp.int32))

        # (B, T, E)
        task_tokens = task_tokens.reshape(batch_size, self.num_task_tokens, -1)
        task_tokens = task_tokens.astype(self.dtype)

        # (B, T+S, E)
        hidden_states = jnp.concatenate([task_tokens, patch_tokens], axis=1)

        # (B, T+S, E)
        hidden_states = self.dropout(hidden_states, key=drop_key, inference=inference)

        # (B, T+S)
        attention_mask = (
            self._token_mask(attention_mask) if attention_mask is not None else None
        )

        # (B, T+S, E)
        encoded = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            key=enc_key,
            inference=inference,
        )

        # (B, T+S, E)
        norm = jax.vmap(jax.vmap(self.norm))
        encoded = norm(encoded)

        # (B, S, E)
        patch_states = encoded[:, self.num_task_tokens :, :]

        # (B, S, P*P*C)
        logits_flat = jax.vmap(jax.vmap(self.head))(patch_states).astype(jnp.float32)

        # (B, G, G, P, P, C)
        logits = logits_flat.reshape(
            batch_size,
            self.token_grid,
            self.token_grid,
            self.patch_size,
            self.patch_size,
            self.num_colors,
        )

        # (B, G, P, G, P, C)
        logits = jnp.transpose(logits, (0, 1, 3, 2, 4, 5))

        # (B, H, W, C)
        logits = logits.reshape(
            batch_size, self.image_size, self.image_size, self.num_colors
        )

        # (B, C, H, W)
        logits = jnp.transpose(logits, (0, 3, 1, 2))
        return logits

    def _token_mask(self, attention_mask: Bool[Array, "B H W"]) -> Bool[Array, "B S"]:
        def _single_mask(mask: Bool[Array, "H W"]) -> Bool[Array, "S"]:
            mask_grid = mask.reshape(
                self.token_grid,
                self.patch_size,
                self.token_grid,
                self.patch_size,
            )
            mask_grid = mask_grid.max(axis=1).max(axis=2)
            flat = mask_grid.reshape(-1)
            is_valid = flat.astype(bool)

            prefix = jnp.ones((self.num_task_tokens,), dtype=bool)
            return jnp.concatenate([prefix, is_valid], axis=0)

        return jax.vmap(_single_mask)(attention_mask)

    @staticmethod
    def _trunc_normal(key: jax.Array, shape: tuple) -> jax.Array:
        return (
            jax.random.truncated_normal(key, lower=-2.0, upper=2.0, shape=shape) * 0.02
        )
