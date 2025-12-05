from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

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
        key: jax.Array,
    ):
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_task_tokens = num_task_tokens
        self.num_colors = num_colors

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
            key=enc_key,
        )

        self.norm = eqx.nn.LayerNorm(embed_dim, use_weight=True, use_bias=True)

        self.head = eqx.nn.Linear(embed_dim, num_colors * (patch_size**2), key=head_key)
        b_head = jnp.zeros_like(self.head.bias)
        self.head = eqx.tree_at(lambda m: m.bias, self.head, b_head)

        self.dropout = eqx.nn.Dropout(dropout)

    def __call__(
        self,
        pixels: Int[Array, "H W"],
        task_id: Int[Array, " 1"],
        *,
        attention_mask: Optional[Int[Array, "H W"]] = None,
        key: Optional[jax.Array] = None,
        inference: bool = False,
    ) -> Float[Array, "C H W"]:
        drop_key, enc_key = (None, None) if key is None else jax.random.split(key)
        inference = inference or key is None

        color_lookup = jax.vmap(jax.vmap(self.color_embed))
        embedded = color_lookup(pixels.astype(jnp.int32))
        embedded = jnp.transpose(embedded, (2, 0, 1))

        patch_tokens = self.patch_embed(embedded)
        patch_tokens = patch_tokens + self.positional_embed
        task_tokens = self.task_token_embed(task_id.astype(jnp.int32))
        task_tokens = task_tokens.reshape(self.num_task_tokens, -1)

        hidden_states = jnp.concatenate([task_tokens, patch_tokens], axis=0)
        hidden_states = self.dropout(hidden_states, key=drop_key, inference=inference)

        attention_mask = (
            self._token_mask(attention_mask) if attention_mask is not None else None
        )

        encoded = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            key=enc_key,
            inference=inference,
        )

        encoded = jax.vmap(self.norm)(encoded)
        patch_states = encoded[self.num_task_tokens :, :]
        logits_flat = jax.vmap(self.head)(patch_states)

        logits = logits_flat.reshape(
            self.token_grid,
            self.token_grid,
            self.patch_size,
            self.patch_size,
            self.num_colors,
        )
        logits = jnp.transpose(logits, (0, 2, 1, 3, 4))
        logits = logits.reshape(self.image_size, self.image_size, self.num_colors)
        logits = jnp.transpose(logits, (2, 0, 1))
        return logits

    def _token_mask(self, attention_mask: Int[Array, "H W"]) -> Int[Array, " S"]:
        mask_grid = attention_mask.reshape(
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

    @staticmethod
    def _trunc_normal(key: jax.Array, shape: tuple) -> jax.Array:
        return (
            jax.random.truncated_normal(key, lower=-2.0, upper=2.0, shape=shape) * 0.02
        )
