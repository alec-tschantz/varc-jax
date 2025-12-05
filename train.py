import time
from pathlib import Path
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import tyro
import wandb
import torch
from torch.utils.data import DataLoader

from varc import ARCViT, ARCDataset, IGNORE_INDEX, collate_batch


@dataclass
class Config:
    data_root: str = "data/arc"
    train_split: str = "training"
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    embed_dim: int = 512
    num_heads: int = 8
    depth: int = 10
    mlp_dim: int = 2048
    dropout: float = 0.1
    patch_size: int = 2
    num_task_tokens: int = 1
    seed: int = 0
    log_every: int = 10
    wandb_project: str = "varc-jax"
    wandb_run_name: Optional[str] = None
    max_grad_norm: float = 1.0

    image_size: int = 64
    num_colors: int = 12


def _shard_batch(batch: Dict[str, jax.Array], num_devices: int) -> Dict[str, jax.Array]:
    def _reshape(x):
        local_batch_size = x.shape[0] // num_devices
        return x.reshape(num_devices, local_batch_size, *x.shape[1:])

    return jax.tree_util.tree_map(_reshape, batch)


def _to_jax(batch: Dict[str, torch.Tensor]) -> Dict[str, jax.Array]:
    return {
        "inputs": jnp.asarray(batch["inputs"].numpy(), dtype=jnp.int32),
        "task_ids": jnp.asarray(batch["task_ids"].numpy(), dtype=jnp.int32),
        "attention_mask": jnp.asarray(batch["attention_mask"].numpy(), dtype=jnp.bool_),
        "targets": jnp.asarray(batch["targets"].numpy(), dtype=jnp.int32),
    }


def loss_fn(
    model: ARCViT, batch: Dict[str, jax.Array], key: jax.Array
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    sample_keys = jax.random.split(key, batch["inputs"].shape[0])

    def forward_one(inp, task_id, mask, subkey):
        return model(
            inp,
            task_id,
            attention_mask=mask,
            key=subkey,
            inference=False,
        )

    logits = jax.vmap(forward_one)(
        batch["inputs"], batch["task_ids"], batch["attention_mask"], sample_keys
    )

    logits_hw = jnp.transpose(logits, (0, 2, 3, 1))
    logits_flat = logits_hw.reshape(-1, logits.shape[1])
    labels_flat = batch["targets"].reshape(-1)

    valid_mask = labels_flat != IGNORE_INDEX
    labels_masked = jnp.where(valid_mask, labels_flat, 0)
    mask_float = valid_mask.astype(jnp.float32)

    per_elem_loss = optax.softmax_cross_entropy_with_integer_labels(
        logits_flat, labels_masked
    )

    loss_sum = (per_elem_loss * mask_float).sum()
    denom = jnp.maximum(mask_float.sum(), 1.0)
    loss = loss_sum / denom

    preds = jnp.argmax(logits, axis=1)
    pred_flat = preds.reshape(-1)
    correct = (pred_flat == labels_flat).astype(jnp.float32) * mask_float
    pixel_acc = correct.sum() / denom

    exact_fn = lambda p, l: jnp.all((p == l) | (l == IGNORE_INDEX))
    exact_acc = jax.vmap(exact_fn)(preds, batch["targets"]).mean()

    return loss, {"loss": loss, "pixel_acc": pixel_acc, "exact_acc": exact_acc}


def make_train_step(optimizer: optax.GradientTransformation):
    def train_step(
        params: ARCViT,
        static: ARCViT,
        opt_state: optax.OptState,
        batch: Dict[str, jax.Array],
        key: jax.Array,
    ) -> Tuple[ARCViT, ARCViT, optax.OptState, Dict[str, jax.Array]]:
        def compute_loss(p):
            mdl = eqx.combine(p, static)
            return loss_fn(mdl, batch, key)

        (loss, metrics), grads = eqx.filter_value_and_grad(compute_loss, has_aux=True)(
            params
        )

        grads = jax.lax.pmean(grads, axis_name="devices")
        loss = jax.lax.pmean(loss, axis_name="devices")
        metrics = jax.tree.map(lambda x: jax.lax.pmean(x, "devices"), metrics)

        updates, opt_state = optimizer.update(grads, opt_state, params=params)
        params = eqx.apply_updates(params, updates)

        metrics["loss"] = loss
        return params, static, opt_state, metrics

    return train_step


def main(config: Config) -> None:
    devices = jax.local_devices()
    num_devices = len(devices)

    key = jax.random.PRNGKey(config.seed)
    model_key, train_key = jax.random.split(key)

    train_dataset = ARCDataset(
        root=Path(config.data_root),
        split=config.train_split,
        subset="train",
        max_size=config.image_size,
        task_lookup=None,
        translation_enabled=True,
        resolution_enabled=True,
        fix_scale_factor=2,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        collate_fn=collate_batch,
    )

    model = ARCViT(
        num_tasks=train_dataset.num_tasks,
        image_size=config.image_size,
        num_colors=config.num_colors,
        embed_dim=config.embed_dim,
        depth=config.depth,
        num_heads=config.num_heads,
        mlp_dim=config.mlp_dim,
        dropout=config.dropout,
        num_task_tokens=config.num_task_tokens,
        patch_size=config.patch_size,
        key=model_key,
    )

    params, static = eqx.partition(model, eqx.is_array)

    steps_per_epoch = len(train_loader)
    total_steps = config.epochs * steps_per_epoch
    warmup_epochs = min(config.epochs // 5, 10)
    warmup_steps = warmup_epochs * steps_per_epoch

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=0.0,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adamw(learning_rate=lr_schedule, weight_decay=config.weight_decay),
    )
    opt_state = optimizer.init(params)

    params = jax.device_put_replicated(params, devices)
    static = jax.device_put_replicated(static, devices)
    opt_state = jax.device_put_replicated(opt_state, devices)

    train_step = make_train_step(optimizer)
    p_train_step = jax.pmap(train_step, axis_name="devices")

    wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        config=asdict(config),
    )
    wandb.log(
        {
            "num_tasks": train_dataset.num_tasks,
            "num_examples": len(train_dataset),
            "batches_per_epoch": len(train_loader),
            "total_steps": len(train_loader) * config.epochs,
        }
    )

    global_step = 0
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()

        for step, batch_torch in enumerate(train_loader):
            batch = _to_jax(batch_torch)
            shard = _shard_batch(batch, num_devices=num_devices)

            train_key, step_key = jax.random.split(train_key)
            device_keys = jax.random.split(step_key, num_devices)

            params, static, opt_state, metrics = p_train_step(
                params, static, opt_state, shard, device_keys
            )

            current_lr = lr_schedule(global_step)
            host_metrics = jax.tree_util.tree_map(lambda x: float(x[0]), metrics)
            global_step += 1

            if step % config.log_every == 0:
                wandb.log(
                    {
                        "train/loss": host_metrics["loss"],
                        "train/pixel_acc": host_metrics["pixel_acc"],
                        "train/exact_acc": host_metrics["exact_acc"],
                        "train/lr": float(current_lr),
                        "epoch": epoch,
                        "global_step": global_step,
                    },
                    step=global_step,
                )

        epoch_time = time.time() - epoch_start
        wandb.log(
            {
                "epoch": epoch,
                "epoch_time": epoch_time,
            },
            step=global_step,
        )

    wandb.finish()


if __name__ == "__main__":
    main(tyro.cli(Config))
