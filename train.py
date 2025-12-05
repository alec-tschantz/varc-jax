import os
import time
from pathlib import Path
from functools import partial
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import tyro
import wandb

from varc import ARCViT, Dataset, IGNORE_INDEX, augment_example


@dataclass
class Config:
    data_path: str = "data/arc"
    rearc_path: Optional[str] = "data/rearc"
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
    dtype: str = "bfloat16"
    seed: int = 0
    log_every: int = 10
    wandb_project: str = "varc-jax"
    wandb_run_name: Optional[str] = None
    max_grad_norm: float = 1.0

    image_size: int = 64
    num_colors: int = 12


def loss_fn(
    model: ARCViT,
    batch: Dict[str, jax.Array],
    key: jax.Array,
    inference: bool,
) -> Tuple[jax.Array, Dict[str, jax.Array]]:

    logits = model(
        batch["inputs"],
        batch["task_ids"],
        attention_mask=batch["attention_mask"],
        key=key,
        inference=inference,
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

    metrics = {"loss": loss, "pixel_acc": pixel_acc, "exact_acc": exact_acc}
    return loss, metrics


def make_train_step(optimizer: optax.GradientTransformation, config: Config):
    train_augment_batch = make_augment_batch(
        max_size=config.image_size,
        resolution_enabled=True,
        translation_enabled=True,
        fix_scale_factor=2,
    )

    def train_step(
        params: ARCViT,
        static: ARCViT,
        opt_state: optax.OptState,
        raw_batch: Dict[str, jax.Array],
        key: jax.Array,
    ) -> Tuple[ARCViT, ARCViT, optax.OptState, Dict[str, jax.Array]]:

        aug_key, model_key = jax.random.split(key)
        batch = train_augment_batch(aug_key, raw_batch)

        def compute_loss(p):
            return loss_fn(eqx.combine(p, static), batch, model_key, inference=False)

        grad_fn = eqx.filter_value_and_grad(compute_loss, has_aux=True)
        (loss, metrics), grads = grad_fn(params)

        grads = jax.lax.pmean(grads, axis_name="devices")
        loss = jax.lax.pmean(loss, axis_name="devices")
        metrics = jax.tree_util.tree_map(lambda x: jax.lax.pmean(x, "devices"), metrics)

        updates, opt_state = optimizer.update(grads, opt_state, params=params)
        params = eqx.apply_updates(params, updates)

        metrics["loss"] = loss
        return params, static, opt_state, metrics

    return train_step


def make_eval_step(config: Config):
    eval_augment_batch = make_augment_batch(
        max_size=config.image_size,
        resolution_enabled=False,
        translation_enabled=False,
        fix_scale_factor=2,
    )

    def eval_step(
        params: ARCViT,
        static: ARCViT,
        raw_batch: Dict[str, jax.Array],
        key: jax.Array,
    ) -> Dict[str, jax.Array]:
        aug_key, model_key = jax.random.split(key)
        batch = eval_augment_batch(aug_key, raw_batch)
        _, metrics = loss_fn(
            eqx.combine(params, static), batch, model_key, inference=True
        )
        return jax.tree.map(lambda x: jax.lax.pmean(x, "devices"), metrics)

    return eval_step


def evaluate_model(
    params: ARCViT,
    static: ARCViT,
    eval_dataset,
    p_eval_step,
    eval_key: jax.Array,
    num_devices: int,
) -> Dict[str, float]:
    metrics_sum = None

    for batch in eval_dataset:
        shard = shard_batch(batch, num_devices)

        eval_key, step_key = jax.random.split(eval_key)
        device_keys = jax.random.split(step_key, num_devices)

        metrics = p_eval_step(params, static, shard, device_keys)
        host_metrics = jax.tree_util.tree_map(lambda x: float(x[0]), metrics)

        metrics_sum = (
            jax.tree_util.tree_map(lambda a, b: a + b, metrics_sum, host_metrics)
            if metrics_sum is not None
            else host_metrics
        )

    return jax.tree_util.tree_map(lambda x: x / len(eval_dataset), metrics_sum)


def make_augment_batch(
    *,
    max_size: int = 64,
    resolution_enabled: bool = True,
    translation_enabled: bool = True,
    fix_scale_factor: int = 2,
) -> Callable[[jax.Array, Dict[str, jax.Array]], Dict[str, jax.Array]]:
    augment_fn = partial(
        augment_example,
        max_size=max_size,
        resolution_enabled=resolution_enabled,
        translation_enabled=translation_enabled,
        fix_scale_factor=fix_scale_factor,
    )

    vmap_augment = jax.vmap(augment_fn, in_axes=(0, 0, 0, 0, 0))

    def augment_batch(
        key: jax.Array,
        batch: Dict[str, jax.Array],
    ) -> Dict[str, jax.Array]:
        keys = jax.random.split(key, batch["inputs"].shape[0])

        aug_out = vmap_augment(
            keys,
            batch["inputs"],
            batch["targets"],
            batch["input_shapes"],
            batch["target_shapes"],
        )

        return {
            "inputs": aug_out["inputs"],
            "attention_mask": aug_out["attention_mask"],
            "targets": aug_out["targets"],
            "task_ids": batch["task_ids"],
            "example_index": batch["example_index"],
            "target_shape": aug_out["target_shape"],
        }

    return augment_batch


def create_datasets(config: Config):
    train_dataset = Dataset(
        path=Path(config.data_path),
        extra_train_path=Path(config.rearc_path) if config.rearc_path else None,
        split="training",
        subset="train",
        max_size=config.image_size,
        task_lookup=None,
        batch_size=config.batch_size,
    )

    eval_dataset = Dataset(
        path=Path(config.data_path),
        split="training",
        subset="test",
        max_size=config.image_size,
        task_lookup=train_dataset.task_lookup,
        batch_size=config.batch_size,
    )

    return train_dataset, eval_dataset


def shard_batch(batch: Dict[str, jax.Array], num_devices: int) -> Dict[str, jax.Array]:
    def _reshape(x):
        return x.reshape(num_devices, x.shape[0] // num_devices, *x.shape[1:])

    return jax.tree_util.tree_map(_reshape, batch)


def main(config: Config) -> None:
    devices = jax.local_devices()
    num_devices = len(devices)

    key = jax.random.PRNGKey(config.seed)
    model_key, train_key, eval_key = jax.random.split(key, 3)

    train_dataset, eval_dataset = create_datasets(config)

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
        dtype=getattr(jnp, config.dtype),
        key=model_key,
    )

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.learning_rate,
        warmup_steps=min(config.epochs // 5, 10) * len(train_dataset),
        decay_steps=config.epochs * len(train_dataset),
        end_value=0.0,
    )

    params, static = eqx.partition(model, eqx.is_array)
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adamw(learning_rate=lr_schedule, weight_decay=config.weight_decay),
    )
    opt_state = optimizer.init(params)

    params = jax.device_put_replicated(params, devices)
    static = jax.device_put_replicated(static, devices)
    opt_state = jax.device_put_replicated(opt_state, devices)

    train_step = make_train_step(optimizer, config)
    p_train_step = jax.pmap(train_step, axis_name="devices")

    eval_step = make_eval_step(config)
    p_eval_step = jax.pmap(eval_step, axis_name="devices")

    wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        config=asdict(config),
    )

    wandb.log(
        {
            "data/num_train_tasks": train_dataset.num_tasks,
            "data/num_train_examples": train_dataset.num_samples,
            "data/train_batches_per_epoch": len(train_dataset),
            "data/train_total_steps": len(train_dataset) * config.epochs,
            "data/num_eval_tasks": eval_dataset.num_tasks,
            "data/num_eval_examples": eval_dataset.num_samples,
            "data/eval_batches_per_epoch": len(eval_dataset),
        }
    )

    global_step = 0
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()

        for step, batch in enumerate(train_dataset):
            shard = shard_batch(batch, num_devices=num_devices)

            train_key, step_key = jax.random.split(train_key)
            device_keys = jax.random.split(step_key, num_devices)

            params, static, opt_state, metrics = p_train_step(
                params, static, opt_state, shard, device_keys
            )
            global_step += 1

            if step % config.log_every == 0:
                host_metrics = jax.tree_util.tree_map(lambda x: float(x[0]), metrics)
                wandb.log(
                    {
                        "train/loss": host_metrics["loss"],
                        "train/pixel_acc": host_metrics["pixel_acc"],
                        "train/exact_acc": host_metrics["exact_acc"],
                        "train/lr": float(lr_schedule(global_step)),
                        "epoch": epoch,
                        "global_step": global_step,
                    },
                    step=global_step,
                )

        epoch_time = time.time() - epoch_start

        eval_key, epoch_key = jax.random.split(eval_key)
        eval_metrics = evaluate_model(
            params,
            static,
            eval_dataset,
            p_eval_step,
            epoch_key,
            num_devices,
        )

        os.makedirs("checkpoints", exist_ok=True)
        params_host = jax.tree.map(lambda x: x[0], params)
        ckpt_path = os.path.join("checkpoints", f"{config.wandb_project}.eqx")
        eqx.tree_serialise_leaves(ckpt_path, eqx.combine(params_host, static))

        wandb.log(
            {
                "epoch": epoch,
                "epoch_time": epoch_time,
                "global_step": global_step,
                **{f"eval/{k}": v for k, v in eval_metrics.items()},
            },
            step=global_step,
        )

    wandb.finish()


if __name__ == "__main__":
    main(tyro.cli(Config))
