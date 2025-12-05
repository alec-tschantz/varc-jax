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

from varc import ARCViT, Dataset, IGNORE_INDEX


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
    log_every: int = 50
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


def make_train_step(optimizer: optax.GradientTransformation):
    def train_step(
        params: ARCViT,
        static: ARCViT,
        opt_state: optax.OptState,
        batch: Dict[str, jax.Array],
        key: jax.Array,
    ) -> Tuple[ARCViT, ARCViT, optax.OptState, Dict[str, jax.Array]]:
        def compute_loss(p):
            model = eqx.combine(p, static)
            return loss_fn(model, batch, key, inference=False)

        (loss, metrics), grads = eqx.filter_value_and_grad(compute_loss, has_aux=True)(
            params
        )

        grads = jax.lax.pmean(grads, axis_name="devices")
        loss = jax.lax.pmean(loss, axis_name="devices")
        metrics = jax.tree_util.tree_map(lambda x: jax.lax.pmean(x, "devices"), metrics)

        updates, opt_state = optimizer.update(grads, opt_state, params=params)
        params = eqx.apply_updates(params, updates)

        metrics["loss"] = loss
        return params, static, opt_state, metrics

    return train_step


def eval_step(
    params: ARCViT,
    static: ARCViT,
    batch: Dict[str, jax.Array],
    key: jax.Array,
) -> Dict[str, jax.Array]:
    model = eqx.combine(params, static)
    _, metrics = loss_fn(model, batch, key, inference=True)
    return jax.tree.map(lambda x: jax.lax.pmean(x, "devices"), metrics)


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
        shard = _shard_batch(batch, num_devices)

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


def create_datasets(config: Config):
    train_dataset = Dataset(
        path=Path(config.data_path),
        extra_train_path=Path(config.rearc_path) if config.rearc_path else None,
        split="training",
        subset="train",
        max_size=config.image_size,
        task_lookup=None,
        translation_enabled=True,
        resolution_enabled=True,
        fix_scale_factor=2,
        batch_size=config.batch_size,
        shuffle=True,
    )

    eval_dataset = Dataset(
        path=Path(config.data_path),
        split="training",
        subset="test",
        max_size=config.image_size,
        task_lookup=train_dataset.task_lookup,
        translation_enabled=False,
        resolution_enabled=False,
        fix_scale_factor=2,
        batch_size=config.batch_size,
        shuffle=True,
    )

    return train_dataset, eval_dataset


def _shard_batch(batch: Dict[str, jax.Array], num_devices: int) -> Dict[str, jax.Array]:
    def _reshape(x):
        local_batch_size = x.shape[0] // num_devices
        return x.reshape(num_devices, local_batch_size, *x.shape[1:])

    return jax.tree_util.tree_map(_reshape, batch)


def main(config: Config) -> None:
    devices = jax.local_devices()
    num_devices = len(devices)

    key = jax.random.PRNGKey(config.seed)
    model_key, train_key, eval_key = jax.random.split(key, 3)

    train_dataset, eval_dataset = create_datasets(config)
    forward_dtype = getattr(jnp, config.dtype)

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
        dtype=forward_dtype,
        key=model_key,
    )

    params, static = eqx.partition(model, eqx.is_array)

    steps_per_epoch = len(train_dataset)
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
    p_eval_step = jax.pmap(eval_step, axis_name="devices")

    wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        config=asdict(config),
    )

    dataset_metrics = {
        "data/num_train_tasks": train_dataset.num_tasks,
        "data/num_train_examples": len(train_dataset.samples),
        "data/train_batches_per_epoch": len(train_dataset),
        "data/train_total_steps": len(train_dataset) * config.epochs,
        "data/num_eval_tasks": eval_dataset.num_tasks,
        "data/num_eval_examples": len(eval_dataset.samples),
        "data/eval_batches_per_epoch": len(eval_dataset),
    }
    wandb.log(dataset_metrics)

    global_step = 0
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()

        for step, batch in enumerate(train_dataset):
            shard = _shard_batch(batch, num_devices=num_devices)

            train_key, step_key = jax.random.split(train_key)
            device_keys = jax.random.split(step_key, num_devices)

            params, static, opt_state, metrics = p_train_step(
                params, static, opt_state, shard, device_keys
            )

            current_lr = lr_schedule(global_step)
            global_step += 1

            if step % config.log_every == 0:
                host_metrics = jax.tree_util.tree_map(lambda x: float(x[0]), metrics)
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

        eval_key, epoch_key = jax.random.split(eval_key)
        eval_metrics = evaluate_model(
            params,
            static,
            eval_dataset,
            p_eval_step,
            epoch_key,
            num_devices,
        )

        log_metrics = {
            "epoch": epoch,
            "epoch_time": epoch_time,
            "global_step": global_step,
        }
        for k, v in eval_metrics.items():
            log_metrics[f"eval/{k}"] = v

        wandb.log(log_metrics, step=global_step)

    wandb.finish()


if __name__ == "__main__":
    main(tyro.cli(Config))
