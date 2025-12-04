import argparse
import math
import time
from typing import Dict

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from dataset import Dataset, DatasetConfig, IGNORE_LABEL_ID
from varc import ARCViT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="JAX/Equinox training for VARC.")
    parser.add_argument("--data-root", type=str, default="data/arc1concept-aug-1000")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--epochs-per-iter", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32, help="Global batch size.")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--mlp-dim", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patch-size", type=int, default=1)
    parser.add_argument("--num-task-tokens", type=int, default=1)
    parser.add_argument("--num-colors", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=30)
    return parser.parse_args()


def _prepare_batch(
    batch: Dict[str, np.ndarray], image_size: int
) -> Dict[str, jax.Array]:
    inputs = batch["inputs"].numpy()
    labels = batch["labels"].numpy()
    mask = batch["attention_mask"].numpy().astype(np.bool_)
    task_ids = batch["puzzle_identifiers"].numpy()

    batch_size = inputs.shape[0]
    inputs = inputs.reshape(batch_size, image_size, image_size)
    labels = labels.reshape(batch_size, image_size, image_size)
    mask = mask.reshape(batch_size, image_size, image_size)

    return {
        "inputs": jnp.array(inputs, dtype=jnp.int32),
        "labels": jnp.array(labels, dtype=jnp.int32),
        "attention_mask": jnp.array(mask, dtype=jnp.bool_),
        "task_ids": jnp.array(task_ids, dtype=jnp.int32),
    }


def _shard_batch(batch: Dict[str, jax.Array], num_devices: int) -> Dict[str, jax.Array]:
    def _reshape(x):
        leading = x.shape[0] // num_devices
        return x.reshape(num_devices, leading, *x.shape[1:])

    return jax.tree_util.tree_map(_reshape, batch)


def _assert_no_none(tree, name: str):
    def _check(x):
        if x is None:
            raise ValueError(f"Found None in {name}")
        return x

    return jax.tree_util.tree_map(_check, tree)


def main() -> None:
    args = parse_args()
    devices = jax.local_devices()
    num_devices = len(devices)
    if args.batch_size % num_devices != 0:
        raise ValueError(
            f"Global batch size {args.batch_size} must be divisible by number of devices {num_devices}."
        )

    dataset_config = DatasetConfig(
        seed=args.seed,
        dataset_path=args.data_root,
        global_batch_size=args.batch_size,
        test_set_mode=False,
        epochs_per_iter=args.epochs_per_iter,
    )
    train_dataset = Dataset(dataset_config, split=args.train_split)
    dataset_image_size = int(math.isqrt(train_dataset.metadata.seq_len))
    if dataset_image_size * dataset_image_size != train_dataset.metadata.seq_len:
        raise ValueError(
            "Dataset sequence length must be a perfect square (unpatched pixels)."
        )

    image_size = args.image_size or dataset_image_size
    if image_size != dataset_image_size:
        raise ValueError(
            f"Configured image_size={image_size} but dataset provides {dataset_image_size}x{dataset_image_size} pixels."
        )
    if image_size % args.patch_size != 0:
        raise ValueError("image_size must be divisible by patch_size.")

    key = jax.random.PRNGKey(args.seed)
    model_key, key = jax.random.split(key)

    if args.num_colors != train_dataset.metadata.vocab_size:
        raise ValueError(
            f"num_colors ({args.num_colors}) must match dataset vocab size ({train_dataset.metadata.vocab_size})."
        )

    model = ARCViT(
        num_tasks=train_dataset.metadata.num_puzzle_identifiers,
        image_size=image_size,
        num_colors=train_dataset.metadata.vocab_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        dropout=args.dropout,
        num_task_tokens=args.num_task_tokens,
        patch_size=args.patch_size,
        key=model_key,
    )
    params, static = eqx.partition(model, eqx.is_array)

    optimizer = optax.adamw(
        learning_rate=args.learning_rate, weight_decay=args.weight_decay
    )
    opt_state = optimizer.init(params)

    params = jax.device_put_replicated(params, devices)
    static = jax.device_put_replicated(static, devices)
    opt_state = jax.device_put_replicated(opt_state, devices)

    def loss_and_metrics(model, batch, key):
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
        logits_hw = jnp.transpose(logits, (0, 2, 3, 1))  # (B, H, W, C)
        logits_flat = logits_hw.reshape(-1, logits.shape[1])
        labels_flat = batch["labels"].reshape(-1)
        valid_mask = (labels_flat != IGNORE_LABEL_ID).astype(jnp.float32)
        labels_masked = jnp.where(valid_mask.astype(bool), labels_flat, 0)

        per_elem_loss = optax.softmax_cross_entropy_with_integer_labels(
            logits_flat, labels_masked
        )
        loss_sum = (per_elem_loss * valid_mask).sum()
        denom = jnp.maximum(valid_mask.sum(), 1.0)
        loss = loss_sum / denom

        preds = jnp.argmax(logits, axis=1)
        pred_flat = preds.reshape(-1)
        correct = (pred_flat == labels_flat).astype(jnp.float32) * valid_mask
        pixel_acc = correct.sum() / denom
        exact = jax.vmap(lambda p, l: jnp.all((p == l) | (l == IGNORE_LABEL_ID)))(
            preds, batch["labels"]
        )
        exact_acc = exact.mean()

        metrics = {
            "loss": loss,
            "pixel_acc": pixel_acc,
            "exact_acc": exact_acc,
        }
        return loss, metrics

    def train_step(params, static, opt_state, batch, key):
        def loss_fn(p, batch, subkey):
            mdl = eqx.combine(p, static)
            return loss_and_metrics(mdl, batch, subkey)

        (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            params, batch, key
        )

        grads = jax.lax.pmean(grads, axis_name="devices")
        metrics = jax.tree_util.tree_map(
            lambda x: jax.lax.pmean(x, axis_name="devices"), metrics
        )
        loss = jax.lax.pmean(loss, axis_name="devices")

        _assert_no_none(grads, "grads")
        _assert_no_none(opt_state, "opt_state")

        updates, opt_state = optimizer.update(grads, opt_state, params=params)
        params = eqx.apply_updates(params, updates)

        metrics["loss"] = loss
        return params, static, opt_state, metrics

    p_train_step = jax.pmap(train_step, axis_name="devices")

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        step_in_epoch = 0
        epoch_start = time.time()
        for set_name, raw_batch, _ in train_dataset:
            batch = _prepare_batch(raw_batch, image_size=image_size)
            shard = _shard_batch(batch, num_devices=num_devices)

            step_key, key = jax.random.split(key)
            step_keys = jax.random.split(step_key, num_devices)

            params, static, opt_state, metrics = p_train_step(
                params, static, opt_state, shard, step_keys
            )
            host_metrics = jax.tree_util.tree_map(lambda x: float(x[0]), metrics)

            global_step += 1
            step_in_epoch += 1
            if step_in_epoch % args.log_every == 0:
                print(
                    f"epoch={epoch} step={step_in_epoch} "
                    f"loss={host_metrics['loss']:.4f} "
                    f"pixel_acc={host_metrics['pixel_acc']:.4f} "
                    f"exact_acc={host_metrics['exact_acc']:.4f}"
                )

            if args.max_steps_per_epoch and step_in_epoch >= args.max_steps_per_epoch:
                break

        epoch_time = time.time() - epoch_start
        print(
            f"Finished epoch {epoch} | steps={step_in_epoch} | time={epoch_time:.1f}s"
        )


if __name__ == "__main__":
    main()
