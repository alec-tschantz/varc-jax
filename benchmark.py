import time
from dataclasses import dataclass, field
from statistics import mean
from typing import Dict, Iterable, Tuple

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import tyro

from train import Config, _shard_batch, loss_fn
from varc import ARCViT, Dataset, make_augment_batch


def _block_until_ready(tree):
    jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else None,
        tree,
    )


def _summary(times: Iterable[float]) -> Dict[str, float]:
    times = list(times)
    return {
        "mean_ms": mean(times) * 1e3,
        "min_ms": min(times) * 1e3,
        "max_ms": max(times) * 1e3,
    }


@dataclass
class BenchmarkArgs:
    runs: int = 10
    warmup: int = 10
    config: Config = field(default_factory=Config)


def benchmark_dataset_iter(dataset: Dataset) -> Tuple[Dict, float]:
    start = time.perf_counter()
    batch = next(iter(dataset))
    end = time.perf_counter()
    return batch, (end - start) * 1e3


def benchmark_augment(
    p_augment, batch, key, num_devices: int, warmup: int, runs: int
) -> Dict[str, float]:
    warm_steps = max(warmup, 1)
    for i in range(warm_steps):
        warm_key = jax.random.fold_in(key, runs + i + 1)
        _block_until_ready(p_augment(jax.random.split(warm_key, num_devices), batch))

    times = []
    for i in range(runs):
        step_key = jax.random.fold_in(key, i)
        keys = jax.random.split(step_key, num_devices)
        t0 = time.perf_counter()
        out = p_augment(keys, batch)
        _block_until_ready(out)
        times.append(time.perf_counter() - t0)

    return _summary(times)


def benchmark_forward(
    p_forward, params, static, batch, key, num_devices: int, runs: int, warmup: int
) -> Dict[str, float]:
    warm_steps = max(warmup, 1)
    for i in range(warm_steps):
        warm_key = jax.random.fold_in(key, runs + i + 1)
        _block_until_ready(p_forward(params, static, batch, jax.random.split(warm_key, num_devices)))

    times = []
    for i in range(runs):
        step_key = jax.random.fold_in(key, i)
        model_keys = jax.random.split(step_key, num_devices)
        t0 = time.perf_counter()
        out = p_forward(params, static, batch, model_keys)
        _block_until_ready(out)
        times.append(time.perf_counter() - t0)

    return _summary(times)


def benchmark_forward_backward(
    p_loss_and_grads,
    params,
    static,
    batch,
    key,
    num_devices: int,
    runs: int,
    warmup: int,
) -> Dict[str, float]:
    warm_steps = max(warmup, 1)
    for i in range(warm_steps):
        warm_key = jax.random.fold_in(key, runs + i + 1)
        _block_until_ready(p_loss_and_grads(params, static, batch, jax.random.split(warm_key, num_devices)))

    times = []
    for i in range(runs):
        step_key = jax.random.fold_in(key, i)
        model_keys = jax.random.split(step_key, num_devices)
        t0 = time.perf_counter()
        out = p_loss_and_grads(params, static, batch, model_keys)
        _block_until_ready(out)
        times.append(time.perf_counter() - t0)

    return _summary(times)


def build_model(config: Config, num_tasks: int, model_key: jax.Array) -> ARCViT:
    forward_dtype = getattr(jnp, config.dtype)
    return ARCViT(
        num_tasks=num_tasks,
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


def main(args: BenchmarkArgs) -> None:
    cfg = args.config
    devices = jax.local_devices()
    num_devices = len(devices)
    print(f"Using {num_devices} devices")

    dataset = Dataset(
        path=Path(cfg.data_path),
        split="training",
        subset="train",
        max_size=cfg.image_size,
        extra_train_path=Path(cfg.rearc_path) if cfg.rearc_path else None,
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    batch, dataset_iter_ms = benchmark_dataset_iter(dataset)
    assert cfg.batch_size % num_devices == 0, "batch_size must divide num_devices"

    shard = _shard_batch(batch, num_devices)
    shard = jax.device_put(shard)

    aug_fn = make_augment_batch(
        max_size=cfg.image_size,
        resolution_enabled=True,
        translation_enabled=True,
        fix_scale_factor=2,
    )
    p_augment = jax.pmap(aug_fn, axis_name="devices")

    key = jax.random.PRNGKey(cfg.seed)
    augment_stats = benchmark_augment(
        p_augment, shard, key, num_devices, warmup=args.warmup, runs=args.runs
    )
    augmented = p_augment(jax.random.split(key, num_devices), shard)

    model_key, fwd_key, bwd_key = jax.random.split(key, 3)
    model = build_model(cfg, dataset.num_tasks, model_key)

    params, static = eqx.partition(model, eqx.is_array)
    params = jax.device_put_replicated(params, devices)
    static = jax.device_put_replicated(static, devices)

    def forward_fn(p, s, b, k):
        model_local = eqx.combine(p, s)
        return model_local(
            b["inputs"],
            b["task_ids"],
            attention_mask=b["attention_mask"],
            key=k,
            inference=False,
        )

    p_forward = jax.pmap(forward_fn, axis_name="devices")

    def loss_and_grad_pmapped(p, s, b, k):
        def loss_only(pp):
            model_local = eqx.combine(pp, s)
            loss, _ = loss_fn(model_local, b, k, inference=False)
            return loss

        loss, grads = eqx.filter_value_and_grad(loss_only)(p)
        loss = jax.lax.pmean(loss, axis_name="devices")
        grads = jax.lax.pmean(grads, axis_name="devices")
        return loss, grads

    p_loss_and_grads = jax.pmap(loss_and_grad_pmapped, axis_name="devices")

    forward_stats = benchmark_forward(
        p_forward,
        params,
        static,
        augmented,
        fwd_key,
        num_devices,
        runs=args.runs,
        warmup=args.warmup,
    )

    backward_stats = benchmark_forward_backward(
        p_loss_and_grads,
        params,
        static,
        augmented,
        bwd_key,
        num_devices,
        runs=args.runs,
        warmup=args.warmup,
    )

    print("Dataset iteration (first batch): {:.3f} ms".format(dataset_iter_ms))
    print(f"Augment (pmap): {augment_stats}")
    print(f"Forward only (pmap): {forward_stats}")
    print(f"Forward + backward (pmap): {backward_stats}")


if __name__ == "__main__":
    main(tyro.cli(BenchmarkArgs))
