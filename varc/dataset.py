import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np

IGNORE_INDEX = 10
PAD_INDEX = 11
MAX_SIZE = 30


class Dataset:
    def __init__(
        self,
        path: Path,
        split: str,
        subset: str = "train",
        max_size: int = 64,
        task_lookup: Optional[Dict[str, int]] = None,
        *,
        extra_train_path: Optional[Path] = None,
        seed: int = 0,
        batch_size: int = 256,
        shuffle: bool = True,
    ) -> None:
        self.path = Path(path)
        self.max_size = max_size
        self.subset = subset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

        self.task_lookup: Dict[str, int] = (
            dict(task_lookup) if task_lookup is not None else {}
        )

        input_list = []
        target_list = []
        input_shape_list = []
        target_shape_list = []
        task_id_list = []
        example_idx_list = []
        task_names_list = []

        def _process_dir(directory: Path):
            files = sorted(directory.glob("*.json"))
            examples_key = "train" if self.subset == "train" else "test"

            for file_path in files:
                task_name = file_path.stem
                if self.subset == "train":
                    if task_name not in self.task_lookup:
                        self.task_lookup[task_name] = len(self.task_lookup)
                    task_index = self.task_lookup[task_name]
                else:
                    if task_name not in self.task_lookup:
                        continue
                    task_index = self.task_lookup[task_name]

                with file_path.open("r") as fh:
                    task_data = json.load(fh)

                examples = task_data.get(examples_key, [])

                for example_index, example in enumerate(examples):
                    inp = example["input"]
                    out = example["output"]

                    h_in = len(inp)
                    w_in = len(inp[0])
                    h_out = len(out)
                    w_out = len(out[0])

                    max_cur_y = max(h_in, h_out)
                    max_cur_x = max(w_in, w_out)

                    if max_cur_y > MAX_SIZE or max_cur_x > MAX_SIZE:
                        continue

                    input_list.append(_pad_to_max(inp, MAX_SIZE, MAX_SIZE))
                    target_list.append(_pad_to_max(out, MAX_SIZE, MAX_SIZE))

                    input_shape_list.append([h_in, w_in])
                    target_shape_list.append([h_out, w_out])
                    task_id_list.append(task_index)
                    example_idx_list.append(example_index)
                    task_names_list.append(task_name)

        _process_dir(self.path / split)
        if self.subset == "train" and extra_train_path:
            _process_dir(Path(extra_train_path))

        self.inputs = np.stack(input_list)
        self.targets = np.stack(target_list)
        self.input_shapes = np.stack(input_shape_list)
        self.target_shapes = np.stack(target_shape_list)
        self.task_ids = np.array(task_id_list, dtype=np.int32)
        self.example_indices = np.array(example_idx_list, dtype=np.int32)
        self.task_names = np.array(task_names_list)
        self.num_samples = len(self.inputs)
        self.num_tasks = len(self.task_lookup)

    def __len__(self) -> int:
        return self.num_samples // self.batch_size

    def __iter__(self):
        indices = np.arange(self.num_samples)
        if self.shuffle:
            self.rng.shuffle(indices)

        usable = self.num_samples - (self.num_samples % self.batch_size)

        for start in range(0, usable, self.batch_size):
            batch_idx = indices[start : start + self.batch_size]

            yield {
                "inputs": self.inputs[batch_idx],
                "targets": self.targets[batch_idx],
                "input_shapes": self.input_shapes[batch_idx],
                "target_shapes": self.target_shapes[batch_idx],
                "task_ids": self.task_ids[batch_idx],
                "example_index": self.example_indices[batch_idx],
            }


def augment_example(
    key: jax.Array,
    raw_input: jax.Array,
    raw_target: jax.Array,
    raw_input_shape: jax.Array,
    raw_target_shape: jax.Array,
    max_size: int,
    resolution_enabled: bool,
    translation_enabled: bool,
    fix_scale_factor: int,
) -> Dict[str, jax.Array]:
    max_img_size = max_size - 2

    h_in = raw_input_shape[0]
    w_in = raw_input_shape[1]
    h_out = raw_target_shape[0]
    w_out = raw_target_shape[1]

    cur_h = jnp.maximum(h_in, h_out)
    cur_w = jnp.maximum(w_in, w_out)

    key_scale, key_offset = jax.random.split(key)

    if resolution_enabled:
        max_len = jnp.maximum(cur_h, cur_w)
        max_len = jnp.maximum(max_len, 1)
        max_scale_factor = max_img_size // max_len
        max_scale_factor = jnp.maximum(max_scale_factor, 1)
        scale_factor = jax.random.randint(key_scale, (), 1, max_scale_factor + 1)
    else:
        max_len = jnp.maximum(cur_h, cur_w)
        max_len = jnp.maximum(max_len, 1)
        max_scale_factor = max_img_size // max_len
        max_scale_factor = jnp.maximum(max_scale_factor, 1)
        desired = jnp.array(fix_scale_factor, dtype=jnp.int32)
        scale_factor = jnp.minimum(desired, max_scale_factor)

    scaled_h_in = h_in * scale_factor
    scaled_w_in = w_in * scale_factor
    scaled_h_out = h_out * scale_factor
    scaled_w_out = w_out * scale_factor

    y_grid, x_grid = jnp.indices((max_size, max_size))

    if translation_enabled:
        key_y, key_x = jax.random.split(key_offset)
        max_dy = max_img_size - jnp.maximum(scaled_h_in, scaled_h_out)
        max_dx = max_img_size - jnp.maximum(scaled_w_in, scaled_w_out)

        def sample_offset(k, max_d):
            max_d = jnp.maximum(max_d, 0)
            return jax.lax.cond(
                max_d > 0,
                lambda kk: jax.random.randint(kk, (), 1, max_d + 1),
                lambda kk: jnp.array(1, dtype=jnp.int32),
                k,
            )

        y_offset = sample_offset(key_y, max_dy)
        x_offset = sample_offset(key_x, max_dx)
    else:
        y_offset = jnp.array(1, dtype=jnp.int32)
        x_offset = jnp.array(1, dtype=jnp.int32)

    y_rel = y_grid - y_offset
    x_rel = x_grid - x_offset

    y_src_in = y_rel // scale_factor
    x_src_in = x_rel // scale_factor

    y_src_out = y_rel // scale_factor
    x_src_out = x_rel // scale_factor

    valid_in = (y_src_in >= 0) & (y_src_in < h_in) & (x_src_in >= 0) & (x_src_in < w_in)
    valid_out = (
        (y_src_out >= 0) & (y_src_out < h_out) & (x_src_out >= 0) & (x_src_out < w_out)
    )

    h_in_safe = jnp.maximum(h_in, 1)
    w_in_safe = jnp.maximum(w_in, 1)
    h_out_safe = jnp.maximum(h_out, 1)
    w_out_safe = jnp.maximum(w_out, 1)

    y_src_in_clipped = jnp.clip(y_src_in, 0, h_in_safe - 1)
    x_src_in_clipped = jnp.clip(x_src_in, 0, w_in_safe - 1)

    y_src_out_clipped = jnp.clip(y_src_out, 0, h_out_safe - 1)
    x_src_out_clipped = jnp.clip(x_src_out, 0, w_out_safe - 1)

    input_vals = raw_input[y_src_in_clipped, x_src_in_clipped]
    input_grid = jnp.where(valid_in, input_vals, IGNORE_INDEX).astype(jnp.int32)
    input_mask = valid_in.astype(jnp.bool_)

    is_border_h = (x_rel == scaled_w_out) & (y_rel >= 0) & (y_rel < scaled_h_out)
    is_border_v = (y_rel == scaled_h_out) & (x_rel >= 0) & (x_rel <= scaled_w_out)
    is_border = is_border_h | is_border_v

    target_vals = raw_target[y_src_out_clipped, x_src_out_clipped]
    target_grid = jnp.where(valid_out, target_vals, IGNORE_INDEX)
    target_grid = jnp.where(is_border, PAD_INDEX, target_grid).astype(jnp.int32)

    final_target_shape = jnp.array([scaled_h_out, scaled_w_out], dtype=jnp.int32)

    return {
        "inputs": input_grid,
        "attention_mask": input_mask,
        "targets": target_grid,
        "target_shape": final_target_shape,
    }


def _pad_to_max(grid: List[List[int]], max_h: int, max_w: int) -> np.ndarray:
    arr = np.array(grid, dtype=np.int32)
    h, w = arr.shape
    padded = np.full((max_h, max_w), IGNORE_INDEX, dtype=np.int32)
    padded[:h, :w] = arr
    return padded
