import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
        max_size: int = 32,
        task_lookup: Optional[Dict[str, int]] = None,
        *,
        translation_enabled: bool = True,
        resolution_enabled: bool = True,
        fix_scale_factor: int = 2,
        extra_train_path: Optional[Path] = None,
        seed: int = 0,
        batch_size: int = 256,
        shuffle: bool = True,
    ) -> None:
        self.rng = random.Random(seed)
        self.path = Path(path)
        self.max_size = max_size
        self.subset = subset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.translation_enabled = translation_enabled
        self.resolution_enabled = resolution_enabled
        self.fix_scale_factor = fix_scale_factor

        self.samples: List[Dict[str, object]] = []
        self.task_lookup: Dict[str, int] = (
            dict(task_lookup) if task_lookup is not None else {}
        )

        self._load_data(self.path / split)

        if self.subset == "train" and extra_train_path:
            self._load_data(Path(extra_train_path))

        self.num_tasks = len(self.task_lookup)

    def _load_data(self, directory: Path) -> None:
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
                max_cur_y = len(example["input"])
                max_cur_x = len(example["input"][0])
                if "output" in example:
                    max_cur_y = max(max_cur_y, len(example["output"]))
                    max_cur_x = max(max_cur_x, len(example["output"][0]))
                if max_cur_y > MAX_SIZE or max_cur_x > MAX_SIZE:
                    continue

                self.samples.append(
                    {
                        "example": example,
                        "task_index": task_index,
                        "task_name": task_name,
                        "example_index": example_index,
                    }
                )

    def __len__(self) -> int:
        return len(self.samples) // self.batch_size

    def __iter__(self):
        indices = list(range(len(self.samples)))
        if self.shuffle:
            self.rng.shuffle(indices)

        usable = len(indices) - (len(indices) % self.batch_size)
        for start in range(0, usable, self.batch_size):
            batch_indices = indices[start : start + self.batch_size]
            batch_items = [self._process_index(i) for i in batch_indices]
            yield self._stack_batch(batch_items)

    def _process_index(self, idx: int) -> Dict[str, np.ndarray]:
        cur_batch = self.samples[idx]
        return self._process_example(
            example=cur_batch["example"],
            task_index=cur_batch["task_index"],
            task_name=cur_batch["task_name"],
            example_index=cur_batch["example_index"],
            rng=self.rng,
        )

    def _process_example(self, example, task_index, task_name, example_index, rng):
        max_cur_y = len(example["input"])
        max_cur_x = len(example["input"][0])
        if "output" in example:
            max_cur_y = max(max_cur_y, len(example["output"]))
            max_cur_x = max(max_cur_x, len(example["output"][0]))
        max_img_size = self.max_size - 2
        max_size = self.max_size

        if self.resolution_enabled:
            example, scale_factor = resolution_augmentation(
                example, max_cur_x, max_cur_y, rng, img_size=max_img_size
            )
        else:
            scale_factor = self.fix_scale_factor
            new_example = {}
            new_example["input"] = np.repeat(
                np.repeat(example["input"], scale_factor, axis=0), scale_factor, axis=1
            ).tolist()
            new_example["output"] = np.repeat(
                np.repeat(example["output"], scale_factor, axis=0),
                scale_factor,
                axis=1,
            ).tolist()
            example = new_example

        max_cur_x = max_cur_x * scale_factor
        max_cur_y = max_cur_y * scale_factor

        if self.translation_enabled:
            x_offset = (
                rng.randint(1, max_img_size - max_cur_x)
                if max_img_size > max_cur_x
                else 1
            )
            y_offset = (
                rng.randint(1, max_img_size - max_cur_y)
                if max_img_size > max_cur_y
                else 1
            )
        else:
            x_offset = 1
            y_offset = 1

        input_grid, input_mask, _, _ = pad_grid_with_translation(
            example["input"], max_size, x_offset, y_offset, output_shape=False
        )

        if "output" in example:
            target_grid, target_mask, target_h, target_w = pad_grid_with_translation(
                example["output"], max_size, x_offset, y_offset, output_shape=True
            )
        else:
            target_grid = np.full((max_size, max_size), IGNORE_INDEX, dtype=np.int32)
            target_mask = np.zeros((max_size, max_size), dtype=np.int32)
            target_h = 0
            target_w = 0

        target_grid = target_grid.copy()
        target_grid[target_mask == 0] = IGNORE_INDEX

        return {
            "inputs": input_grid.astype(np.int32),
            "attention_mask": input_mask.astype(np.bool_),
            "targets": target_grid.astype(np.int32),
            "task_ids": np.array(task_index, dtype=np.int32),
            "task_name": task_name,
            "example_index": np.array(example_index, dtype=np.int32),
            "target_shape": np.array([target_h, target_w], dtype=np.int32),
        }

    def _stack_batch(
        self, batch_items: List[Dict[str, np.ndarray]]
    ) -> Dict[str, jnp.ndarray]:
        inputs = np.stack([item["inputs"] for item in batch_items], axis=0)
        attention = np.stack([item["attention_mask"] for item in batch_items], axis=0)
        targets = np.stack([item["targets"] for item in batch_items], axis=0)
        task_ids = np.stack([item["task_ids"] for item in batch_items], axis=0)

        return {
            "inputs": jnp.asarray(inputs, dtype=jnp.int32),
            "attention_mask": jnp.asarray(attention, dtype=jnp.bool_),
            "targets": jnp.asarray(targets, dtype=jnp.int32),
            "task_ids": jnp.asarray(task_ids, dtype=jnp.int32),
        }


def pad_grid_with_translation(
    grid: List[List[int]],
    max_size: int,
    x_offset: int,
    y_offset: int,
    output_shape: bool = True,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0

    tensor = np.full((max_size, max_size), IGNORE_INDEX, dtype=np.int32)
    mask = np.zeros((max_size, max_size), dtype=np.int32)

    values = np.array(grid, dtype=np.int32)
    tensor[y_offset : y_offset + height, x_offset : x_offset + width] = values
    mask[y_offset : y_offset + height, x_offset : x_offset + width] = 1

    if output_shape:
        tensor[y_offset : y_offset + height, x_offset + width] = PAD_INDEX
        tensor[y_offset + height, x_offset : x_offset + width + 1] = PAD_INDEX
        mask[y_offset : y_offset + height + 1, x_offset : x_offset + width + 1] = 1
    return tensor, mask, height, width


def resolution_augmentation(example, max_cur_x, max_cur_y, rng, img_size=60):
    max_len = max(max_cur_x, max_cur_y)
    max_scale_factor = img_size // max_len
    scale_factor = rng.randint(1, max_scale_factor)
    new_example = {}
    new_example["input"] = np.repeat(
        np.repeat(example["input"], scale_factor, axis=0), scale_factor, axis=1
    ).tolist()
    new_example["output"] = np.repeat(
        np.repeat(example["output"], scale_factor, axis=0), scale_factor, axis=1
    ).tolist()
    return new_example, scale_factor
