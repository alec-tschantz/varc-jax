import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

IGNORE_INDEX = 10
PAD_INDEX = 11
MAX_SIZE = 30


class ARCDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        subset: str = "train",
        max_size: int = 32,
        task_lookup: Optional[Dict[str, int]] = None,
        *,
        translation_enabled: bool = True,
        resolution_enabled: bool = True,
        fix_scale_factor: int = 2,
    ) -> None:
        self.rng = random.Random(42)
        self.root = Path(root)
        self.max_size = max_size
        self.subset = subset

        self.translation_enabled = translation_enabled
        self.resolution_enabled = resolution_enabled
        self.fix_scale_factor = fix_scale_factor

        self.samples: List[Dict[str, torch.Tensor]] = []
        self.task_lookup: Dict[str, int] = (
            dict(task_lookup) if task_lookup is not None else {}
        )

        split_dir = self.root / split
        files = sorted(split_dir.glob("*.json"))
        examples_key = "train" if subset == "train" else "test"

        for file_path in files:
            task_name = file_path.stem
            if task_lookup is None:
                task_index = self.task_lookup.setdefault(
                    task_name, len(self.task_lookup)
                )
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

        self.num_tasks = len(self.task_lookup)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        cur_batch = self.samples[idx]
        return self.process_per_example(
            example=cur_batch["example"],
            task_index=cur_batch["task_index"],
            task_name=cur_batch["task_name"],
            example_index=cur_batch["example_index"],
            rng=self.rng,
            if_translation=self.translation_enabled,
        )

    def _get_or_add_task_index(self, task_name: str) -> int:
        if task_name in self.task_lookup:
            return self.task_lookup[task_name]
        task_index = len(self.task_lookup)
        self.task_lookup[task_name] = task_index
        return task_index

    def process_per_example(
        self, example, task_index, task_name, example_index, rng, if_translation=True
    ):
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
                np.repeat(example["output"], scale_factor, axis=0), scale_factor, axis=1
            ).tolist()
            example = new_example

        max_cur_x = max_cur_x * scale_factor
        max_cur_y = max_cur_y * scale_factor

        if if_translation:
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
            target_grid = torch.full(
                (max_size, max_size), IGNORE_INDEX, dtype=torch.long
            )
            target_mask = torch.zeros((max_size, max_size), dtype=torch.long)
            target_h = 0
            target_w = 0

        target_grid = target_grid.clone()
        target_grid[target_mask == 0] = IGNORE_INDEX

        raw_input = example.get("input", [])
        raw_output = example.get("output") if "output" in example else None

        return {
            "inputs": input_grid,
            "attention_mask": input_mask,
            "targets": target_grid,
            "task_id": torch.tensor(task_index, dtype=torch.long),
            "task_name": task_name,
            "example_index": torch.tensor(example_index, dtype=torch.long),
            "target_shape": torch.tensor([target_h, target_w], dtype=torch.long),
            "raw_input": raw_input,
            "raw_output": raw_output,
            "offset": (x_offset, y_offset),
            "scale_factor": scale_factor,
        }


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    inputs = torch.stack([item["inputs"] for item in batch], dim=0)
    attention = torch.stack([item["attention_mask"] for item in batch], dim=0)
    targets = torch.stack([item["targets"] for item in batch], dim=0)
    task_ids = torch.stack([item["task_id"] for item in batch], dim=0)
    target_shapes = torch.stack([item["target_shape"] for item in batch], dim=0)
    example_indices = torch.stack([item["example_index"] for item in batch], dim=0)
    offset = torch.stack([torch.tensor(item["offset"]) for item in batch], dim=0)
    scale_factors = torch.stack(
        [torch.tensor(item["scale_factor"]) for item in batch], dim=0
    )
    task_names = [item["task_name"] for item in batch]
    raw_inputs = [item["raw_input"] for item in batch]
    raw_outputs = [item["raw_output"] for item in batch]
    return {
        "inputs": inputs,
        "attention_mask": attention,
        "targets": targets,
        "task_ids": task_ids,
        "target_shapes": target_shapes,
        "example_indices": example_indices,
        "task_names": task_names,
        "raw_inputs": raw_inputs,
        "raw_outputs": raw_outputs,
        "offset": offset,
        "scale_factors": scale_factors,
    }


def pad_grid_with_translation(
    grid: List[List[int]],
    max_size: int,
    x_offset: int,
    y_offset: int,
    output_shape: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0

    tensor = torch.full((max_size, max_size), IGNORE_INDEX, dtype=torch.long)
    mask = torch.zeros((max_size, max_size), dtype=torch.long)

    values = torch.tensor(grid, dtype=torch.long)
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
