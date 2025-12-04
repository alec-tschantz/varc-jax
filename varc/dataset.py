import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np

IGNORE_LABEL_ID = -100


@dataclass
class DatasetMetadata:
    pad_id: int
    ignore_label_id: Optional[int]
    blank_identifier_id: int
    vocab_size: int
    seq_len: int
    num_puzzle_identifiers: int
    total_groups: int
    mean_puzzle_examples: float
    total_puzzles: int
    sets: List[str]


@dataclass
class DatasetConfig:
    seed: int
    dataset_path: str
    global_batch_size: int


class Dataset:
    def __init__(self, config: DatasetConfig, split: str = "train"):
        self.config = config
        self.split = split

        metadata = self._load_metadata(config.dataset_path)
        image_size = int(math.isqrt(metadata.seq_len))

        self.image_size = image_size
        self.metadata = metadata

        self._inputs: Optional[np.ndarray] = None
        self._labels: Optional[np.ndarray] = None
        self._task_ids: Optional[np.ndarray] = None

        self._iters = 0

    def _load_metadata(self, dataset_path: str) -> DatasetMetadata:
        with open(os.path.join(dataset_path, self.split, "dataset.json"), "r") as f:
            return DatasetMetadata(**json.load(f))

    def _lazy_load_dataset(self) -> None:
        if self._inputs is not None:
            return

        all_inputs = []
        all_labels = []
        all_task_ids = []

        for set_name in self.metadata.sets:
            path_prefix = os.path.join(
                self.config.dataset_path, self.split, f"{set_name}__"
            )

            inputs = np.load(path_prefix + "inputs.npy", mmap_mode="r")
            labels = np.load(path_prefix + "labels.npy", mmap_mode="r")

            puzzle_identifiers = np.load(path_prefix + "puzzle_identifiers.npy")
            puzzle_indices = np.load(path_prefix + "puzzle_indices.npy")

            num_examples = inputs.shape[0]
            task_ids_expanded = np.zeros(num_examples, dtype=np.int32)

            for i in range(len(puzzle_identifiers)):
                start = puzzle_indices[i]
                end = puzzle_indices[i + 1]
                pid = puzzle_identifiers[i]
                task_ids_expanded[start:end] = pid

            all_inputs.append(inputs)
            all_labels.append(labels)
            all_task_ids.append(task_ids_expanded)

        self._inputs = np.concatenate(all_inputs, axis=0)
        self._labels = np.concatenate(all_labels, axis=0)
        self._task_ids = np.concatenate(all_task_ids, axis=0)

    def __len__(self):
        self._lazy_load_dataset()
        return self._inputs.shape[0]

    def __iter__(self):
        self._lazy_load_dataset()

        self._iters += 1
        rng = np.random.Generator(np.random.Philox(seed=self.config.seed + self._iters))

        num_samples = self._inputs.shape[0]
        indices = rng.permutation(num_samples)

        batch_size = self.config.global_batch_size

        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i : i + batch_size]

            current_batch_size = batch_indices.size
            if current_batch_size < batch_size:
                continue

            batch_inputs = self._inputs[batch_indices]
            batch_labels = self._labels[batch_indices]
            batch_task_ids = self._task_ids[batch_indices]

            formatted_batch = self._format_batch(
                batch_inputs, batch_labels, batch_task_ids
            )

            yield formatted_batch

    def _format_batch(
        self, inputs: np.ndarray, labels: np.ndarray, task_ids: np.ndarray
    ) -> Dict[str, jnp.ndarray]:
        inputs = inputs.astype(np.int32)
        labels = labels.astype(np.int32)
        task_ids = task_ids.astype(np.int32)

        attention_mask = inputs != self.metadata.pad_id

        if self.metadata.ignore_label_id is not None:
            label_pad_mask = labels != self.metadata.ignore_label_id
            labels = labels.copy()
            labels[~label_pad_mask] = IGNORE_LABEL_ID

        batch_size = labels.shape[0]
        inputs = inputs.reshape(batch_size, self.image_size, self.image_size)
        labels = labels.reshape(batch_size, self.image_size, self.image_size)
        attention_mask = attention_mask.reshape(
            batch_size, self.image_size, self.image_size
        )

        return {
            "inputs": jnp.array(inputs, dtype=jnp.int32),
            "labels": jnp.array(labels, dtype=jnp.int32),
            "attention_mask": jnp.array(attention_mask, dtype=jnp.bool_),
            "task_ids": jnp.array(task_ids, dtype=jnp.int32),
        }
