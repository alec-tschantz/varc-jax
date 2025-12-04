import os
import json
from typing import Tuple, List, Dict, Optional
import numpy as np
import pydantic

import torch
from torch.utils.data import IterableDataset, get_worker_info


IGNORE_LABEL_ID = -100


class DatasetMetadata(pydantic.BaseModel):
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


class DatasetConfig(pydantic.BaseModel):
    seed: int
    dataset_path: str
    global_batch_size: int
    test_set_mode: bool
    epochs_per_iter: int


class Dataset(IterableDataset):
    def __init__(self, config: DatasetConfig, split: str = "train"):
        super().__init__()
        self.config = config
        self.split = split

        current_metadata = self._load_metadata(config.dataset_path)
        prev_seq_len = current_metadata.seq_len
        prev_vocab_size = current_metadata.vocab_size
        prev_pad_id = current_metadata.pad_id
        prev_ignore_label_id = current_metadata.ignore_label_id
        prev_blank_identifier_id = current_metadata.blank_identifier_id
        prev_sets = current_metadata.sets
        num_identifiers = current_metadata.num_puzzle_identifiers
        mean_puzzle_examples = current_metadata.mean_puzzle_examples
        total_puzzles = current_metadata.total_puzzles
        total_groups = current_metadata.total_groups

        self.metadata = DatasetMetadata(
            seq_len=prev_seq_len,
            vocab_size=prev_vocab_size,
            pad_id=prev_pad_id,
            ignore_label_id=prev_ignore_label_id,
            blank_identifier_id=prev_blank_identifier_id,
            num_puzzle_identifiers=num_identifiers,
            total_groups=total_groups,
            mean_puzzle_examples=mean_puzzle_examples,
            total_puzzles=total_puzzles,
            sets=prev_sets,
        )

        self._data = None
        self._iters = 0

    def _load_metadata(self, dataset_path) -> DatasetMetadata:
        with open(os.path.join(dataset_path, self.split, "dataset.json"), "r") as f:
            return DatasetMetadata(**json.load(f))

    def _lazy_load_dataset(self):
        if self._data is not None:
            return

        field_mmap_modes = {
            "inputs": "r",
            "labels": "r",
            "puzzle_identifiers": None,
            "puzzle_indices": None,
            "group_indices": None,
        }

        self._data = {}
        for set_name in self.metadata.sets:
            set_data = {
                field_name: np.load(
                    os.path.join(
                        self.config.dataset_path, self.split, f"{set_name}__{field_name}.npy"
                    ),
                    mmap_mode=mmap_mode,
                )
                for field_name, mmap_mode in field_mmap_modes.items()
            }
            puzzle_group_indices = (
                np.searchsorted(
                    set_data["group_indices"],
                    np.arange(set_data["puzzle_identifiers"].shape[0]),
                    side="right",
                )
                - 1
            ).astype(np.int32)
            set_data["puzzle_group_indices"] = puzzle_group_indices
            self._data[set_name] = set_data

    def _collate_batch(self, batch):
        batch = {k: v.astype(np.int32) for k, v in batch.items()}

        labels = batch["labels"]
        if self.metadata.ignore_label_id is not None:
            attention_mask = labels != self.metadata.ignore_label_id
            labels = labels.copy()
            labels[~attention_mask] = IGNORE_LABEL_ID
        else:
            attention_mask = np.ones_like(labels, dtype=bool)
        batch["labels"] = labels
        batch["attention_mask"] = attention_mask.astype(np.bool_)

        if batch["puzzle_identifiers"].size < self.config.global_batch_size:
            pad_size = self.config.global_batch_size - batch["puzzle_identifiers"].size
            pad_values = {
                "inputs": self.metadata.pad_id,
                "labels": IGNORE_LABEL_ID,
                "puzzle_identifiers": self.metadata.blank_identifier_id,
                "puzzle_group_indices": 0,
                "attention_mask": False,
            }
            batch = {
                k: np.pad(
                    v,
                    ((0, pad_size),) + ((0, 0),) * (v.ndim - 1),
                    constant_values=pad_values[k],
                )
                for k, v in batch.items()
            }

        return {k: torch.from_numpy(v) for k, v in batch.items()}

    def _iter_test(self):
        for set_i, (set_name, dataset) in enumerate(self._data.items()):
            total_examples = len(dataset["inputs"])

            start_index = 0
            while start_index < total_examples:
                end_index = min(
                    total_examples, start_index + self.config.global_batch_size
                )

                puzzle_indices = []
                puzzle_index = (
                    np.searchsorted(
                        dataset["puzzle_indices"], start_index, side="right"
                    )
                    - 1
                )
                for i in range(start_index, end_index):
                    while (
                        puzzle_index + 1 < len(dataset["puzzle_indices"])
                        and i >= dataset["puzzle_indices"][puzzle_index + 1]
                    ):
                        puzzle_index += 1

                    puzzle_indices.append(puzzle_index)

                puzzle_indices = np.asarray(puzzle_indices, dtype=np.int32)
                batch = self._collate_batch(
                    {
                        "inputs": dataset["inputs"][start_index:end_index],
                        "labels": dataset["labels"][start_index:end_index],
                        "puzzle_identifiers": dataset["puzzle_identifiers"][
                            puzzle_indices
                        ],
                        "puzzle_group_indices": dataset["puzzle_group_indices"][
                            puzzle_indices
                        ],
                    }
                )

                yield set_name, batch, end_index - start_index

                start_index += self.config.global_batch_size

    def _iter_train(self):
        for set_name, dataset in self._data.items():
            self._iters += 1

            rng = np.random.Generator(
                np.random.Philox(seed=self.config.seed + self._iters)
            )

            group_order = np.concatenate(
                [
                    rng.permutation(dataset["group_indices"].size - 1)
                    for _i in range(self.config.epochs_per_iter)
                ]
            )
            start_index = 0

            while start_index < group_order.size:
                start_index, batch_indices, batch_puzzle_indices = _sample_batch(
                    rng,
                    group_order=group_order,
                    puzzle_indices=dataset["puzzle_indices"],
                    group_indices=dataset["group_indices"],
                    start_index=start_index,
                    global_batch_size=self.config.global_batch_size,
                )

                global_effective_batch_size = batch_puzzle_indices.size

                if global_effective_batch_size < self.config.global_batch_size:
                    break

                batch = self._collate_batch(
                    {
                        "inputs": dataset["inputs"][batch_indices],
                        "labels": dataset["labels"][batch_indices],
                        "puzzle_identifiers": dataset["puzzle_identifiers"][
                            batch_puzzle_indices
                        ],
                        "puzzle_group_indices": dataset["puzzle_group_indices"][
                            batch_puzzle_indices
                        ],
                    }
                )

                yield set_name, batch, global_effective_batch_size

    def __iter__(self):
        worker_info = get_worker_info()
        assert (
            worker_info is None or worker_info.num_workers == 1
        ), "Multithreaded data loading is not currently supported."

        self._lazy_load_dataset()

        if self.config.test_set_mode:
            yield from self._iter_test()
        else:
            yield from self._iter_train()


def _sample_batch(
    rng: np.random.Generator,
    group_order: np.ndarray,
    puzzle_indices: np.ndarray,
    group_indices: np.ndarray,
    start_index: int,
    global_batch_size: int,
):
    batch = []
    batch_puzzle_indices = []
    current_size = 0

    while (start_index < group_order.size) and (current_size < global_batch_size):
        group_id = group_order[start_index]
        puzzle_id = rng.integers(group_indices[group_id], group_indices[group_id + 1])
        start_index += 1

        puzzle_start = puzzle_indices[puzzle_id]
        puzzle_size = int(puzzle_indices[puzzle_id + 1] - puzzle_start)

        append_size = min(puzzle_size, global_batch_size - current_size)

        batch_puzzle_indices.append(np.full(append_size, puzzle_id, dtype=np.int32))
        batch.append(
            puzzle_start + np.random.choice(puzzle_size, append_size, replace=False)
        )

        current_size += append_size

    return start_index, np.concatenate(batch), np.concatenate(batch_puzzle_indices)
