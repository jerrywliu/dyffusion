from __future__ import annotations

import math
import os
from os.path import join
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from einops import rearrange
from torch import Tensor

from src.datamodules.abstract_datamodule import BaseDataModule
from src.datamodules.datasets.nstk import NSTK
from src.datamodules.torch_datasets import MyTensorDataset
from src.utilities.utils import (
    get_logger,
    raise_error_if_invalid_type,
    raise_error_if_invalid_value,
    raise_if_invalid_shape,
)

log = get_logger(__name__)

class NSTKDataModule(BaseDataModule):
    def __init__(
        self,
        data_dir: str,
        physical_system: str = "nstk",
        window: int = 1,
        horizon: int = 1,
        prediction_horizon: int = None,  # None means use horizon
        multi_horizon: bool = False,
        num_test_obstacles: int = 1, # TODO
        test_out_of_distribution: bool = False, # TODO
        num_trajectories: int = None,  # None means all trajectories for training
        **kwargs,
    ):
        raise_error_if_invalid_type(data_dir, possible_types=[str], name="data_dir")
        if "nn-benchmark" not in data_dir:
            for sub_dir in ["physical-nn-benchmark", "nn-benchmark"]:
                if os.path.isdir(join(data_dir, sub_dir)):
                    data_dir = join(data_dir, sub_dir)
                    break
        super().__init__(data_dir=data_dir, **kwargs)
        self.save_hyperparameters()
        self.test_batch_size = 1  # to make sure that the test dataloader returns a single trajectory
        assert window == 1, "window > 1 is not supported yet for this data module."
        raise_error_if_invalid_value(
            physical_system, possible_values=["nstk"], name="nstk"
        )
        ood_infix = "outdist-" if test_out_of_distribution else ""

        if physical_system == "nstk":
            pass
        else:
            raise NotImplementedError(f"Physical system {physical_system} is not implemented yet.")

        log.info(f"Using data directory: {self.hparams.data_dir}")

    @property
    def test_set_name(self):
        raise NotImplementedError("test_set_name is not implemented yet for NSTKDataModule")

    def get_horizon(self, split: str):
        if split in ["predict", "test"]:
            return self.hparams.prediction_horizon or self.hparams.horizon
        else:
            return self.hparams.horizon

    def _check_args(self):
        h = self.hparams.horizon
        w = self.hparams.window
        assert isinstance(h, list) or h > 0, f"horizon must be > 0 or a list, but is {h}"
        assert w > 0, f"window must be > 0, but is {w}"

    def get_nstk_dataset(self, split: str) -> NSTK: # TODO split
        return NSTK(
            patch_size=256,
            stride=128,
            scratch_dir=self.hparams.data_dir,
            size=1 if split in ["val", "test", "predict"] else 1000,
        )

    def update_predict_data(self, trajectory_subdir: str):
        raise NotImplementedError("update_predict_data is not implemented yet for NSTKDataModule")
        """
        self.subdirs["predict"] = trajectory_subdir
        assert os.path.isdir(
            self._get_numpy_filename("predict")
        ), f"Could not find data for prediction in {self._get_numpy_filename('predict')}"
        """

    def setup(self, stage: Optional[str] = None):
        """Load data. Set internal variables: self._data_train, self._data_val, self._data_test."""
        assert stage in ["fit", "validate", "test", "predict", None], f"Invalid stage {stage}"
        print(f"Setting up NSTKDataModule for stage {stage}...")
        ds_train = self.get_nstk_dataset("train") if stage in ["fit", None] else None
        ds_val = self.get_nstk_dataset("val") if stage in ["fit", "validate", None] else None
        ds_test = self.get_nstk_dataset("test") if stage in ["test", None] else None
        ds_predict = self.get_nstk_dataset("predict") if stage == "predict" else None
        ds_splits = {"train": ds_train, "val": ds_val, "test": ds_test, "predict": ds_predict}

        for split, split_ds in ds_splits.items():
            print(f"Creating tensor dataset for {split}...")
            dkwargs = {"split": split, "dataset": split_ds, "keep_trajectory_dim": False}  # split == "test"}
            if split_ds is None:
                continue
            elif self.hparams.multi_horizon:
                numpy_tensors = self.create_dataset_multi_horizon(**dkwargs)
            else:
                numpy_tensors = self.create_dataset_single_horizon(**dkwargs)

            # Create the pytorch tensor dataset
            # For the test set, we keep the trajectory dimension, so that we can evaluate the predictions
            # on the full trajectories, thus the test dataset will have a length of num_trajectories
            tensor_ds = MyTensorDataset(numpy_tensors, dataset_id=split)
            # Save the tensor dataset to self._data_{split}
            setattr(self, f"_data_{split}", tensor_ds)
            assert getattr(self, f"_data_{split}") is not None, f"Could not create {split} dataset"

        # Print sizes of the datasets (how many examples)
        self.print_data_sizes(stage)

    def create_dataset_single_horizon(
        self, split: str, dataset: NSTK, keep_trajectory_dim: bool = False
    ) -> Dict[str, np.ndarray]:
        """Create a torch dataset from the given TrajectoryDataset and return it."""
        data = self.create_dataset_multi_horizon(split, dataset, keep_trajectory_dim)
        dynamics = data.pop("dynamics")
        window, horizon = self.hparams.window, self.get_horizon(split)
        assert dynamics.shape[1] == window + horizon, f"Expected dynamics to have shape (b, {window + horizon}, ...)"
        inputs = dynamics[:, :window, ...]
        targets = dynamics[:, -1, ...]
        return {"inputs": inputs, "targets": targets, **data}

    def create_dataset_multi_horizon(
        self, split: str, dataset: NSTK, keep_trajectory_dim: bool = False
    ) -> Dict[str, np.ndarray]:
        """Create a numpy dataset from the given xarray dataset and return it."""
        # dataset is 4D tensor with dimensions (channels, time, lat, lon)
        # Create a tensor, X, of shape (batch-dim, horizon, lat, lon),
        # where each X[i] is a temporal sequence of horizon time steps
        window, horizon = self.hparams.window, self.get_horizon(split)

        print(f"Creating dataset for split {split} with window={window}, horizon={horizon}...")

        trajectories = []
        # Dataset = (time, lat, lon)
        n_trajectories = len(dataset)
        for i in range(n_trajectories):
            trajectory = dataset[i]
            time_len = len(trajectory) - horizon - window + 1
            X = np.lib.stride_tricks.sliding_window_view(trajectory, time_len, axis=0) # (horizon + window - 1, lat, lon, time_len)
            X = rearrange(X, "dynamics lat lon time_len -> time_len dynamics 1 lat lon")
            trajectories.append(X)
        X = np.concatenate(trajectories, axis=0)
        # X is now 4D tensor with dimensions (example, dynamics, lat, lon)
        # E.g. with 90 total examples, horizon=5, window=1:
        print(X.shape) # (90, 6, 256, 256)
        return {"dynamics": X}

    def boundary_conditions(
        self,
        preds: Tensor,
        targets: Tensor,
        metadata,
        time: float = None,
    ) -> Tensor:
        raise NotImplementedError(f"boundary_conditions for {self.hparams.physical_system} not implemented")

    def get_boundary_condition_kwargs(self, batch: Any, batch_idx: int, split: str) -> dict:
        raise NotImplementedError(f"get_boundary_condition_kwargs for {self.hparams.physical_system} not implemented")
