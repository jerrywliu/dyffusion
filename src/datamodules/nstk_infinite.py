from typing import Optional
from torch.utils.data import DataLoader

from src.datamodules.abstract_datamodule import BaseDataModule
from src.datamodules.datasets.nstk import NSTK

from src.utilities.utils import (
    get_logger,
    raise_error_if_invalid_type,
    raise_error_if_invalid_value,
    raise_if_invalid_shape,
)

log = get_logger(__name__)

class NSTKInfiniteDataModule(BaseDataModule):
    def __init__(
        self,
        data_dir: str,
        physical_system: str = "nstk",
        window: int = 1,
        horizon: int = 1,
        prediction_horizon: int = None,  # None means use horizon
        size: int = 1000,
        **kwargs,
    ):
        super().__init__(
            data_dir=data_dir,
            **kwargs,
        )
        self.save_hyperparameters()
        self.test_batch_size = 1  # to make sure that the test dataloader returns a single trajectory
        assert window == 1, "window > 1 is not supported yet for this data module."

        log.info(f"Using data directory: {self.hparams.data_dir}")

    # TODO JL - bugfix the following methods
    def setup(self, stage: Optional[str] = None):
        """Load data and set up NSTK dataset for train/val/test."""
        if stage in ["fit", None]:
            self._data_train = NSTK(
                scratch_dir=self.hparams.data_dir,
                size=self.hparams.size,
                window=self.hparams.window,
                horizon=self.hparams.horizon,
            )
        if stage in ["fit", "validate", None]:
            print(self.hparams.size)
            self._data_val = NSTK(
                scratch_dir=self.hparams.data_dir,
                size=16,
                window=self.hparams.window,
                horizon=self.hparams.horizon,
            )
        if stage in ["test", None]:
            self._data_test = NSTK(
                scratch_dir=self.hparams.data_dir,
                size=16,
                window=self.hparams.window,
                horizon=self.hparams.horizon,
            )
        if stage == "predict":
            self._data_predict = NSTK(
                scratch_dir=self.hparams.data_dir,
                size=64,
                window=self.hparams.window,
                horizon=self.hparams.horizon,
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self._data_train,
            batch_size=self.hparams.batch_size,
            drop_last=self.hparams.drop_last,
            shuffle=True,
            **self._shared_dataloader_kwargs(),
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self._data_val,
            batch_size=self.hparams.eval_batch_size,
            **self._shared_eval_dataloader_kwargs(),
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self._data_test,
            batch_size=self.hparams.eval_batch_size,
            **self._shared_eval_dataloader_kwargs(),
        )

    def predict_dataloader(self):
        return DataLoader(
            dataset=self._data_predict,
            batch_size=self.hparams.eval_batch_size,
            **self._shared_eval_dataloader_kwargs(),
        )
