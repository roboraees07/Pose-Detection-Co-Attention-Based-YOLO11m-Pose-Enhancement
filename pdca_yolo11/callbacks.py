"""Ultralytics training callbacks for attention ablation (no WaCIPR)."""

from __future__ import annotations

from torch.optim.lr_scheduler import ReduceLROnPlateau

from ultralytics.utils import LOGGER, colorstr


class ReduceLROnPlateauCallback:
    """Reduce LR when validation fitness plateaus (elbow), instead of fixed cosine/linear decay."""

    def __init__(
        self,
        patience: int = 5,
        factor: float = 0.5,
        min_lr: float = 1e-6,
        threshold: float = 1e-4,
    ):
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.threshold = threshold
        self._scheduler: ReduceLROnPlateau | None = None

    def on_train_start(self, trainer):
        self._scheduler = ReduceLROnPlateau(
            trainer.optimizer,
            mode="max",
            factor=self.factor,
            patience=self.patience,
            threshold=self.threshold,
            min_lr=self.min_lr,
        )
        prefix = colorstr("ReduceLROnPlateau: ")
        LOGGER.info(
            f"{prefix}Monitoring val fitness (maximize); "
            f"patience={self.patience}, factor={self.factor}, min_lr={self.min_lr}"
        )

    def on_fit_epoch_end(self, trainer):
        if self._scheduler is None or trainer.fitness is None:
            return
        before = trainer.optimizer.param_groups[0]["lr"]
        self._scheduler.step(float(trainer.fitness))
        after = trainer.optimizer.param_groups[0]["lr"]
        if after < before:
            prefix = colorstr("ReduceLROnPlateau: ")
            LOGGER.info(f"{prefix}LR {before:.6g} -> {after:.6g} (fitness={float(trainer.fitness):.5f})")


def register_reduce_lr_callback(model, **kwargs) -> ReduceLROnPlateauCallback:
    cb = ReduceLROnPlateauCallback(**kwargs)
    model.add_callback("on_train_start", cb.on_train_start)
    model.add_callback("on_fit_epoch_end", cb.on_fit_epoch_end)
    return cb
