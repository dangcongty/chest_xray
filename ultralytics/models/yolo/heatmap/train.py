# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy
from pathlib import Path

from ultralytics.models import yolo
from ultralytics.nn.tasks import HeatmapModel
from ultralytics.utils import DEFAULT_CFG, RANK


class HeatmapTrainer(yolo.detect.DetectionTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides: dict | None = None, _callbacks=None):
        if overrides is None:
            overrides = {}
        overrides["task"] = "heatmap"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg: dict | str | None = None, weights: str | Path | None = None, verbose: bool = True):
        model = HeatmapModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        """Return an instance of HeatmapValidator for validation of YOLO model."""
        self.loss_names = "box_loss", "heat_loss", "cls_loss", "dfl_loss"
        return yolo.heatmap.HeatmapValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
