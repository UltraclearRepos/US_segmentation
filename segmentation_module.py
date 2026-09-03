"""LightningModule with U-Net, loss, metrics and optimizer."""

import numpy as np
import pytorch_lightning as pl
import torch
from losses import CombinedLoss
from metrics import batch_metrics, aggregate
from model import UNet2D


class SegmentationModule(pl.LightningModule):
    def __init__(self, config, class_weights):
        super().__init__()
        self.config = config
        self.save_hyperparameters(
            {
                "model": config["model"],
                "loss": config["loss"],
                "training": config["training"],
            }
        )
        self.model = UNet2D(**config["model"])
        self.register_buffer("class_weights", class_weights)
        self.criterion = CombinedLoss(
            self.class_weights, config["model"]["num_classes"], config["loss"]
        )
        self.validation_metrics = []

    def forward(self, images):
        return self.model(images)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        loss = self.criterion(self(images), masks)
        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=images.size(0),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        loss = self.criterion(logits, masks)
        self.validation_metrics.append(
            batch_metrics(logits, masks, self.config["model"]["num_classes"])
        )
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=images.size(0),
        )

    def on_validation_epoch_end(self):
        if not self.validation_metrics:
            return
        values = aggregate(self.validation_metrics)
        self.validation_metrics.clear()
        self.log("val/mean_dice_fg", values["mean_dice_fg"], prog_bar=True)
        self.log("val/mean_iou_fg", values["mean_iou_fg"])
        for class_id, value in enumerate(values["dice_per_class"]):
            self.log(f"val/dice_class_{class_id}", value)
        for class_id, value in enumerate(values["iou_per_class"]):
            self.log(f"val/iou_class_{class_id}", value)

    def configure_optimizers(self):
        training = self.config["training"]
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=training["learning_rate"],
            weight_decay=training["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", **training["scheduler"]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val/mean_dice_fg"},
        }
