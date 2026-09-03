"""LightningModule with U-Net, loss, TorchMetrics and optimizer."""
import pytorch_lightning as pl
import torch

from losses import CombinedLoss
from metrics import build_segmentation_metrics
from model import UNet2D


class SegmentationModule(pl.LightningModule):
    def __init__(self, config, class_weights, num_classes, class_name_mapping):
        super().__init__()

        expected_class_ids = set(range(num_classes))

        if set(class_name_mapping.keys()) != expected_class_ids:
            raise ValueError(
                f"class_name_mapping keys must be consecutive integers starting at 0. "
                f"Expected {expected_class_ids}, got {set(class_name_mapping.keys())}"
            )

        self.class_name_mapping = class_name_mapping

        self.config = config
        model_config = {**config["model"], "num_classes": num_classes}
        self.save_hyperparameters(
            {
                "model": model_config,
                "loss": config["loss"],
                "training": config["training"],
                "class_name_mapping": class_name_mapping
            }
        )

        self.model = UNet2D(**model_config)
        self.register_buffer("class_weights", class_weights)
        self.criterion = CombinedLoss(
            self.class_weights,
            num_classes,
            config["loss"],
        )

        self.train_metrics = build_segmentation_metrics(num_classes, prefix="train/")
        self.val_metrics = build_segmentation_metrics(num_classes, prefix="val/")

    def forward(self, images):
        return self.model(images)

    def training_step(self, batch, batch_idx):
        images, masks = batch

        logits = self(images)
        loss = self.criterion(logits, masks)
        predictions = logits.argmax(dim=1)

        self.train_metrics.update(predictions, masks)
        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=images.size(0),
        )

        return loss

    def on_train_epoch_end(self):
        metrics = self.train_metrics.compute()
        self._log_metrics(metrics)
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        images, masks = batch

        logits = self(images)
        loss = self.criterion(logits, masks)
        predictions = logits.argmax(dim=1)

        self.val_metrics.update(predictions, masks)
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=images.size(0),
        )

    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute()
        self._log_metrics(metrics)
        self.val_metrics.reset()

    def _log_metrics(self, metrics):
        to_log = {}

        for metric_name, value in metrics.items():
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                for class_id, class_value in enumerate(value):
                    to_log[f"{metric_name}_{self.class_name_mapping[class_id]}"] = class_value
            else:
                to_log[metric_name] = value

        to_log["step"] = self.current_epoch

        self.log_dict(
            to_log,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    def configure_optimizers(self):
        training = self.config["training"]

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=training["learning_rate"],
            weight_decay=training["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            **training["scheduler"],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/mean_dice_fg",
            },
        }
