"""LightningModule with U-Net, loss, TorchMetrics and optimizer."""
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from losses import CombinedLoss
from metrics import build_segmentation_metrics
from model import UNet


class SegmentationModelWrapper(pl.LightningModule):
    def __init__(self, config, class_weights, num_classes, class_name_mappings):
        super().__init__()

        expected_class_ids = set(range(num_classes))

        for task_name, mapping in class_name_mappings.items():
            if set(mapping.keys()) != expected_class_ids:
                raise ValueError(
                    f"Class IDs for {task_name} must be consecutive integers starting at 0. "
                    f"Expected {expected_class_ids}, got {set(mapping.keys())}"
                )

        self.task_names = list(class_name_mappings)
        self.class_name_mappings = class_name_mappings

        self.config = config
        model_config = {
            **config["model"],
            "checkpoint_path": config["paths"]["input_checkpoint"],
            "n_classes": num_classes
        }
        self.save_hyperparameters(
            {
                "model": model_config,
                "loss": config["loss"],
                "training": config["training"],
                "class_name_mappings": class_name_mappings,
            }
        )

        self.model = UNet(**model_config)
        self.criteria = torch.nn.ModuleDict(
            {
                task_name: CombinedLoss(
                    class_weights[task_name],
                    num_classes,
                    config["loss"],
                )
                for task_name in self.task_names
            }
        )

        self.train_metrics = torch.nn.ModuleDict(
            {
                task_name: build_segmentation_metrics(
                    num_classes,
                    prefix=f"train/{task_name}/",
                )
                for task_name in self.task_names
            }
        )
        self.val_metrics = torch.nn.ModuleDict(
            {
                task_name: build_segmentation_metrics(
                    num_classes,
                    prefix=f"val/{task_name}/",
                )
                for task_name in self.task_names
            }
        )

    def forward(self, images):
        return self.model(images)

    @staticmethod
    def _select_task(task_names, task_name, device):
        return torch.tensor(
            [name == task_name for name in task_names],
            dtype=torch.bool,
            device=device,
        )

    def _compute_loss(self, outputs, masks, task_names):
        losses = []
        for task_name in self.task_names:
            selected = self._select_task(task_names, task_name, masks.device)
            if selected.any():
                losses.append(
                    self.criteria[task_name](
                        outputs[task_name][selected],
                        masks[selected],
                    )
                )
        return torch.stack(losses).mean()

    def training_step(self, batch, batch_idx):
        images, masks, _, task_names = batch

        outputs = self(images)
        loss = self._compute_loss(outputs, masks, task_names)

        for task_name in self.task_names:
            selected = self._select_task(task_names, task_name, masks.device)
            if selected.any():
                predictions = outputs[task_name][selected].argmax(dim=1)
                self.train_metrics[task_name].update(predictions, masks[selected])
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
        self._log_task_metrics(self.train_metrics, "train")

    def validation_step(self, batch, batch_idx):
        images, resized_masks, original_masks, task_names = batch

        outputs = self(images)
        loss = self._compute_loss(outputs, resized_masks, task_names)

        for task_name in self.task_names:
            selected = self._select_task(task_names, task_name, resized_masks.device)
            indices = torch.where(selected)[0]
            predictions = outputs[task_name][selected].argmax(dim=1)
            for prediction, index in zip(predictions, indices.tolist()):
                original_mask = original_masks[index]
                prediction_at_original_size = F.interpolate(
                    prediction[None, None].float(),
                    size=original_mask.shape,
                    mode="nearest",
                )[0, 0].long()
                self.val_metrics[task_name].update(
                    prediction_at_original_size.unsqueeze(0),
                    original_mask.unsqueeze(0),
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
        if self.trainer.sanity_checking:
            for collection in self.val_metrics.values():
                collection.reset()
            return
        self._log_task_metrics(self.val_metrics, "val")

    def _log_task_metrics(self, metric_collections, stage):
        to_log = {}
        foreground_dice = []

        for task_name, collection in metric_collections.items():
            metrics = collection.compute()
            foreground_dice.append(metrics[f"{stage}/{task_name}/mean_dice_fg"])
            for metric_name, value in metrics.items():
                if isinstance(value, torch.Tensor) and value.ndim > 0:
                    for class_id, class_value in enumerate(value):
                        class_name = self.class_name_mappings[task_name][class_id]
                        to_log[f"{metric_name}_{class_name}"] = class_value
                else:
                    to_log[metric_name] = value
            collection.reset()

        to_log[f"{stage}/mean_dice_fg"] = torch.stack(foreground_dice).mean()
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

        return {
            "optimizer": optimizer
        }
