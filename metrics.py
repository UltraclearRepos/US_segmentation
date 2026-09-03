"""TorchMetrics collections used by the segmentation LightningModule."""
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassJaccardIndex
from torchmetrics.segmentation import DiceScore


def build_segmentation_metrics(num_classes, prefix):
    return MetricCollection(
        {
            "dice_per_class": DiceScore(
                num_classes=num_classes,
                include_background=True,
                average="none",
                aggregation_level="global",
                input_format="index",
            ),
            "iou_per_class": MulticlassJaccardIndex(
                num_classes=num_classes,
                average="none",
            ),
            "mean_dice_fg": DiceScore(
                num_classes=num_classes,
                include_background=False,
                average="macro",
                aggregation_level="global",
                input_format="index",
            ),
            "mean_iou_fg": MulticlassJaccardIndex(
                num_classes=num_classes,
                average="macro",
                ignore_index=0,
            ),
        },
        prefix=prefix,
    )
