"""Lightning callbacks for segmentation."""

from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)


def build_callbacks(config, checkpoint_dir):
    callback_config = config["callbacks"]
    monitor = callback_config["monitor"]
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best-{epoch:03d}",
            monitor=monitor,
            mode="max",
            save_top_k=1,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        RichProgressBar(),
    ]
    if callback_config["early_stopping"]["enabled"]:
        callbacks.append(
            EarlyStopping(
                monitor=monitor,
                mode="max",
                patience=callback_config["early_stopping"]["patience"],
                min_delta=callback_config["early_stopping"]["min_delta"],
            )
        )
    return callbacks
