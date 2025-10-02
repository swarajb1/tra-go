import os
import subprocess

# import band_2_1.keras_model as km_21_model
from core.config import settings
from tensorflow.keras.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
    TerminateOnNaN,
)

from database.enums import BandType, IntervalType, TickerOne

# Configuration: Set to True to use optimized tf.data pipelines, False for traditional data loading
# This can also be controlled via settings.USE_OPTIMIZED_DATA_LOADER in config

X_TYPE: BandType = BandType.BAND_1_CLOSE
Y_TYPE: BandType = BandType.BAND_1_1

TICKER: TickerOne = TickerOne.ICICIBANK
INTERVAL: IntervalType = IntervalType.MIN_1


list_of_tickers: list[TickerOne] = [
    TickerOne.ICICIBANK,
    # TickerOne.RELIANCE,
    TickerOne.SBIN,
    TickerOne.LT,
    # TickerOne.ITC,
    # TickerOne.TCS,
    # TickerOne.HDFCBANK,
    # TickerOne.BHARTIARTL,
    # TickerOne.AXISBANK,
    # TickerOne.HINDUNILVR,
    # TickerOne.KOTAKBANK,
    # TickerOne.INFY,
    # TickerOne.BAJFINANCE,
    # TickerOne.ASIANPAINT,
    # TickerOne.M_M,
    # TickerOne.TITAN,
    # TickerOne.HCLTECH,
    # TickerOne.MARUTI,
    # TickerOne.SUNPHARMA,
    # TickerOne.NTPC,
]


def suppress_cpu_usage():
    if settings.NUMBER_OF_NEURONS <= 128:
        return

    # Get the current process ID
    pid = os.getpid()

    SUPPRESSION_LEVEL: int = 15

    # The command you want to run
    command = f"cpulimit -l {SUPPRESSION_LEVEL} -p {pid}"

    # Open a new terminal and run the command
    subprocess.Popen(
        ["osascript", "-e", f'tell app "Terminal" to do script "{command}"'],
    )


def create_training_callbacks(checkpoint_prefix: str, log_dir: str) -> list[Callback]:
    """
    Create a modular list of training callbacks including model checkpoints,
    early stopping, and learning rate decay.

    Args:
        checkpoint_prefix: Base path for model checkpoints
        log_dir: Directory for TensorBoard logs

    Returns:
        List of configured callbacks
    """
    callbacks: list[Callback] = []

    # TensorBoard logging
    tensorboard_cb = TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks.append(tensorboard_cb)

    # Terminate on NaN
    ter_nan = TerminateOnNaN()
    callbacks.append(ter_nan)

    # Model checkpoints for various metrics
    model_checkpoints_mandatory = [
        ModelCheckpoint(
            f"{checkpoint_prefix} - modelCheckPoint-1.keras",
            save_best_only=True,
            monitor="loss",
            mode="min",
        ),
        ModelCheckpoint(
            f"{checkpoint_prefix} - modelCheckPoint-2.keras",
            save_best_only=True,
            monitor="val_loss",
            mode="min",
        ),
    ]
    callbacks.extend(model_checkpoints_mandatory)

    other_model_checkpoints = [
        ModelCheckpoint(
            f"{checkpoint_prefix} - modelCheckPoint-3.keras",
            save_best_only=True,
            monitor="metric_win_pred_capture_total_percent",
            mode="max",
        ),
        ModelCheckpoint(
            f"{checkpoint_prefix} - modelCheckPoint-4.keras",
            save_best_only=True,
            monitor="val_metric_win_pred_capture_total_percent",
            mode="max",
        ),
        ModelCheckpoint(
            f"{checkpoint_prefix} - modelCheckPoint-5.keras",
            save_best_only=True,
            monitor="metric_win_pred_trend_capture_percent",
            mode="max",
        ),
        ModelCheckpoint(
            f"{checkpoint_prefix} - modelCheckPoint-6.keras",
            save_best_only=True,
            monitor="val_metric_win_pred_trend_capture_percent",
            mode="max",
        ),
        ModelCheckpoint(
            f"{checkpoint_prefix} - modelCheckPoint-7.keras",
            save_best_only=True,
            monitor="metric_try_1",
            mode="max",
        ),
        ModelCheckpoint(
            f"{checkpoint_prefix} - modelCheckPoint-8.keras",
            save_best_only=True,
            monitor="val_metric_try_1",
            mode="max",
        ),
        ModelCheckpoint(
            f"{checkpoint_prefix} - modelCheckPoint-9.keras",
            save_best_only=True,
            monitor="metric_loss_comp_2",
            mode="min",
        ),
        ModelCheckpoint(
            f"{checkpoint_prefix} - modelCheckPoint-10.keras",
            save_best_only=True,
            monitor="val_metric_loss_comp_2",
            mode="min",
        ),
    ]

    callbacks.extend(other_model_checkpoints)

    # Training enhancements - Early Stopping
    if settings.EARLY_STOPPING_ENABLED:
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=settings.EARLY_STOPPING_PATIENCE,
            restore_best_weights=settings.EARLY_STOPPING_RESTORE_BEST_WEIGHTS,
            min_delta=settings.EARLY_STOPPING_MIN_DELTA,
            verbose=1,
        )
        callbacks.append(early_stopping)

    # Training enhancements - Learning Rate Decay
    if settings.LR_DECAY_ENABLED:
        lr_decay = ReduceLROnPlateau(
            monitor="val_loss",
            factor=settings.LR_DECAY_FACTOR,
            patience=settings.LR_DECAY_PATIENCE,
            min_lr=settings.LR_DECAY_MIN_LR,
            verbose=1,
        )
        callbacks.append(lr_decay)

    return callbacks
