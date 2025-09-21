import gc
import os
import subprocess
import time
from datetime import datetime
from typing import Final

import band_2.keras_model as km_2
import band_2_1.keras_model as km_21_model
import psutil
import tensorflow as tf
import training.common as training_common
import training_zero as an
from core.config import settings
from core.logger import log_model_training_complete, log_model_training_start, logger
from data_loader import DataLoader
from data_loader_tf import create_optimized_data_loader
from decorators.time import time_taken
from numpy.typing import NDArray
from tensorflow.keras.callbacks import (
    Callback,
    ModelCheckpoint,
    TensorBoard,
    TerminateOnNaN,
)
from tensorflow.keras.models import Model
from tf_data_utils import configure_tf_data_performance, log_tf_data_performance_tips

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
    # TickerOne.SBIN,
    # TickerOne.LT,
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


@time_taken
def main_training(ticker=None):
    """Main training function with proper error handling and logging."""

    time_start = time.time()

    now_datetime: Final[str] = datetime.now().strftime("%Y-%m-%d %H-%M")

    # Configure TensorFlow for optimal performance with tf.data pipelines (if enabled)

    if settings.USE_OPTIMIZED_DATA_LOADER:
        configure_tf_data_performance()
        log_tf_data_performance_tips()
        logger.info("Optimized data loader is ENABLED (tf.data pipelines)")
    else:
        logger.info("Optimized data loader is DISABLED (traditional numpy arrays)")

    # suppress_cpu_usage()

    if ticker is not None:
        global TICKER

        TICKER = ticker

    logger.info(f"Starting training for ticker: {TICKER.name}, interval: {INTERVAL.value}")

    try:
        df = an.get_data_all_df(ticker=TICKER, interval=INTERVAL.value)

        if df is None or df.empty:
            raise ValueError(f"No data loaded for ticker {TICKER.name} with interval {INTERVAL.value}")

        logger.info(f"Data loaded successfully with shape: {df.shape}")

        # Force garbage collection after data loading
        gc.collect()
    except Exception as e:
        logger.error(f"Error occurred while loading data: {e}")
        return

    model: Model

    X_train: NDArray
    Y_train: NDArray
    train_prev_close: NDArray

    X_test: NDArray
    Y_test: NDArray
    test_prev_close: NDArray

    terNan: Callback = TerminateOnNaN()

    log_model_training_start(TICKER.name, Y_TYPE.value, now_datetime)

    log_dir: str = os.path.join(
        "training",
        "logs",
        f"{now_datetime} - {Y_TYPE.value.lower()}",
    )

    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    checkpoint_path_prefix: str = os.path.join(
        "training",
        "models",
        f"model - {now_datetime} - {X_TYPE.value.lower()} - {Y_TYPE.value.lower()} - {TICKER.name}",
    )

    mcp_save_1 = ModelCheckpoint(
        f"{checkpoint_path_prefix} - modelCheckPoint-1.keras",
        save_best_only=True,
        monitor="loss",
        mode="min",
    )

    mcp_save_2 = ModelCheckpoint(
        f"{checkpoint_path_prefix} - modelCheckPoint-2.keras",
        save_best_only=True,
        monitor="val_loss",
        mode="min",
    )

    mcp_save_3 = ModelCheckpoint(
        f"{checkpoint_path_prefix} - modelCheckPoint-3.keras",
        save_best_only=True,
        monitor="metric_win_pred_capture_total_percent",
        mode="max",
    )

    mcp_save_4 = ModelCheckpoint(
        f"{checkpoint_path_prefix} - modelCheckPoint-4.keras",
        save_best_only=True,
        monitor="val_metric_win_pred_capture_total_percent",
        mode="max",
    )

    mcp_save_5 = ModelCheckpoint(
        f"{checkpoint_path_prefix} - modelCheckPoint-5.keras",
        save_best_only=True,
        monitor="metric_win_pred_trend_capture_percent",
        mode="max",
    )

    mcp_save_6 = ModelCheckpoint(
        f"{checkpoint_path_prefix} - modelCheckPoint-6.keras",
        save_best_only=True,
        monitor="val_metric_win_pred_trend_capture_percent",
        mode="max",
    )

    mcp_save_7 = ModelCheckpoint(
        f"{checkpoint_path_prefix} - modelCheckPoint-7.keras",
        save_best_only=True,
        monitor="metric_try_1",
        mode="max",
    )

    mcp_save_8 = ModelCheckpoint(
        f"{checkpoint_path_prefix} - modelCheckPoint-8.keras",
        save_best_only=True,
        monitor="val_metric_try_1",
        mode="max",
    )

    mcp_save_9 = ModelCheckpoint(
        f"{checkpoint_path_prefix} - modelCheckPoint-9.keras",
        save_best_only=True,
        monitor="metric_loss_comp_2",
        mode="min",
    )

    mcp_save_10 = ModelCheckpoint(
        f"{checkpoint_path_prefix} - modelCheckPoint-10.keras",
        save_best_only=True,
        monitor="val_metric_loss_comp_2",
        mode="min",
    )

    callbacks = [
        tensorboard_callback,
        terNan,
        mcp_save_1,
        mcp_save_2,
        mcp_save_3,
        mcp_save_4,
        mcp_save_5,
        mcp_save_6,
        mcp_save_7,
        mcp_save_8,
        mcp_save_9,
        mcp_save_10,
    ]

    print(f"\n\nnow_datetime:\t{now_datetime}\n\n")

    if Y_TYPE == BandType.BAND_2:
        (
            (X_train, Y_train, train_prev_close),
            (X_test, Y_test, test_prev_close),
        ) = an.train_test_split(
            data_df=df,
            test_size=settings.TEST_SIZE,
            x_type=X_TYPE,
            y_type=Y_TYPE,
            interval=INTERVAL.value,
        )

        model = km_2.get_untrained_model(X_train=X_train, Y_train=Y_train)

        optimizer = training_common.get_optimiser(learning_rate=settings.LEARNING_RATE)

        loss = km_2.metric_new_idea

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[
                training_common.metric_rmse_percent,
                training_common.metric_abs_percent,
                km_2.metric_loss_comp_2,
                km_2.metric_win_percent,
                km_2.metric_win_pred_capture_percent,
                km_2.metric_win_correct_trend_percent,
                km_2.metric_pred_capture_percent,
                km_2.metric_pred_trend_capture_percent,
            ],
        )

        # Force garbage collection after model compilation
        gc.collect()

    elif Y_TYPE in [BandType.BAND_2_1, BandType.BAND_1_1]:
        if settings.USE_OPTIMIZED_DATA_LOADER:
            logger.info("Using optimized TensorFlow DataLoader with tf.data pipelines")

            # Use the new TensorFlow DataLoader for better performance
            data_loader = create_optimized_data_loader(
                ticker=TICKER,
                interval=INTERVAL,
                x_type=X_TYPE,
                y_type=Y_TYPE,
                test_size=settings.TEST_SIZE,
                batch_size=settings.BATCH_SIZE,
                shuffle_buffer_size=1000,
                prefetch_buffer_size=tf.data.AUTOTUNE,
                enable_shuffle=True,
            )

            # Get optimized tf.data datasets
            train_dataset, test_dataset = data_loader.get_tf_datasets()

            # Also get numpy arrays for model creation (required for determining input/output shapes)
            (X_train, Y_train), (X_test, Y_test) = data_loader.get_train_test_split_data()

            # Log dataset information
            dataset_info = data_loader.get_dataset_info()
            logger.info("TensorFlow DataLoader configuration:")
            for key, value in dataset_info.items():
                logger.info(f"  {key}: {value}")

            # Benchmark data loading performance
            try:
                perf_metrics = data_loader.benchmark_performance(num_batches=5)
                logger.info("Data loading performance metrics recorded")
            except Exception as e:
                logger.warning(f"Performance benchmarking failed: {e}")
        else:
            logger.info("Using traditional DataLoader (optimized loader disabled)")

            # Use the traditional DataLoader
            data_loader = DataLoader(
                ticker=TICKER,
                interval=INTERVAL,
                x_type=X_TYPE,
                y_type=Y_TYPE,
                test_size=settings.TEST_SIZE,
            )

            (X_train, Y_train), (X_test, Y_test) = data_loader.get_train_test_split_data()

        model = km_21_model.get_untrained_model(X_train=X_train, Y_train=Y_train)

        # Force garbage collection after model creation
        gc.collect()

    print("training x data shape\t", X_train.shape)
    print("training y data shape\t", Y_train.shape)

    print("model input shape\t", model.input_shape)
    print("model output shape\t", model.output_shape, "\n" * 2)

    try:
        if (
            settings.USE_OPTIMIZED_DATA_LOADER
            and Y_TYPE in [BandType.BAND_2_1, BandType.BAND_1_1]
            and "train_dataset" in locals()
        ):
            logger.info("Training with optimized tf.data pipelines")
            history = model.fit(
                train_dataset,
                epochs=settings.NUMBER_OF_EPOCHS,
                validation_data=test_dataset,
                callbacks=callbacks,
            )
        else:
            logger.info("Training with numpy arrays (legacy mode)")
            history = model.fit(
                x=X_train,
                y=Y_train,
                epochs=settings.NUMBER_OF_EPOCHS,
                batch_size=settings.BATCH_SIZE,
                validation_data=(X_test, Y_test),
                callbacks=callbacks,
            )

        model.save(f"{checkpoint_path_prefix}.keras")
        logger.info(f"Model saved successfully to {checkpoint_path_prefix}.keras")

        # Force garbage collection after model saving
        gc.collect()

    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        raise

    log_model_training_complete(TICKER.name, Y_TYPE.value, time.time() - time_start)

    print("\nmodel : training done. \n")

    print(f"\n\nnow_datetime:\t{now_datetime}\n\n")
    print("-" * 30)

    try:
        battery = psutil.sensors_battery()
        is_plugged = battery.power_plugged
        print("is battery on charging: ", is_plugged)
    except Exception as e:
        logger.warning(f"Could not check battery status: {str(e)}")

    # Final garbage collection before function completion
    gc.collect()


def main_training_4_cores(ticker=None):
    """
    Run main training using 4 CPU cores for enhanced performance.

    This function configures TensorFlow to use 4 CPU cores for the training process,
    which can improve training speed compared to single-core execution while
    maintaining system responsiveness.

    Args:
        ticker: Optional ticker to train. If None, uses the global TICKER.
    """
    logger.info(
        f"TensorFlow threading configured via environment variables for {settings.TF_INTRA_OP_PARALLELISM_THREADS} CPU cores",
    )

    # Run the main training function
    return main_training(ticker=ticker)


if __name__ == "__main__":
    # Example usage of the new 3-core functions

    # Option 1: Train a single ticker using 3 cores
    # main_training_3_cores()  # Uses default TICKER
    # main_training_3_cores(TickerOne.RELIANCE)  # Train specific ticker

    # Default: Run single training (original behavior)
    main_training()
