import gc
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Final, Optional, cast

import band_2.keras_model as km_2
import band_2.model_metrics as km_2_metrics

# import band_2_1.keras_model as km_21_model
import band_2_1.keras_model_improved as km_21_model
import model_training.common as training_common
import psutil
import tensorflow as tf
import training_zero as an
from core.config import settings
from core.logger import log_exceptions, logger
from data_loader import DataLoader
from data_loader_tf import create_optimized_data_loader
from decorators.time import format_time, time_taken
from numpy.typing import NDArray
from tensorflow.keras.callbacks import (  # type: ignore[import-error]
    Callback,
    TerminateOnNaN,
)
from tensorflow.keras.models import Model  # type: ignore[import-error]
from tf_data_utils import log_tf_data_performance_tips
from utils.training_utils import create_training_callbacks

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


@dataclass
class TrainingArtifacts:
    model: Model
    X_train: NDArray
    Y_train: NDArray
    X_test: NDArray
    Y_test: NDArray
    train_prev_close: Optional[NDArray] = None
    test_prev_close: Optional[NDArray] = None
    train_dataset: Optional[tf.data.Dataset] = None
    test_dataset: Optional[tf.data.Dataset] = None


@log_exceptions(exit_on_exception=False)
@time_taken
def main_training(ticker=None):
    """Main training function with proper error handling and logging."""

    time_start = time.time()

    now_datetime: Final[str] = datetime.now().strftime("%Y-%m-%d %H-%M")

    # Configure TensorFlow for optimal performance with tf.data pipelines (if enabled)

    if settings.USE_OPTIMIZED_DATA_LOADER:
        log_tf_data_performance_tips()
        logger.info("Optimized data loader is ENABLED (tf.data pipelines)")
    else:
        logger.info("Optimized data loader is DISABLED (traditional numpy arrays)")

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

    terNan: Callback = TerminateOnNaN()

    logger.info(f"Starting model training | Ticker: {TICKER.name} | Type: {Y_TYPE.value} | Time: {now_datetime}")

    log_dir: str = os.path.join(
        "training",
        "logs",
        f"{now_datetime} - {Y_TYPE.value.lower()}",
    )

    checkpoint_path_prefix: str = os.path.join(
        "training",
        "models",
        f"model - {now_datetime} - {X_TYPE.value.lower()} - {Y_TYPE.value.lower()} - {TICKER.name}",
    )

    callbacks = create_training_callbacks(checkpoint_path_prefix, log_dir)

    print(f"\n\nnow_datetime:\t{now_datetime}\n\n")

    artifacts: TrainingArtifacts

    if Y_TYPE == BandType.BAND_2:
        train_split, test_split = an.train_test_split(
            data_df=df,
            test_size=settings.TEST_SIZE,
            x_type=X_TYPE,
            y_type=Y_TYPE,
            interval=INTERVAL.value,
        )

        X_train, Y_train, train_prev_close = cast(tuple[NDArray, NDArray, NDArray], train_split)
        X_test, Y_test, test_prev_close = cast(tuple[NDArray, NDArray, NDArray], test_split)

        model = km_2.get_untrained_model(X_train=X_train, Y_train=Y_train)

        optimizer = training_common.get_optimiser(learning_rate=settings.LEARNING_RATE)
        loss = km_2_metrics.metric_new_idea

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[
                training_common.metric_rmse_percent,
                training_common.metric_abs_percent,
                km_2_metrics.metric_loss_comp_2,
                km_2_metrics.metric_win_percent,
                km_2_metrics.metric_win_pred_capture_percent,
                km_2_metrics.metric_win_correct_trend_percent,
                km_2_metrics.metric_pred_capture_percent,
                km_2_metrics.metric_pred_trend_capture_percent,
            ],
        )

        gc.collect()

        artifacts = TrainingArtifacts(
            model=model,
            X_train=X_train,
            Y_train=Y_train,
            X_test=X_test,
            Y_test=Y_test,
            train_prev_close=train_prev_close,
            test_prev_close=test_prev_close,
        )

    elif Y_TYPE in [BandType.BAND_2_1, BandType.BAND_1_1]:
        train_dataset: Optional[tf.data.Dataset] = None
        test_dataset: Optional[tf.data.Dataset] = None

        if settings.USE_OPTIMIZED_DATA_LOADER:
            logger.info("Using optimized TensorFlow DataLoader with tf.data pipelines")

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

            train_dataset, test_dataset = data_loader.get_tf_datasets()

            (X_train, Y_train), (X_test, Y_test) = data_loader.get_train_test_split_data()

            dataset_info = data_loader.get_dataset_info()
            logger.info("TensorFlow DataLoader configuration:")
            for key, value in dataset_info.items():
                logger.info(f"  {key}: {value}")

            try:
                perf_metrics = data_loader.benchmark_performance(num_batches=5)
                logger.info("Data loading performance metrics recorded")
            except Exception as e:
                logger.warning(f"Performance benchmarking failed: {e}")
        else:
            logger.info("Using traditional DataLoader (optimized loader disabled)")

            data_loader = DataLoader(
                ticker=TICKER,
                interval=INTERVAL,
                x_type=X_TYPE,
                y_type=Y_TYPE,
                test_size=settings.TEST_SIZE,
            )

            (X_train, Y_train), (X_test, Y_test) = data_loader.get_train_test_split_data()

        model = km_21_model.get_untrained_model(X_train=X_train, Y_train=Y_train)

        gc.collect()

        artifacts = TrainingArtifacts(
            model=model,
            X_train=X_train,
            Y_train=Y_train,
            X_test=X_test,
            Y_test=Y_test,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
        )

    else:
        raise ValueError(f"Unsupported Y_TYPE: {Y_TYPE}")

    X_train = artifacts.X_train
    Y_train = artifacts.Y_train
    X_test = artifacts.X_test
    Y_test = artifacts.Y_test
    model = artifacts.model
    train_dataset = artifacts.train_dataset
    test_dataset = artifacts.test_dataset

    print("training x data shape\t", X_train.shape)
    print("training y data shape\t", Y_train.shape)

    print("model input shape\t", model.input_shape)
    print("model output shape\t", model.output_shape, "\n" * 2)

    if train_dataset is not None and test_dataset is not None:
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

    gc.collect()

    duration = format_time(time.time() - time_start)
    logger.info(f"Completed model training | Ticker: {TICKER.name} | Type: {Y_TYPE.value} | Duration: {duration}")

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
