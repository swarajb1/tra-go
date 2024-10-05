import os
import subprocess
import sys
import time
from datetime import datetime

import band_2.keras_model_band_2 as km_2
import band_2_1.keras_model as km_21_model
import band_4.keras_model_band_4 as km_4
import keras_model as km
import psutil
import training_zero as an
from core.assertions import assert_env_vals
from core.config import settings
from core.evaluate_models import evaluate_models
from data_loader import DataLoader
from keras.callbacks import Callback, ModelCheckpoint, TensorBoard, TerminateOnNaN
from numpy.typing import NDArray
from tensorflow.keras.models import Model

from database.enums import BandType, IntervalType, ModelLocationType, TickerOne

X_TYPE: BandType = BandType.BAND_1_CLOSE
Y_TYPE: BandType = BandType.BAND_1_1

TICKER: TickerOne = TickerOne.ICICIBANK
INTERVAL: IntervalType = IntervalType.MIN_1


list_of_tickers: list[TickerOne] = [
    TickerOne.ICICIBANK,
    TickerOne.RELIANCE,
    TickerOne.SBIN,
    TickerOne.LT,
    TickerOne.ITC,
    TickerOne.TCS,
    TickerOne.HDFCBANK,
    TickerOne.BHARTIARTL,
    TickerOne.AXISBANK,
    TickerOne.HINDUNILVR,
    TickerOne.KOTAKBANK,
    TickerOne.INFY,
    TickerOne.BAJFINANCE,
    TickerOne.ASIANPAINT,
    TickerOne.M_M,
    TickerOne.TITAN,
    TickerOne.HCLTECH,
    TickerOne.MARUTI,
    TickerOne.SUNPHARMA,
    TickerOne.NTPC,
]


def main_training():
    df = an.get_data_all_df(ticker=TICKER, interval=INTERVAL.value)

    model: Model

    X_train: NDArray
    Y_train: NDArray
    train_prev_close: NDArray

    X_test: NDArray
    Y_test: NDArray
    test_prev_close: NDArray

    terNan: Callback = TerminateOnNaN()

    now_datetime: str = datetime.now().strftime("%Y-%m-%d %H-%M")

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

    if Y_TYPE == BandType.BAND_4:
        (X_train, Y_train, train_prev_close), (
            X_test,
            Y_test,
            test_prev_close,
        ) = an.train_test_split(
            data_df=df,
            test_size=settings.TEST_SIZE,
            x_type=X_TYPE,
            y_type=Y_TYPE,
            interval=INTERVAL.value,
        )

        model = km.get_untrained_model(X_train=X_train, Y_train=Y_train)

        optimizer = km.get_optimiser(learning_rate=settings.LEARNING_RATE)

        loss = km_4.metric_new_idea

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[
                km.metric_rmse_percent,
                km.metric_abs_percent,
                km_4.metric_loss_comp_2,
                km_4.metric_win_percent,
                km_4.metric_win_pred_capture_percent,
                km_4.metric_pred_capture_percent,
                km_4.metric_win_checkpoint,
            ],
        )

    elif Y_TYPE == BandType.BAND_2:
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

        model = km.get_untrained_model(X_train=X_train, Y_train=Y_train)

        optimizer = km.get_optimiser(learning_rate=settings.LEARNING_RATE)

        loss = km_2.metric_new_idea

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[
                km.metric_rmse_percent,
                km.metric_abs_percent,
                km_2.metric_loss_comp_2,
                km_2.metric_win_percent,
                km_2.metric_win_pred_capture_percent,
                km_2.metric_win_correct_trend_percent,
                km_2.metric_pred_capture_percent,
                km_2.metric_pred_trend_capture_percent,
            ],
        )

    elif Y_TYPE in [BandType.BAND_2_1, BandType.BAND_1_1]:
        data_loader = DataLoader(
            ticker=TICKER,
            interval=INTERVAL,
            x_type=X_TYPE,
            y_type=Y_TYPE,
            test_size=settings.TEST_SIZE,
        )

        (X_train, Y_train), (X_test, Y_test) = data_loader.get_train_test_split_data()

        model = km_21_model.get_untrained_model(X_train=X_train, Y_train=Y_train)

    print("training x data shape\t", X_train.shape)
    print("training y data shape\t", Y_train.shape)

    print("model input shape\t", model.input_shape)
    print("model output shape\t", model.output_shape, "\n" * 2)

    history = model.fit(
        x=X_train,
        y=Y_train,
        epochs=settings.NUMBER_OF_EPOCHS,
        batch_size=settings.BATCH_SIZE,
        validation_data=(X_test, Y_test),
        callbacks=callbacks,
    )

    model.save(f"{checkpoint_path_prefix}.keras")

    print("\nmodel : training done. \n")

    print(f"\n\nnow_datetime:\t{now_datetime}\n\n")
    print("-" * 30)

    # number_of_model_checkpoints: int = 10

    # num_models_to_evaluate: int = number_of_model_checkpoints + 1

    # evaluate_models(
    #     model_location_type=ModelLocationType.TRAINED_NEW,
    #     number_of_models=num_models_to_evaluate,
    #     newly_trained_models=True,
    # )

    battery = psutil.sensors_battery()

    is_plugged = battery.power_plugged

    print("is battery on charging: ", is_plugged)


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


def main():
    os.system("clear")
    time_1 = time.time()

    assert_env_vals()

    if len(sys.argv) > 1:
        number_of_models: int = 7
        move_files: bool = False

        if len(sys.argv) > 2:
            if not sys.argv[2].isdigit():
                raise ValueError("2nd argument should be an integer")

            number_of_models = int(sys.argv[2])

        if len(sys.argv) > 3:
            if sys.argv[3] not in ["true", "false"]:
                raise ValueError("3rd argument should be either 'true' or 'false'")

            move_files = sys.argv[3] == "true"

        if sys.argv[1] == "true":
            global TICKER

            suppress_cpu_usage()

            for ticker in list_of_tickers:
                TICKER = ticker

                main_training()

        elif sys.argv[1] == "training_new":
            evaluate_models(
                model_location_type=ModelLocationType.TRAINED_NEW,
                number_of_models=6,
                newly_trained_models=True,
            )

        elif sys.argv[1] == "training":
            evaluate_models(
                model_location_type=ModelLocationType.TRAINED_NEW,
                number_of_models=number_of_models,
                move_files=move_files,
            )

        elif sys.argv[1] == "saved":
            evaluate_models(
                model_location_type=ModelLocationType.SAVED,
                number_of_models=number_of_models,
                move_files=move_files,
            )

        elif sys.argv[1] == "saved_double":
            evaluate_models(model_location_type=ModelLocationType.SAVED_DOUBLE, number_of_models=number_of_models)

        elif sys.argv[1] == "saved_triple":
            evaluate_models(model_location_type=ModelLocationType.SAVED_TRIPLE, number_of_models=number_of_models)

        elif sys.argv[1] == "old":
            evaluate_models(
                model_location_type=ModelLocationType.OLD,
                number_of_models=number_of_models,
                move_files=move_files,
            )

        elif sys.argv[1] == "discarded":
            evaluate_models(
                model_location_type=ModelLocationType.DISCARDED,
                number_of_models=number_of_models,
                move_files=True,
            )

    else:
        # main_training()

        evaluate_models(
            model_location_type=ModelLocationType.SAVED,
            number_of_models=6,
            move_files=False,
        )

    print(f"\ntime taken = {round(time.time() - time_1, 2)} sec\n")


if __name__ == "__main__":
    main()
