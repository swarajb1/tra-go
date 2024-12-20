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
from core.evaluate_models import evaluate_models
from data_loader import DataLoader
from keras.callbacks import Callback, ModelCheckpoint, TensorBoard, TerminateOnNaN
from numpy.typing import NDArray
from tensorflow.keras.models import Model

from database.enums import BandType, IntervalType, ModelLocationType, TickerOne

NUMBER_OF_NEURONS: int = int(os.getenv("NUMBER_OF_NEURONS"))

NUMBER_OF_EPOCHS: int = int(os.getenv("NUMBER_OF_EPOCHS"))
BATCH_SIZE: int = int(os.getenv("BATCH_SIZE"))
LEARNING_RATE: float = float(os.getenv("LEARNING_RATE"))
TEST_SIZE: float = float(os.getenv("TEST_SIZE"))


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
        monitor="metric_win_pred_capture_percent",
        mode="max",
    )

    mcp_save_4 = ModelCheckpoint(
        f"{checkpoint_path_prefix} - modelCheckPoint-4.keras",
        save_best_only=True,
        monitor="val_metric_win_pred_capture_percent",
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

    callbacks = [
        tensorboard_callback,
        terNan,
        mcp_save_1,
        mcp_save_2,
        mcp_save_3,
        mcp_save_4,
        mcp_save_5,
        mcp_save_6,
    ]

    print(f"\n\nnow_datetime:\t{now_datetime}\n\n")

    if Y_TYPE == BandType.BAND_4:
        (X_train, Y_train, train_prev_close), (
            X_test,
            Y_test,
            test_prev_close,
        ) = an.train_test_split(
            data_df=df,
            test_size=TEST_SIZE,
            x_type=X_TYPE,
            y_type=Y_TYPE,
            interval=INTERVAL.value,
        )

        model = km.get_untrained_model(X_train=X_train, Y_train=Y_train)

        print("training data shape\t", X_train.shape)
        print("training element shape\t", X_train[0].shape)

        print("model output shape\t", model.output_shape)

        optimizer = km.get_optimiser(learning_rate=LEARNING_RATE)

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
            test_size=TEST_SIZE,
            x_type=X_TYPE,
            y_type=Y_TYPE,
            interval=INTERVAL.value,
        )

        print("training x data shape\t", X_train.shape)
        print("training y data shape\t", Y_train.shape)

        model = km.get_untrained_model(X_train=X_train, Y_train=Y_train)

        print("model output shape\t", model.output_shape)

        optimizer = km.get_optimiser(learning_rate=LEARNING_RATE)

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
            test_size=TEST_SIZE,
        )

        (X_train, Y_train), (X_test, Y_test) = data_loader.get_train_test_split_data()

        print("training x data shape\t", X_train.shape)
        print("training y data shape\t", Y_train.shape)

        model = km_21_model.get_untrained_model(X_train=X_train, Y_train=Y_train)

        print("model input shape\t", model.input_shape)
        print("model output shape\t", model.output_shape)

    history = model.fit(
        x=X_train,
        y=Y_train,
        epochs=NUMBER_OF_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, Y_test),
        callbacks=callbacks,
    )

    model.save(f"{checkpoint_path_prefix}.keras")

    print("\nmodel : training done. \n")

    print(f"\n\nnow_datetime:\t{now_datetime}\n\n")
    print("-" * 30)

    # number_of_model_checkpoints: int = 6

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
    if NUMBER_OF_NEURONS <= 128:
        return

    # Get the current process ID
    pid = os.getpid()

    SUPPRESSION_LEVEL: int = 14

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

    suppress_cpu_usage()

    if len(sys.argv) > 1:
        number_of_models: int = 7
        move_files: bool = False

        if len(sys.argv) > 2:
            number_of_models = int(sys.argv[2])

        if len(sys.argv) > 3:
            move_files = bool(int(sys.argv[3]))

        if sys.argv[1] == "true":
            global TICKER

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
        main_training()

        # evaluate_models(
        #     model_location_type=ModelLocationType.SAVED,
        #     number_of_models=6,
        #     move_files=False,
        # )

    print(f"\ntime taken = {round(time.time() - time_1, 2)} sec\n")


if __name__ == "__main__":
    main()
