import os
import subprocess
import sys
import time
from datetime import datetime

import keras_model as km
import psutil
import training_zero as an
from keras.callbacks import ModelCheckpoint, TensorBoard, TerminateOnNaN

import tra_go.band_2.keras_model_band_2 as km_2
import tra_go.band_2_1.keras_model as km_21_model
import tra_go.band_4.keras_model_band_4 as km_4
from database.enums import BandType, IntervalType, ModelLocationType, TickerOne
from tra_go.core.evaluate_models import evaluate_models

NUMBER_OF_NEURONS: int = int(os.getenv("NUMBER_OF_NEURONS"))

NUMBER_OF_EPOCHS: int = int(os.getenv("NUMBER_OF_EPOCHS"))
BATCH_SIZE: int = int(os.getenv("BATCH_SIZE"))
LEARNING_RATE: float = float(os.getenv("LEARNING_RATE"))
TEST_SIZE: float = float(os.getenv("TEST_SIZE"))


IS_TRAINING_MODEL: bool = True
prev_model: str = "2024-04-08 11-50"


X_TYPE: BandType = BandType.BAND_4
Y_TYPE: BandType = BandType.BAND_2_1

# (HDFCBANK,    RELIANCE,    ICICIBANK,    INFY,    LT,    ITC,    TCS,    BHARTIARTL,    AXISBANK,    SBIN) #ignore

TICKER: TickerOne = TickerOne.RELIANCE
INTERVAL: str = "1m"
INTERVAL_1: IntervalType = IntervalType.MIN_1

PREV_MODEL_TRAINING: bool = False


def main():
    df = an.get_data_all_df(ticker=TICKER, interval=INTERVAL)

    if IS_TRAINING_MODEL:
        # common elements for types

        terNan = TerminateOnNaN()

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
            interval=INTERVAL,
        )

        if IS_TRAINING_MODEL and not PREV_MODEL_TRAINING:
            now_datetime = datetime.now().strftime("%Y-%m-%d %H-%M")
        else:
            now_datetime = prev_model

        if IS_TRAINING_MODEL:
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

            log_dir: str = os.path.join("training", "logs", f"{now_datetime} - {Y_TYPE}")

            tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

            mcp_save_1 = ModelCheckpoint(
                f"training/models/model - {now_datetime} - {Y_TYPE} - modelCheckPoint-1.keras",
                save_best_only=True,
                monitor="loss",
                mode="min",
            )

            mcp_save_2 = ModelCheckpoint(
                f"training/models/model - {now_datetime} - {Y_TYPE} - modelCheckPoint-2.keras",
                save_best_only=True,
                monitor="val_loss",
                mode="min",
            )

            mcp_save_3 = ModelCheckpoint(
                f"training/models/model - {now_datetime} - {Y_TYPE} - modelCheckPoint-3.keras",
                save_best_only=True,
                monitor="metric_pred_capture_percent",
                mode="max",
            )

            mcp_save_4 = ModelCheckpoint(
                f"training/models/model - {now_datetime} - {Y_TYPE} - modelCheckPoint-4.keras",
                save_best_only=True,
                monitor="val_metric_pred_capture_percent",
                mode="max",
            )

            mcp_save_5 = ModelCheckpoint(
                f"training/models/model - {now_datetime} - {Y_TYPE} - modelCheckPoint-5.keras",
                save_best_only=True,
                monitor="metric_win_checkpoint",
                mode="max",
            )

            mcp_save_6 = ModelCheckpoint(
                f"training/models/model - {now_datetime} - {Y_TYPE} - modelCheckPoint-6.keras",
                save_best_only=True,
                monitor="val_metric_win_checkpoint",
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

            history = model.fit(
                x=X_train,
                y=Y_train,
                epochs=NUMBER_OF_EPOCHS,
                batch_size=BATCH_SIZE,
                validation_data=(X_test, Y_test),
                callbacks=callbacks,
            )

            model.save(f"training/models/model - {now_datetime} - {Y_TYPE}.keras")

            print("\nmodel : training done. \n")

        print(f"\n\nnow_datetime:\t{now_datetime}\n\n")
        print("-" * 30)

    elif Y_TYPE == BandType.BAND_2:
        (
            (X_train, Y_train, train_prev_close),
            (X_test, Y_test, test_prev_close),
        ) = an.train_test_split(
            data_df=df,
            test_size=TEST_SIZE,
            x_type=X_TYPE,
            y_type=Y_TYPE,
            interval=INTERVAL,
        )

        if IS_TRAINING_MODEL and not PREV_MODEL_TRAINING:
            now_datetime = datetime.now().strftime("%Y-%m-%d %H-%M")
        else:
            now_datetime = prev_model

        if IS_TRAINING_MODEL:
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
                monitor="metric_pred_capture_percent",
                mode="max",
            )

            mcp_save_4 = ModelCheckpoint(
                f"{checkpoint_path_prefix} - modelCheckPoint-4.keras",
                save_best_only=True,
                monitor="val_metric_pred_capture_percent",
                mode="max",
            )

            mcp_save_5 = ModelCheckpoint(
                f"{checkpoint_path_prefix} - modelCheckPoint-5.keras",
                save_best_only=True,
                monitor="metric_pred_trend_capture_percent",
                mode="max",
            )

            mcp_save_6 = ModelCheckpoint(
                f"{checkpoint_path_prefix} - modelCheckPoint-6.keras",
                save_best_only=True,
                monitor="val_metric_pred_trend_capture_percent",
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

        NUMBER_OF_MODEL_CHECKPOINTS: int = 6

        evaluate_models(
            model_location_type=ModelLocationType.TRAINED_NEW,
            number_of_models=NUMBER_OF_MODEL_CHECKPOINTS,
        )

    elif Y_TYPE == BandType.BAND_2_1:
        (
            (X_train, Y_train, Y_train_full, train_prev_close),
            (X_test, Y_test, Y_test_full, test_prev_close),
        ) = an.train_test_split_lh(
            data_df=df,
            test_size=TEST_SIZE,
            x_type=X_TYPE,
            interval=INTERVAL,
        )

        if IS_TRAINING_MODEL and not PREV_MODEL_TRAINING:
            now_datetime = datetime.now().strftime("%Y-%m-%d %H-%M")
        else:
            now_datetime = prev_model

        if IS_TRAINING_MODEL:
            print("training x data shape\t", X_train.shape)
            print("training y data shape\t", Y_train.shape)

            model = km_21_model.get_untrained_model(X_train=X_train, Y_train=Y_train)

            print("model output shape\t", model.output_shape)

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

        number_of_model_checkpoints: int = 6

        evaluate_models(
            model_location_type=ModelLocationType.TRAINED_NEW,
            number_of_models=number_of_model_checkpoints,
            newly_trained_models=True,
        )

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


def get_list_of_files(model_type: ModelLocationType, number_of_files: int = 6) -> list[str]:
    list_of_files = os.listdir("training/models")

    list_of_files = [file for file in list_of_files if not file.startswith(".")]

    folder_name = model_type.value

    list_of_files = sorted(
        list_of_files,
        key=lambda x: os.path.getmtime(folder_name + "/" + x),
    )

    return list_of_files[:number_of_files]


if __name__ == "__main__":
    os.system("clear")
    time_1 = time.time()

    suppress_cpu_usage()

    if len(sys.argv) > 1:
        if sys.argv[1] == "true":
            IS_TRAINING_MODEL = True

            main()

        elif sys.argv[1] == "training_new":
            IS_TRAINING_MODEL = False

            evaluate_models(
                model_location_type=ModelLocationType.TRAINED_NEW,
                number_of_models=6,
                newly_trained_models=True,
            )

        elif sys.argv[1] == "training":
            IS_TRAINING_MODEL = False

            evaluate_models(model_location_type=ModelLocationType.TRAINED_NEW, number_of_models=6)

        elif sys.argv[1] == "saved":
            IS_TRAINING_MODEL = False

            evaluate_models(model_location_type=ModelLocationType.SAVED, number_of_models=6)

        elif sys.argv[1] == "saved_double":
            IS_TRAINING_MODEL = False

            evaluate_models(model_location_type=ModelLocationType.SAVED_DOUBLE, number_of_models=6)

        elif sys.argv[1] == "saved_triple":
            IS_TRAINING_MODEL = False

            evaluate_models(model_location_type=ModelLocationType.SAVED_TRIPLE, number_of_models=6)

    else:
        main()

    print(f"\ntime taken = {round(time.time() - time_1, 2)} sec\n")
