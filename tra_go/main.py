import os
import subprocess
import sys
import time
from datetime import datetime

import keras_model as km
import training_zero as an
from band_2.training_yf_band_2 import CustomEvaluation
from band_4.training_yf_band_4 import CustomEvaluation as CustomEvaluation_4
from keras.callbacks import ModelCheckpoint, TensorBoard, TerminateOnNaN

import tra_go.band_2.keras_model_band_2 as km_2
import tra_go.band_4.keras_model_band_4 as km_4
from database.enums import BandType, TickerOne

IS_TRAINING_MODEL: bool = True
prev_model: str = "2024-04-08 11-50"


NUMBER_OF_EPOCHS: int = 1000
BATCH_SIZE: int = 512
LEARNING_RATE: float = 0.0001
TEST_SIZE: float = 0.2

X_TYPE: BandType = BandType.BAND_4
Y_TYPE: BandType = BandType.BAND_2

TICKER: TickerOne = TickerOne.BRITANNIA
INTERVAL: str = "1m"

PREV_MODEL_TRAINING: bool = False


def main():
    df = an.get_data_all_df(ticker=TICKER, interval=INTERVAL)

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

            log_dir: str = f"training/logs/{now_datetime} - {Y_TYPE}"

            tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
            terNan = TerminateOnNaN()

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

        for model_num in range(1, 7):
            training_data_custom_evaluation = CustomEvaluation_4(
                X_data=X_train,
                Y_data=Y_train,
                prev_close=train_prev_close,
                y_type=Y_TYPE,
                test_size=TEST_SIZE,
                now_datetime=now_datetime,
                model_num=model_num,
            )

            valid_data_custom_evaluation = CustomEvaluation_4(
                X_data=X_test,
                Y_data=Y_test,
                prev_close=test_prev_close,
                y_type=Y_TYPE,
                test_size=0,
                now_datetime=now_datetime,
                model_num=model_num,
            )

    elif Y_TYPE == BandType.BAND_2:
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
                    km_2.metric_pred_capture_percent,
                    km_2.metric_win_checkpoint,
                ],
            )

            log_dir: str = f"training/logs/{now_datetime} - {Y_TYPE.value.lower()}"

            tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
            terNan = TerminateOnNaN()

            file_path_prefix: str = f"training/models/model - {now_datetime} - {X_TYPE.value.lower()} - {Y_TYPE.value.lower()} - {TICKER.name}"

            mcp_save_1 = ModelCheckpoint(
                f"{file_path_prefix} - modelCheckPoint-1.keras",
                save_best_only=True,
                monitor="loss",
                mode="min",
            )

            mcp_save_2 = ModelCheckpoint(
                f"{file_path_prefix} - modelCheckPoint-2.keras",
                save_best_only=True,
                monitor="val_loss",
                mode="min",
            )

            mcp_save_3 = ModelCheckpoint(
                f"{file_path_prefix} - modelCheckPoint-3.keras",
                save_best_only=True,
                monitor="metric_pred_capture_percent",
                mode="max",
            )

            mcp_save_4 = ModelCheckpoint(
                f"{file_path_prefix} - modelCheckPoint-4.keras",
                save_best_only=True,
                monitor="val_metric_pred_capture_percent",
                mode="max",
            )

            mcp_save_5 = ModelCheckpoint(
                f"{file_path_prefix} - modelCheckPoint-5.keras",
                save_best_only=True,
                monitor="metric_win_checkpoint",
                mode="min",
            )

            mcp_save_6 = ModelCheckpoint(
                f"{file_path_prefix} - modelCheckPoint-6.keras",
                save_best_only=True,
                monitor="val_metric_win_checkpoint",
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

            model.save(f"{file_path_prefix}.keras")

            print("\nmodel : training done. \n")

        print(f"\n\nnow_datetime:\t{now_datetime}\n\n")
        print("-" * 30)

        for model_num in range(1, 7):
            training_data_custom_evaluation = CustomEvaluation(
                ticker=TICKER,
                X_data=X_train,
                Y_data=Y_train,
                prev_close=train_prev_close,
                x_type=X_TYPE,
                y_type=Y_TYPE,
                test_size=TEST_SIZE,
                now_datetime=now_datetime,
                model_num=model_num,
            )

            valid_data_custom_evaluation = CustomEvaluation(
                ticker=TICKER,
                X_data=X_test,
                Y_data=Y_test,
                prev_close=test_prev_close,
                x_type=X_TYPE,
                y_type=Y_TYPE,
                test_size=0,
                now_datetime=now_datetime,
                model_num=model_num,
            )


def suppress_cpu_usage():
    # Get the current process ID
    pid = os.getpid()

    suppress_level: int = 10

    # The command you want to run
    command = f"cpulimit -l {suppress_level} -p {pid}"

    # quit terminal app
    # subprocess.run(["osascript", "-e", 'quit app "Terminal"'])

    # Open a new terminal and run the command
    subprocess.Popen(
        ["osascript", "-e", f'tell app "Terminal" to do script "{command}"'],
    )


def set_globals(file_index_from_end: int):
    global X_TYPE, Y_TYPE, prev_model, TICKER

    # get prev_model
    list_of_files = os.listdir("training/models")

    list_of_files.remove(".DS_Store")

    list_of_files = sorted(
        list_of_files,
        key=lambda x: os.path.getmtime("training/models/" + x),
    )

    if list_of_files:
        latest_file = list_of_files[-file_index_from_end]

        print(latest_file)

        date_str: str = latest_file.split(" - ")[1]
        prev_model = date_str

        x_type_str = latest_file.split(" - ")[2]
        for band_type in BandType:
            if band_type.value == x_type_str:
                X_TYPE = band_type
                break

        y_type_str = latest_file.split(" - ")[3]
        for band_type in BandType:
            if band_type.value == y_type_str:
                Y_TYPE = band_type
                break

        ticker_str = latest_file.split(" - ")[4]
        for ticker in TickerOne:
            if ticker.name == ticker_str:
                TICKER = ticker
                break

    else:
        raise FileNotFoundError("File not found in folder")


if __name__ == "__main__":
    os.system("clear")

    if len(sys.argv) > 1:
        if sys.argv[1] == "true":
            IS_TRAINING_MODEL = True
            suppress_cpu_usage()

        elif sys.argv[1] == "new":
            IS_TRAINING_MODEL = False

            set_globals(1)

        elif sys.argv[1] == "new_prev":
            IS_TRAINING_MODEL = False

            set_globals(7)

    if IS_TRAINING_MODEL and len(sys.argv) == 1:
        suppress_cpu_usage()

    time_1 = time.time()
    main()
    print(f"\ntime taken = {round(time.time() - time_1, 2)} sec\n")
