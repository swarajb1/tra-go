import os
import subprocess
import sys
import time
from datetime import datetime

import keras_model as km
import psutil
import training_zero as an
from band_2.training_yf_band_2 import CustomEvaluation
from band_2_1.evaluation import CustomEvaluation as CustomEvaluation_2_1
from band_4.training_yf_band_4 import CustomEvaluation as CustomEvaluation_4
from keras.callbacks import ModelCheckpoint, TensorBoard, TerminateOnNaN

import tra_go.band_2.keras_model_band_2 as km_2
import tra_go.band_2_1.keras_model as km_21_model
import tra_go.band_4.keras_model_band_4 as km_4
from database.enums import BandType, ModelLocationType, TickerOne

IS_TRAINING_MODEL: bool = True
prev_model: str = "2024-04-08 11-50"


NUMBER_OF_EPOCHS: int = 3
BATCH_SIZE: int = 512
LEARNING_RATE: float = 0.0001
TEST_SIZE: float = 0.2

X_TYPE: BandType = BandType.BAND_4
Y_TYPE: BandType = BandType.BAND_2_1

TICKER: TickerOne = TickerOne.SBIN
INTERVAL: str = "1m"

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


def get_custom_evaluation_class(x_type: BandType, y_type: BandType):
    if y_type == BandType.BAND_4:
        return CustomEvaluation_4
    elif y_type == BandType.BAND_2_1:
        return CustomEvaluation_2_1

    # y_type == BandType.BAND_2:
    return CustomEvaluation


def evaluate_models(
    model_location_type: ModelLocationType,
    number_of_models: int,
    newly_trained_models: bool = False,
) -> None:
    model_location_prefix: str = model_location_type.value

    list_of_files = os.listdir(model_location_prefix)

    list_of_files = [file for file in list_of_files if not file.startswith(".")]

    list_of_files = sorted(list_of_files, key=lambda x: x, reverse=newly_trained_models)

    if not list_of_files:
        print("\n\nNo models found in the folder: ", model_location_prefix)
        return

    if len(list_of_files) >= number_of_models:
        list_of_files = list_of_files[:number_of_models]

    models_worth_saving: list[str] = []
    models_worth_double_saving: list[str] = []
    models_worth_not_saving: list[str] = []

    max_250_days_win_value: float = 0

    for file in list_of_files:
        print("\n" * 4, "*" * 280, "\n" * 4, sep="")
        print("Evaluating model:\t", file)

        if "modelCheckPoint" not in file:
            print(ValueError("modelCheckpoint not found in file name"))
            continue

        model_x_type: BandType
        model_y_type: BandType
        model_ticker: TickerOne
        model_datetime: str
        model_checkpoint_num: int

        model_datetime = file.split(" - ")[1]

        x_type_str = file.split(" - ")[2]
        for band_type in BandType:
            if band_type.value == x_type_str:
                model_x_type = band_type
                break

        y_type_str = file.split(" - ")[3]
        for band_type in BandType:
            if band_type.value == y_type_str:
                model_y_type = band_type
                break

        ticker_str = file.split(" - ")[4]
        for ticker in TickerOne:
            if ticker.name == ticker_str:
                model_ticker = ticker
                break

        model_checkpoint_num_str = file.split(" - ")[5]
        if "modelCheckPoint" not in model_checkpoint_num_str:
            raise ValueError("modelCheckpoint not found in file name")

        model_checkpoint_num = int(model_checkpoint_num_str.split("-")[1].split(".keras")[0])

        df = an.get_data_all_df(ticker=model_ticker, interval=INTERVAL)

        if Y_TYPE == BandType.BAND_2_1:
            (
                (X_train, Y_train, Y_train_full, train_prev_close),
                (X_test, Y_test, Y_test_full, test_prev_close),
            ) = an.train_test_split_lh(
                data_df=df,
                test_size=TEST_SIZE,
                x_type=X_TYPE,
                interval=INTERVAL,
            )
        else:
            (
                (X_train, Y_train, train_prev_close),
                (X_test, Y_test, test_prev_close),
            ) = an.train_test_split(
                data_df=df,
                test_size=TEST_SIZE,
                x_type=model_x_type,
                y_type=model_y_type,
                interval=INTERVAL,
            )

        evaluation_class = get_custom_evaluation_class(x_type=model_x_type, y_type=model_y_type)

        training_data_custom_evaluation = evaluation_class(
            ticker=model_ticker,
            X_data=X_train,
            Y_data=Y_train_full,
            prev_close=train_prev_close,
            x_type=model_x_type,
            y_type=model_y_type,
            test_size=TEST_SIZE,
            now_datetime=model_datetime,
            model_num=model_checkpoint_num,
            model_location_type=model_location_type,
        )

        valid_data_custom_evaluation = evaluation_class(
            ticker=model_ticker,
            X_data=X_test,
            Y_data=Y_test_full,
            prev_close=test_prev_close,
            x_type=model_x_type,
            y_type=model_y_type,
            test_size=0,
            now_datetime=model_datetime,
            model_num=model_checkpoint_num,
            model_location_type=model_location_type,
        )

        if (
            training_data_custom_evaluation.is_model_worth_saving
            and valid_data_custom_evaluation.is_model_worth_saving
        ):
            models_worth_double_saving.append(training_data_custom_evaluation.model_file_name)
        elif (
            training_data_custom_evaluation.is_model_worth_saving or valid_data_custom_evaluation.is_model_worth_saving
        ):
            models_worth_saving.append(training_data_custom_evaluation.model_file_name)
        else:
            models_worth_not_saving.append(training_data_custom_evaluation.model_file_name)

        max_250_days_win_value = max(
            max_250_days_win_value,
            training_data_custom_evaluation.win_250_days,
            valid_data_custom_evaluation.win_250_days,
        )

    print("\n\n", "-" * 280, "\n", sep="")

    print("\nMAX 250 days Win Value achieved:\t", max_250_days_win_value, "%")

    print("\n\nMODELS NOT WORTH SAVING:")
    for model_file_name in models_worth_not_saving:
        print("\t", model_file_name, "\t" * 2, "\033[91m--\033[0m")

    print("\n\nMODELS WORTH SAVING:")
    for model_file_name in models_worth_saving:
        print("\t", model_file_name, "\t" * 2, "\033[92m++\033[0m")

    print("\n\nMODELS WORTH DOUBLE SAVING:")
    for model_file_name in models_worth_double_saving:
        print("\t", model_file_name, "\t" * 2, "\033[92m+++++++++++++++\033[0m")


def suppress_cpu_usage():
    from keras_model import NUMBER_OF_NEURONS

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

    if len(sys.argv) > 1:
        if sys.argv[1] == "true":
            IS_TRAINING_MODEL = True

            suppress_cpu_usage()
            main()

        elif sys.argv[1] == "training":
            IS_TRAINING_MODEL = False

            evaluate_models(model_location_type=ModelLocationType.TRAINED_NEW, number_of_models=6)

        elif sys.argv[1] == "training_new":
            IS_TRAINING_MODEL = False

            evaluate_models(
                model_location_type=ModelLocationType.TRAINED_NEW,
                number_of_models=6,
                newly_trained_models=True,
            )

        elif sys.argv[1] == "saved":
            IS_TRAINING_MODEL = False

            evaluate_models(model_location_type=ModelLocationType.SAVED, number_of_models=10)

    else:
        if IS_TRAINING_MODEL:
            suppress_cpu_usage()

        main()

    print(f"\ntime taken = {round(time.time() - time_1, 2)} sec\n")
