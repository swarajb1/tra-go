import os
import time

import training_zero as an
from data_loader import DataLoader

from database.enums import BandType, IntervalType, ModelLocationType, TickerOne

TEST_SIZE: float = float(os.getenv("TEST_SIZE"))


def _get_custom_evaluation_class(x_type: BandType, y_type: BandType):
    if y_type not in [BandType.BAND_4, BandType.BAND_2, BandType.BAND_2_1]:
        raise ValueError("Invalid y_type")

    if y_type == BandType.BAND_4:
        from band_4.training_yf_band_4 import CustomEvaluation

    elif y_type == BandType.BAND_2:
        from band_2.training_yf_band_2 import CustomEvaluation

    elif y_type == BandType.BAND_2_1:
        from band_2_1.evaluation import CustomEvaluation

    return CustomEvaluation


def is_valid_model_file_name(file_name: str) -> bool:
    return "modelCheckPoint" in file_name and ".keras" in file_name and not file_name.startswith(".")


def is_file_after_date(file_path: str) -> bool:
    target_date = time.mktime(time.strptime("2024-07-01", "%Y-%m-%d"))

    return os.path.getctime(file_path) > target_date


def evaluate_models(
    model_location_type: ModelLocationType,
    number_of_models: int,
    newly_trained_models: bool = False,
) -> None:
    model_location_prefix: str = model_location_type.value

    list_of_files = os.listdir(model_location_prefix)

    list_of_files = [
        file
        for file in list_of_files
        if is_valid_model_file_name(file) and is_file_after_date(os.path.join(model_location_prefix, file))
    ]

    list_of_files = sorted(list_of_files, key=lambda x: x, reverse=newly_trained_models)

    if not list_of_files:
        raise ValueError("\n\nNo models found in the folder: ", model_location_prefix)

    if len(list_of_files) >= number_of_models:
        list_of_files = list_of_files[:number_of_models]

    if newly_trained_models:
        list_of_files = list_of_files[::-1]

    models_worth_saving: list[str] = []
    models_worth_double_saving: list[str] = []
    models_worth_triple_saving: list[str] = []
    models_worth_not_saving: list[str] = []

    max_250_days_win_value: float = 0

    for file in list_of_files:
        print("\n" * 25, "*" * 280, "\n" * 4, sep="")
        print("Evaluating model:\t", file)

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

        interval_from_model: str = "1m"

        model_checkpoint_num_str = file.split(" - ")[5]
        if "modelCheckPoint" not in model_checkpoint_num_str:
            raise ValueError("modelCheckpoint not found in file name")

        model_checkpoint_num = int(model_checkpoint_num_str.split("-")[1].split(".keras")[0])

        df = an.get_data_all_df(ticker=model_ticker, interval=interval_from_model)

        data_loader = DataLoader(
            ticker=model_ticker,
            interval=IntervalType.MIN_1,
            x_type=model_x_type,
            y_type=model_y_type,
            test_size=TEST_SIZE,
        )

        Y_train_data_real, Y_test_data_real = data_loader.get_real_y_data()

        if model_y_type == BandType.BAND_2_1 and model_x_type == BandType.BAND_4:
            train_prev_close, test_prev_close = data_loader.get_prev_close_data()

            (X_train, Y_train), (X_test, Y_test) = data_loader.get_train_test_split_data()

            # (
            #     (X_train, Y_train, Y_train_full, train_prev_close),
            #     (X_test, Y_test, Y_test_full, test_prev_close),
            # ) = an.train_test_split_lh(
            #     data_df=df,
            #     test_size=TEST_SIZE,
            #     x_type=model_x_type,
            #     interval=interval_from_model,
            # )

            # Y_train = Y_train_full
            # Y_test = Y_test_full

        else:
            (
                (X_train, Y_train, train_prev_close),
                (X_test, Y_test, test_prev_close),
            ) = an.train_test_split(
                data_df=df,
                test_size=TEST_SIZE,
                x_type=model_x_type,
                y_type=model_y_type,
                interval=interval_from_model,
            )

        evaluation_class = _get_custom_evaluation_class(x_type=model_x_type, y_type=model_y_type)

        training_data_custom_evaluation = evaluation_class(
            ticker=model_ticker,
            X_data=X_train,
            Y_data=Y_train,
            Y_data_real=Y_train_data_real,
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
            Y_data=Y_test,
            Y_data_real=Y_test_data_real,
            prev_close=test_prev_close,
            x_type=model_x_type,
            y_type=model_y_type,
            test_size=0,
            now_datetime=model_datetime,
            model_num=model_checkpoint_num,
            model_location_type=model_location_type,
        )

        is_triple_saving: bool = (
            training_data_custom_evaluation.is_model_worth_double_saving
            and valid_data_custom_evaluation.is_model_worth_double_saving
        )

        is_double_saving: bool = not is_triple_saving and (
            training_data_custom_evaluation.is_model_worth_saving
            and valid_data_custom_evaluation.is_model_worth_saving
        )

        is_single_saving: bool = (
            training_data_custom_evaluation.is_model_worth_saving or valid_data_custom_evaluation.is_model_worth_saving
        )

        if is_triple_saving:
            models_worth_triple_saving.append(training_data_custom_evaluation.model_file_name)

        elif is_double_saving:
            models_worth_double_saving.append(training_data_custom_evaluation.model_file_name)

        elif is_single_saving:
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
        print("\t", model_file_name, "\t" * 2, " \033[91m--\033[0m ")

    print("\n\nMODELS WORTH SAVING:")
    for model_file_name in models_worth_saving:
        print("\t", model_file_name, "\t" * 2, " \033[92m++\033[0m ")

    print("\n\nMODELS WORTH DOUBLE SAVING:")
    for model_file_name in models_worth_double_saving:
        print("\t", model_file_name, "\t" * 2, " \033[92m++++\033[0m ")

    print("\n\nMODELS WORTH TRIPLE SAVING:")
    for model_file_name in models_worth_triple_saving:
        print("\t", model_file_name, "\t" * 2, " \033[92m+++++++++++++++\033[0m ")

    print("\n\n")

    # add code to move files into one/double/triple saving folders
    # on prompt if written 'MOVE'
