import os
import time

import training_zero as an
from core.config import settings
from core.logger import logger
from core.model_utils import move_model_to_discarded
from data_loader import DataLoader

from database.enums import BandType, IntervalType, ModelLocationType, TickerOne


def _get_custom_evaluation_class(x_type: BandType, y_type: BandType):
    if y_type not in [BandType.BAND_4, BandType.BAND_2, BandType.BAND_2_1, BandType.BAND_1_1]:
        raise ValueError(f"Invalid y_type: {y_type}")

    elif y_type == BandType.BAND_2:
        from band_2.training_yf_band_2 import CustomEvaluation

    elif y_type in [BandType.BAND_2_1, BandType.BAND_1_1]:
        from band_2_1.evaluation import CustomEvaluation

    return CustomEvaluation


def is_valid_model_file_name(file_name: str) -> bool:
    return ".keras" in file_name and not file_name.startswith(".")


def is_file_after_date(file_path: str) -> bool:
    target_date = time.mktime(time.strptime("2024-07-01", "%Y-%m-%d"))

    return os.path.getctime(file_path) > target_date


def evaluate_models(
    model_location_type: ModelLocationType,
    number_of_models: int = 6,
    newly_trained_models: bool = False,
    move_files: bool = False,
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
        raise ValueError(f"\n\nNo models found in the folder: {model_location_prefix}")

    if len(list_of_files) >= number_of_models:
        list_of_files = list_of_files[:number_of_models]

    if newly_trained_models:
        list_of_files = list_of_files[::-1]

    models_worth_saving: list[str] = []
    models_worth_double_saving: list[str] = []
    models_worth_triple_saving: list[str] = []
    models_worth_not_saving: list[str] = []

    max_250_days_win_value: float = 0
    max_win_pred_capture_percent_value: float = 0
    max_250_days_simulation_value: float = 0
    max_all_simulations_max_250_days: float = 0

    for index, file_name in enumerate(list_of_files):
        print("\n" * 25, "*" * 280, "\n" * 4, sep="")
        print(f"{index+1}/{len(list_of_files)} - Evaluating model:\t{file_name}")

        file_name_1: str = file_name[: -(len(".keras"))]

        model_x_type: BandType
        model_y_type: BandType
        model_ticker: TickerOne

        model_interval: IntervalType = IntervalType.MIN_1

        x_type_str = file_name_1.split(" - ")[2]
        for band_type in BandType:
            if band_type.value == x_type_str:
                model_x_type = band_type
                break

        y_type_str = file_name_1.split(" - ")[3]
        for band_type in BandType:
            if band_type.value == y_type_str:
                model_y_type = band_type
                break

        ticker_str = file_name_1.split(" - ")[4]
        for ticker in TickerOne:
            if ticker.name == ticker_str:
                model_ticker = ticker
                break

        df = an.get_data_all_df(ticker=model_ticker, interval=model_interval.value)

        data_loader = DataLoader(
            ticker=model_ticker,
            interval=IntervalType.MIN_1,
            x_type=model_x_type,
            y_type=model_y_type,
            test_size=settings.TEST_SIZE,
        )

        Y_train_data_real, Y_test_data_real = data_loader.get_real_y_data()

        if model_y_type in [BandType.BAND_2_1, BandType.BAND_1_1]:
            train_prev_close, test_prev_close = data_loader.get_prev_close_data()

            (X_train, Y_train), (X_test, Y_test) = data_loader.get_train_test_split_data()

        else:
            (
                (X_train, Y_train, train_prev_close),
                (X_test, Y_test, test_prev_close),
            ) = an.train_test_split(
                data_df=df,
                test_size=settings.TEST_SIZE,
                x_type=model_x_type,
                y_type=model_y_type,
                interval=model_interval.value,
            )

        evaluation_class = _get_custom_evaluation_class(x_type=model_x_type, y_type=model_y_type)

        # Create training evaluation; if creation fails (e.g. model load error),
        # move the problematic model file to the discarded folder and skip it.
        try:
            training_data_custom_evaluation = evaluation_class(
                ticker=model_ticker,
                X_data=X_train,
                Y_data=Y_train,
                Y_data_real=Y_train_data_real,
                prev_day_close=train_prev_close,
                x_type=model_x_type,
                y_type=model_y_type,
                test_size=settings.TEST_SIZE,
                model_file_name=file_name,
                model_location_type=model_location_type,
            )
        except Exception as err:  # pylint: disable=broad-except
            logger.warning(
                "Skipping model due to evaluation error: %s | error: %s",
                file_name,
                err,
            )

            # Attempt to move the file to the discarded folder
            move_model_to_discarded(model_location_prefix, file_name)

            # Skip this file and continue with next
            continue

        # Create validation evaluation; if it fails, treat similarly and skip file
        try:
            valid_data_custom_evaluation = evaluation_class(
                ticker=model_ticker,
                X_data=X_test,
                Y_data=Y_test,
                Y_data_real=Y_test_data_real,
                prev_day_close=test_prev_close,
                x_type=model_x_type,
                y_type=model_y_type,
                test_size=0,
                model_file_name=file_name,
                model_location_type=model_location_type,
            )
        except Exception as err:  # pylint: disable=broad-except
            logger.warning(
                "Skipping model due to validation evaluation error: %s | error: %s",
                file_name,
                err,
            )
            # Attempt to move the file to the discarded folder
            move_model_to_discarded(model_location_prefix, file_name)

            continue

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

        max_win_pred_capture_percent_value = max(
            max_win_pred_capture_percent_value,
            training_data_custom_evaluation.win_pred_capture_percent,
            valid_data_custom_evaluation.win_pred_capture_percent,
        )

        max_250_days_simulation_value = max(
            max_250_days_simulation_value,
            training_data_custom_evaluation.simulation_250_days,
            valid_data_custom_evaluation.simulation_250_days,
        )

        max_all_simulations_max_250_days = max(
            max_all_simulations_max_250_days,
            training_data_custom_evaluation.all_simulations_max_250_days,
            valid_data_custom_evaluation.all_simulations_max_250_days,
        )

        # move files into discarded/saved/saved_double folders
        if move_files and model_location_type in [
            ModelLocationType.TRAINED_NEW,
            ModelLocationType.SAVED_DOUBLE,
            ModelLocationType.SAVED,
            ModelLocationType.OLD,
            ModelLocationType.DISCARDED,
        ]:
            destination_model_location_type: ModelLocationType = model_location_type

            if is_triple_saving:
                destination_model_location_type = ModelLocationType.SAVED_TRIPLE
            elif is_double_saving:
                destination_model_location_type = ModelLocationType.SAVED_DOUBLE
            elif is_single_saving:
                destination_model_location_type = ModelLocationType.SAVED
            else:
                destination_model_location_type = ModelLocationType.DISCARDED

            if model_location_type == destination_model_location_type:
                continue

            source_file: str = os.path.join(model_location_type.value, file_name)
            destination_file: str = os.path.join(destination_model_location_type.value, file_name)

            os.rename(source_file, destination_file)

    print("\n\n", "-" * 280, "\n", sep="")

    print("\nMAX 250 days Win Value achieved:\t\t", f"{max_250_days_win_value:.2f}", "%")
    print("\nMAX Win Pred Capture Percent achieved:\t\t", f"{max_win_pred_capture_percent_value:.2f}", "%")
    print("\nMAX 250 Days Simulation Value:\t\t\t", f"{max_250_days_simulation_value:.2f}", "%")
    print("\nMAX All Possible 250 Days Simulation Value:\t", f"{max_all_simulations_max_250_days:.2f}", "%")

    print(f"\n\n\nMODELS NOT WORTH SAVING: \t\t[{len(models_worth_not_saving)}]\n")
    for model_file_name in models_worth_not_saving:
        print("\t", model_file_name, " " * (95 - len(model_file_name)), " \033[91m--\033[0m ")

    print(f"\n\nMODELS WORTH SAVING: \t\t\t[{len(models_worth_saving)}]\n")
    for model_file_name in models_worth_saving:
        print("\t", model_file_name, " " * (95 - len(model_file_name)), " \033[92m++\033[0m ")

    print(f"\n\nMODELS WORTH DOUBLE SAVING: \t\t[{len(models_worth_double_saving)}]\n")
    for model_file_name in models_worth_double_saving:
        print("\t", model_file_name, " " * (95 - len(model_file_name)), " \033[92m++++\033[0m ")

    print(f"\n\nMODELS WORTH TRIPLE SAVING: \t\t[{len(models_worth_triple_saving)}]\n")
    for model_file_name in models_worth_triple_saving:
        print("\t", model_file_name, " " * (95 - len(model_file_name)), " \033[92m+++++++++++++++\033[0m ")

    print("\n\n")
