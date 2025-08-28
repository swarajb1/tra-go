import os
import sys

from core.assertions import assert_env_vals
from core.evaluate_models import evaluate_models
from main_training import (
    list_of_tickers,
    main_training,
    main_training_4_cores,
    parallel_train_tickers,
)

from database.enums import ModelLocationType


def main():
    os.system("clear")

    assert_env_vals()

    if len(sys.argv) > 1:
        number_of_models: int = 6
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
            # suppress_cpu_usage()

            for ticker in list_of_tickers * 1:
                main_training(ticker)

        elif sys.argv[1] == "true_parallel":
            parallel_train_tickers()

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
            evaluate_models(
                model_location_type=ModelLocationType.SAVED_DOUBLE,
                number_of_models=number_of_models,
                move_files=move_files,
            )

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
            raise ValueError("1st argument, model is not correct")

    else:
        # main_training()
        main_training_4_cores()

        # evaluate_models(
        #     model_location_type=ModelLocationType.SAVED,
        #     number_of_models=6,
        #     move_files=False,
        # )


if __name__ == "__main__":
    main()
