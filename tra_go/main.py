import argparse
import os
import sys

from core.assertions import assert_env_vals
from core.evaluate_models import evaluate_models
from main_training import list_of_tickers, main_training_4_cores
from tf_utils import configure_tensorflow_performance

from database.enums import ModelLocationType


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments, supporting both positional (Makefile) and named args."""
    parser = argparse.ArgumentParser(description="TRA-GO AI Trading System CLI")

    # Positional arguments (for Makefile compatibility)
    parser.add_argument(
        "command",
        nargs="?",
        choices=[
            "train",
            "training_new",
            "eval_trained",
            "saved",
            "saved_double",
            "saved_triple",
            "old",
            "discarded",
        ],
        help="The command to execute (e.g., train, saved)",
    )
    parser.add_argument("num_models", nargs="?", type=int, default=6, help="Number of models to evaluate (default: 6)")
    parser.add_argument(
        "move_files",
        nargs="?",
        choices=["true", "false"],
        default="false",
        help="Whether to move files during evaluation (default: false)",
    )

    # Named arguments (for direct CLI usage)
    parser.add_argument("--num_models", type=int, help="Number of models to evaluate (overrides positional)")
    parser.add_argument("--move_files", action="store_true", help="Whether to move files during evaluation")

    args = parser.parse_args()

    # Override positional with named if provided
    if args.num_models is not None:
        args.num_models = args.num_models
    if args.move_files:
        args.move_files = "true"

    return args


def dispatch_command(args: argparse.Namespace) -> None:
    """Dispatch the parsed command to the appropriate function."""
    command = args.command
    num_models = args.num_models
    move_files = args.move_files == "true"

    if not command:
        # Default behavior when no command is provided
        evaluate_models(
            model_location_type=ModelLocationType.TRAINED_NEW,
            number_of_models=1,
            move_files=False,
        )
        return

    if command == "train":
        for ticker in list_of_tickers * 5:
            main_training_4_cores(ticker)

    elif command == "training_new":
        evaluate_models(
            model_location_type=ModelLocationType.TRAINED_NEW,
            number_of_models=6,
            newly_trained_models=True,
        )

    elif command in ["eval_trained", "saved", "saved_double", "old"]:
        model_type_map = {
            "eval_trained": ModelLocationType.TRAINED_NEW,
            "saved": ModelLocationType.SAVED,
            "saved_double": ModelLocationType.SAVED_DOUBLE,
            "old": ModelLocationType.OLD,
        }
        evaluate_models(
            model_location_type=model_type_map[command],
            number_of_models=num_models,
            move_files=move_files,
        )

    elif command == "saved_triple":
        evaluate_models(
            model_location_type=ModelLocationType.SAVED_TRIPLE,
            number_of_models=num_models,
        )

    elif command == "discarded":
        evaluate_models(
            model_location_type=ModelLocationType.DISCARDED,
            number_of_models=num_models,
            move_files=True,
        )


def main():
    """Main entry point for the TRA-GO application."""
    os.system("clear")  # Optional: Consider removing or making conditional for non-interactive runs

    assert_env_vals()

    configure_tensorflow_performance()

    try:
        args = parse_arguments()
        dispatch_command(args)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
