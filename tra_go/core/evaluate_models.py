import concurrent.futures
import gc
import logging
import os
import time
from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, NamedTuple, Optional

import training_zero as an
from core.config import settings
from data_loader import DataLoader

from database.enums import BandType, IntervalType, ModelLocationType, TickerOne

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Constants
DEFAULT_DATE_FILTER = "2024-07-01"
DEFAULT_MAX_WORKERS = 4
MAX_CACHE_SIZE = 8
BANNER_SEPARATOR = "=" * 80
FILE_NAME_MIN_LENGTH = 5


class ModelMetadata(NamedTuple):
    """Structured container for model metadata."""

    x_type: BandType
    y_type: BandType
    ticker: TickerOne


class ModelCategory(str, Enum):
    """Category classification for evaluated models."""

    NOT_SAVING = "not_saving"
    SAVING = "saving"
    DOUBLE_SAVING = "double_saving"
    TRIPLE_SAVING = "triple_saving"


class ModelEvaluationResult(NamedTuple):
    """Container for model evaluation results."""

    category: ModelCategory
    metrics: dict[str, float]
    file_name: str


class ModelEvaluationMetrics(NamedTuple):
    """Container for model evaluation metrics."""

    max_250_days_win_value: float
    max_win_pred_capture_percent_value: float
    max_250_days_simulation_value: float
    max_all_simulations_max_250_days: float


@contextmanager
def timed_operation(description: str) -> Generator[None, None, None]:
    """Context manager for timing operations.

    Args:
        description: Description of the operation being timed.

    Yields:
        None
    """
    start_time = time.time()
    try:
        yield
    finally:
        elapsed_time = time.time() - start_time
        logger.info(f"{description} completed in {elapsed_time:.2f} seconds")


@contextmanager
def memory_management() -> Generator[None, None, None]:
    """Context manager for memory management.

    Ensures garbage collection after operation completes.

    Yields:
        None
    """
    try:
        yield
    finally:
        # Force garbage collection to free memory
        gc.collect()


@lru_cache(maxsize=MAX_CACHE_SIZE)
def _get_custom_evaluation_class(y_type: BandType) -> type:
    """Get the appropriate CustomEvaluation class based on band type.

    Uses lru_cache for efficiency when the same y_type is requested multiple times.

    Args:
        y_type: The output band type.

    Returns:
        The CustomEvaluation class appropriate for the given band type.

    Raises:
        ValueError: If an invalid y_type is provided.
    """
    valid_y_types = {BandType.BAND_4, BandType.BAND_2, BandType.BAND_2_1, BandType.BAND_1_1}
    if y_type not in valid_y_types:
        raise ValueError(f"Invalid y_type: {y_type}. Must be one of {valid_y_types}")

    if y_type == BandType.BAND_4:
        from band_4.training_yf_band_4 import CustomEvaluation
    elif y_type == BandType.BAND_2:
        from band_2.training_yf_band_2 import CustomEvaluation
    elif y_type in {BandType.BAND_2_1, BandType.BAND_1_1}:
        from band_2_1.evaluation import CustomEvaluation

    return CustomEvaluation


def is_valid_model_file_name(file_name: str) -> bool:
    """Check if a file name is a valid model file.

    Args:
        file_name: The file name to check.

    Returns:
        True if the file name is a valid model file, False otherwise.
    """
    return file_name.endswith(".keras") and not file_name.startswith(".") and len(file_name) > FILE_NAME_MIN_LENGTH


def is_file_after_date(file_path: Path, target_date_str: str = DEFAULT_DATE_FILTER) -> bool:
    """Check if a file was created after a specific date.

    Args:
        file_path: The path to the file.
        target_date_str: The target date string in YYYY-MM-DD format.

    Returns:
        True if the file was created after the target date, False otherwise.
    """
    target_timestamp = time.mktime(time.strptime(target_date_str, "%Y-%m-%d"))

    try:
        creation_time = os.path.getctime(file_path)
        return creation_time > target_timestamp
    except (FileNotFoundError, PermissionError) as e:
        logger.warning(f"Could not check creation time for {file_path}: {e}")
        return False


def filter_model_files(
    directory: Path,
    max_files: int = 0,
    reverse_sort: bool = False,
    date_filter: str = DEFAULT_DATE_FILTER,
) -> list[str]:
    """Filter and sort model files from a directory.

    Args:
        directory: Directory to search for model files.
        max_files: Maximum number of files to return (0 for all).
        reverse_sort: Whether to sort in reverse order.
        date_filter: Only include files created after this date.

    Returns:
        List of filtered and sorted model file names.
    """
    # Get all files in directory
    try:
        all_files = list(directory.iterdir())
    except (FileNotFoundError, PermissionError) as e:
        logger.error(f"Could not access directory {directory}: {e}")
        return []

    # Filter valid model files created after date_filter
    valid_files = [
        file.name
        for file in all_files
        if file.is_file() and is_valid_model_file_name(file.name) and is_file_after_date(file, date_filter)
    ]

    # Sort files
    valid_files.sort(reverse=reverse_sort)

    # Limit to max_files if specified
    if max_files > 0 and len(valid_files) > max_files:
        valid_files = valid_files[:max_files]

    return valid_files


def parse_model_metadata(file_name: str) -> ModelMetadata:
    """Parse model metadata from a file name.

    Args:
        file_name: The file name to parse.

    Returns:
        ModelMetadata object containing extracted metadata.

    Raises:
        ValueError: If metadata cannot be extracted properly.
    """
    # Remove .keras extension
    file_base = file_name.replace(".keras", "")

    # Split by separator
    parts = file_base.split(" - ")

    if len(parts) < 5:
        raise ValueError(f"Invalid model file name format: {file_name}")

    try:
        # Extract metadata components
        x_type_str = parts[2]
        y_type_str = parts[3]
        ticker_str = parts[4]

        # Find matching BandType for x_type
        x_type = next((bt for bt in BandType if bt.value == x_type_str), None)
        if x_type is None:
            raise ValueError(f"Unknown x_type in model file: {x_type_str}")

        # Find matching BandType for y_type
        y_type = next((bt for bt in BandType if bt.value == y_type_str), None)
        if y_type is None:
            raise ValueError(f"Unknown y_type in model file: {y_type_str}")

        # Find matching TickerOne for ticker
        ticker = next((t for t in TickerOne if t.name == ticker_str), None)
        if ticker is None:
            raise ValueError(f"Unknown ticker in model file: {ticker_str}")

        return ModelMetadata(x_type=x_type, y_type=y_type, ticker=ticker)

    except (IndexError, ValueError) as e:
        raise ValueError(f"Error parsing metadata from {file_name}: {e}")


def evaluate_models(
    model_location_type: ModelLocationType,
    number_of_models: int = 6,
    newly_trained_models: bool = False,
    move_files: bool = False,
    parallel: bool = False,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> None:
    """Evaluate machine learning models and categorize them based on performance.

    Args:
        model_location_type: The location type of the models to evaluate.
        number_of_models: The maximum number of models to evaluate.
        newly_trained_models: Whether to prioritize newly trained models.
        move_files: Whether to move model files based on evaluation results.
        parallel: Whether to evaluate models in parallel.
        max_workers: Maximum number of worker threads for parallel evaluation.

    Raises:
        ValueError: If no models are found in the specified location.
    """
    with timed_operation("Total model evaluation"):
        # Get directory path
        model_location_path = Path(model_location_type.value)

        # Filter and sort model files
        with timed_operation("Model file filtering"):
            valid_model_files = filter_model_files(
                directory=model_location_path,
                max_files=number_of_models,
                reverse_sort=newly_trained_models,
            )

        if not valid_model_files:
            logger.error(f"No models found in the folder: {model_location_path}")
            raise ValueError(f"No models found in the folder: {model_location_path}")

        logger.info(f"Found {len(valid_model_files)} model(s) to evaluate")

        # Initialize model categorization dict using defaultdict
        model_categories = defaultdict(list)

        # Initialize metrics tracking
        max_metrics = ModelEvaluationMetrics(
            max_250_days_win_value=0.0,
            max_win_pred_capture_percent_value=0.0,
            max_250_days_simulation_value=0.0,
            max_all_simulations_max_250_days=0.0,
        )

        # Create a banner to separate evaluations visually
        evaluation_banner = "\n" * 3 + "*" * 80 + "\n" * 2

        # Evaluate models
        with timed_operation("Model evaluations"), memory_management():
            evaluation_results = _run_model_evaluations(
                valid_model_files,
                model_location_type,
                parallel,
                max_workers,
                evaluation_banner,
            )

        # Process evaluation results
        model_categories, updated_max_metrics = _process_evaluation_results(
            evaluation_results,
            model_categories,
            max_metrics._asdict(),
        )

        # Move files if requested
        if move_files:
            with timed_operation("File movement"):
                _move_model_files(model_location_type=model_location_type, model_categories=model_categories)

        # Print evaluation summary
        _print_evaluation_summary(
            max_metrics=updated_max_metrics,
            models_worth_saving=model_categories[ModelCategory.SAVING],
            models_worth_double_saving=model_categories[ModelCategory.DOUBLE_SAVING],
            models_worth_triple_saving=model_categories[ModelCategory.TRIPLE_SAVING],
            models_worth_not_saving=model_categories[ModelCategory.NOT_SAVING],
        )


def _run_model_evaluations(
    model_files: list[str],
    model_location_type: ModelLocationType,
    parallel: bool,
    max_workers: int,
    evaluation_banner: str,
) -> list[ModelEvaluationResult]:
    """Run evaluations on a list of model files.

    Args:
        model_files: List of model file names to evaluate.
        model_location_type: Location type of the models.
        parallel: Whether to evaluate models in parallel.
        max_workers: Maximum number of worker threads for parallel evaluation.
        evaluation_banner: Banner to print before each evaluation.

    Returns:
        List of evaluation results.
    """
    evaluation_results = []

    if parallel and len(model_files) > 1:
        # Parallel evaluation
        logger.info(f"Evaluating {len(model_files)} models in parallel with {max_workers} workers")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            evaluation_func = partial(_evaluate_model_wrapper, model_location_type=model_location_type)
            future_to_file = {executor.submit(evaluation_func, file_name): file_name for file_name in model_files}

            for i, future in enumerate(concurrent.futures.as_completed(future_to_file)):
                file_name = future_to_file[future]
                print(evaluation_banner)
                print(f"{i+1}/{len(model_files)} - Completed evaluation of model: {file_name}")

                try:
                    result = future.result()
                    if result:
                        evaluation_results.append(result)
                except Exception as e:
                    logger.error(f"Error evaluating model {file_name}: {e}")
    else:
        # Sequential evaluation
        for index, file_name in enumerate(model_files):
            print(evaluation_banner)
            print(f"{index+1}/{len(model_files)} - Evaluating model: {file_name}")

            try:
                result = _evaluate_model_wrapper(file_name, model_location_type)
                if result:
                    evaluation_results.append(result)
            except Exception as e:
                logger.error(f"Error evaluating model {file_name}: {e}")

    return evaluation_results


def _process_evaluation_results(
    evaluation_results: list[ModelEvaluationResult],
    model_categories: dict[ModelCategory, list[str]],
    max_metrics: dict[str, float],
) -> tuple[dict[ModelCategory, list[str]], dict[str, float]]:
    """Process evaluation results to update metrics and categorize models.

    Args:
        evaluation_results: List of evaluation results.
        model_categories: Dictionary mapping categories to lists of model file names.
        max_metrics: Dictionary of maximum metric values.

    Returns:
        Tuple of updated model categories and max metrics.
    """
    # Process evaluation results
    for result in evaluation_results:
        # Update max metrics
        for metric_key, metric_value in result.metrics.items():
            if metric_key in max_metrics:
                max_metrics[metric_key] = max(max_metrics[metric_key], metric_value)

        # Categorize model
        model_categories[result.category].append(result.file_name)

    return model_categories, max_metrics


def _evaluate_model_wrapper(file_name: str, model_location_type: ModelLocationType) -> Optional[ModelEvaluationResult]:
    """Wrapper function for model evaluation to handle exceptions.

    Args:
        file_name: Name of the model file.
        model_location_type: Location type of the model.

    Returns:
        ModelEvaluationResult if successful, None otherwise.
    """
    try:
        logger.info(f"Starting evaluation of model: {file_name}")

        with memory_management():
            # Parse model metadata
            model_metadata = parse_model_metadata(file_name)

            # Evaluate model
            evaluation_result = _evaluate_single_model(
                file_name=file_name,
                model_location_type=model_location_type,
                model_metadata=model_metadata,
            )

            category = ModelCategory(evaluation_result["category"])

            logger.info(f"Completed evaluation of model: {file_name}, category: {category}")

            return ModelEvaluationResult(category=category, metrics=evaluation_result["metrics"], file_name=file_name)
    except Exception as e:
        logger.error(f"Error evaluating model {file_name}: {e}", exc_info=True)
        return None


def _evaluate_single_model(
    file_name: str,
    model_location_type: ModelLocationType,
    model_metadata: ModelMetadata,
) -> dict[str, Any]:
    """Evaluate a single model and return the results.

    Args:
        file_name: The model file name.
        model_location_type: The model location type.
        model_metadata: The model metadata.

    Returns:
        Dictionary containing evaluation results and metrics.
    """
    model_x_type, model_y_type, model_ticker = model_metadata
    model_interval = IntervalType.MIN_1

    # Load and prepare data
    with timed_operation(f"Data loading for {file_name}"):
        df = an.get_data_all_df(ticker=model_ticker, interval=model_interval.value)

        data_loader = DataLoader(
            ticker=model_ticker,
            interval=IntervalType.MIN_1,
            x_type=model_x_type,
            y_type=model_y_type,
            test_size=settings.TEST_SIZE,
        )

        Y_train_data_real, Y_test_data_real = data_loader.get_real_y_data()

        # Prepare data split based on model type
        if model_y_type in {BandType.BAND_2_1, BandType.BAND_1_1}:
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

    # Get appropriate evaluation class and create instances
    with timed_operation(f"Model evaluation for {file_name}"):
        evaluation_class = _get_custom_evaluation_class(y_type=model_y_type)

        training_evaluation = evaluation_class(
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

        validation_evaluation = evaluation_class(
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

    # Determine model quality
    is_triple_saving = (
        training_evaluation.is_model_worth_double_saving and validation_evaluation.is_model_worth_double_saving
    )

    is_double_saving = not is_triple_saving and (
        training_evaluation.is_model_worth_saving and validation_evaluation.is_model_worth_saving
    )

    is_single_saving = training_evaluation.is_model_worth_saving or validation_evaluation.is_model_worth_saving

    # Determine category
    if is_triple_saving:
        category = ModelCategory.TRIPLE_SAVING
    elif is_double_saving:
        category = ModelCategory.DOUBLE_SAVING
    elif is_single_saving:
        category = ModelCategory.SAVING
    else:
        category = ModelCategory.NOT_SAVING

    # Compile metrics
    metrics = {
        "max_250_days_win_value": max(
            training_evaluation.win_250_days,
            validation_evaluation.win_250_days,
        ),
        "max_win_pred_capture_percent_value": max(
            training_evaluation.win_pred_capture_percent,
            validation_evaluation.win_pred_capture_percent,
        ),
        "max_250_days_simulation_value": max(
            training_evaluation.simulation_250_days,
            validation_evaluation.simulation_250_days,
        ),
        "max_all_simulations_max_250_days": max(
            training_evaluation.all_simulations_max_250_days,
            validation_evaluation.all_simulations_max_250_days,
        ),
    }

    return {"category": category, "metrics": metrics}


def _move_model_files(
    model_location_type: ModelLocationType,
    model_categories: dict[ModelCategory, list[str]],
) -> None:
    """Move model files to appropriate destinations based on categories.

    Args:
        model_location_type: Current location type of models.
        model_categories: Dictionary mapping categories to lists of model file names.
    """
    # Define mapping of categories to destination types
    category_to_destination = {
        ModelCategory.TRIPLE_SAVING: ModelLocationType.SAVED_TRIPLE,
        ModelCategory.DOUBLE_SAVING: ModelLocationType.SAVED_DOUBLE,
        ModelCategory.SAVING: ModelLocationType.SAVED,
        ModelCategory.NOT_SAVING: ModelLocationType.DISCARDED,
    }

    # Skip if model location type isn't one of the movable types
    movable_locations = {
        ModelLocationType.TRAINED_NEW,
        ModelLocationType.SAVED_DOUBLE,
        ModelLocationType.SAVED,
        ModelLocationType.OLD,
        ModelLocationType.DISCARDED,
    }

    if model_location_type not in movable_locations:
        logger.info(f"Skipping file movement: {model_location_type} is not a movable location")
        return

    moved_count = 0
    error_count = 0

    # Iterate through categories and move files
    for category, files in model_categories.items():
        if not files:
            continue

        destination_type = category_to_destination[category]

        # Skip if source and destination are the same
        if model_location_type == destination_type:
            continue

        # Ensure destination directory exists
        destination_dir = Path(destination_type.value)
        destination_dir.mkdir(exist_ok=True, parents=True)

        logger.info(f"Moving {len(files)} files to {destination_type.name}")

        # Move files
        for file_name in files:
            source_file = Path(model_location_type.value) / file_name
            destination_file = destination_dir / file_name

            try:
                if not source_file.exists():
                    logger.warning(f"Source file does not exist: {source_file}")
                    continue

                if destination_file.exists():
                    logger.warning(f"Destination file already exists: {destination_file}")
                    # Generate a unique name by appending timestamp
                    timestamp = int(time.time())
                    new_name = f"{file_name.replace('.keras', '')}_{timestamp}.keras"
                    destination_file = destination_dir / new_name

                # Move the file
                os.rename(source_file, destination_file)
                logger.info(f"Moved {file_name} to {destination_type.name}")
                moved_count += 1

            except Exception as e:
                logger.error(f"Error moving file {file_name}: {e}")
                error_count += 1

    logger.info(f"File movement summary: {moved_count} moved, {error_count} errors")


def _print_evaluation_summary(
    max_metrics: dict[str, float],
    models_worth_saving: list[str],
    models_worth_double_saving: list[str],
    models_worth_triple_saving: list[str],
    models_worth_not_saving: list[str],
) -> None:
    """Print a summary of the evaluation results.

    Args:
        max_metrics: Dictionary containing the maximum metrics values.
        models_worth_saving: List of models worth saving.
        models_worth_double_saving: List of models worth double saving.
        models_worth_triple_saving: List of models worth triple saving.
        models_worth_not_saving: List of models not worth saving.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("\n\n", BANNER_SEPARATOR, "\n", sep="")
    print(f"EVALUATION SUMMARY ({timestamp})")
    print("-" * 80)

    # Print max metrics
    metric_format = "{} {:>8.2f}%"

    print("\nMAXIMUM METRICS:")
    print(metric_format.format("250 days Win Value:", max_metrics["max_250_days_win_value"]))
    print(metric_format.format("Win Pred Capture Percent:", max_metrics["max_win_pred_capture_percent_value"]))
    print(metric_format.format("250 Days Simulation Value:", max_metrics["max_250_days_simulation_value"]))
    print(metric_format.format("All Possible 250 Days Simulation:", max_metrics["max_all_simulations_max_250_days"]))

    # Helper function to print model lists with formatting
    def print_model_list(title: str, models: list[str], indicator: str) -> None:
        if not models:
            return

        print(f"\n\n{title}: [{len(models)}]")
        print("-" * 80)

        max_name_len = 80  # Limit very long names for better formatting

        # Group models by metadata for better organization
        grouped_models = _group_models_by_metadata(models)

        for group, group_models in grouped_models.items():
            if len(grouped_models) > 1:  # Only show group headers if there are multiple groups
                print(f"\n  {group} ({len(group_models)} models):")

            for model_file_name in sorted(group_models):
                display_name = model_file_name[:max_name_len]
                padding = " " * max(1, max_name_len - len(display_name))
                print(f"  {display_name}{padding} {indicator}")

    # Print model categories
    print_model_list("MODELS NOT WORTH SAVING", models_worth_not_saving, "\033[91m--\033[0m")
    print_model_list("MODELS WORTH SAVING", models_worth_saving, "\033[92m++\033[0m")
    print_model_list("MODELS WORTH DOUBLE SAVING", models_worth_double_saving, "\033[92m++++\033[0m")
    print_model_list("MODELS WORTH TRIPLE SAVING", models_worth_triple_saving, "\033[92m+++++++++++++++\033[0m")

    # Print summary counts
    total_models = (
        len(models_worth_saving)
        + len(models_worth_double_saving)
        + len(models_worth_triple_saving)
        + len(models_worth_not_saving)
    )

    total_worth_saving = len(models_worth_saving) + len(models_worth_double_saving) + len(models_worth_triple_saving)

    print("\n\nSUMMARY:")
    print(f"  Total models evaluated: {total_models}")
    print(f"  Models worth saving:    {total_worth_saving}")
    print(f"  Models discarded:       {len(models_worth_not_saving)}")

    # Calculate success rate
    if total_models > 0:
        success_rate = 100 * total_worth_saving / total_models
        print(f"  Success rate:          {success_rate:.1f}%")

    print("\n" + BANNER_SEPARATOR + "\n")


def _group_models_by_metadata(models: list[str]) -> dict[str, list[str]]:
    """Group models by metadata for better organization in summary.

    Args:
        models: List of model file names.

    Returns:
        Dictionary mapping metadata groups to lists of model file names.
    """
    grouped_models = defaultdict(list)

    for model in models:
        try:
            metadata = parse_model_metadata(model)
            group_key = f"{metadata.ticker.name} - {metadata.y_type.value}"
            grouped_models[group_key].append(model)
        except ValueError:
            # If we can't parse metadata, put in "Other" group
            grouped_models["Other"].append(model)

    # If only one group, return without grouping
    if len(grouped_models) <= 1:
        return {"All": models}

    return grouped_models
