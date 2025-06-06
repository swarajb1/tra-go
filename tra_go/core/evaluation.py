import logging
import pprint
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from core.win_graph import WinGraph
from keras.models import load_model
from keras.utils import custom_object_scope
from numpy.typing import NDArray
from tensorflow.keras.models import Model

from database.enums import BandType, ModelLocationType, TickerOne

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class CoreEvaluation:
    """Base class for model evaluation functionality.

    This class handles the core evaluation logic for machine learning models,
    including loading models, processing data, and generating win graphs.
    """

    def __init__(
        self,
        ticker: TickerOne,
        X_data: NDArray[np.float64],
        Y_data: NDArray[np.float64],
        Y_data_real: NDArray[np.float64],
        prev_day_close: NDArray[np.float64],
        x_type: BandType,
        y_type: BandType,
        test_size: float,
        model_file_name: str,
        model_location_type: ModelLocationType,
    ) -> None:
        """Initialize the CoreEvaluation class.

        Args:
            ticker: The ticker symbol for the evaluation.
            X_data: Input features data.
            Y_data: Target data.
            Y_data_real: Real OHLC price data.
            prev_day_close: Previous day's closing prices.
            x_type: The band type for input features.
            y_type: The band type for target values.
            test_size: The proportion of data used for testing.
            model_file_name: The filename of the model to evaluate.
            model_location_type: The location type where the model is stored.

        Raises:
            AssertionError: If input data doesn't meet expected shape requirements.
            FileNotFoundError: If the model file doesn't exist.
        """
        self._validate_input_data(Y_data_real, prev_day_close)

        # Store input parameters
        self.ticker = ticker
        self.x_data = X_data
        self.y_data = Y_data
        self.y_data_real = Y_data_real
        self.prev_day_close = prev_day_close
        self.x_type = x_type
        self.y_type = y_type
        self.test_size = test_size
        self.model_location_type = model_location_type
        self.model_file_name = model_file_name

        # Initialize result metrics
        self.number_of_days: int = self.x_data.shape[0]
        self.is_model_worth_saving: bool = False
        self.is_model_worth_double_saving: bool = False
        self.win_250_days: float = 0.0
        self.win_pred_capture_percent: float = 0.0
        self.simulation_250_days: float = 0.0
        self.all_simulations_max_250_days: float = 0.0

        # Set model file path and validate it exists
        self.model_file_path = self._get_model_file_path()

        # Print evaluation start message
        self._print_start_of_evaluation_message()

        # Process the input data
        self._preprocess_data()

    def _validate_input_data(self, Y_data_real: NDArray[np.float64], prev_day_close: NDArray[np.float64]) -> None:
        """Validate input data shapes.

        Args:
            Y_data_real: Real OHLC price data.
            prev_day_close: Previous day's closing prices.

        Raises:
            AssertionError: If data doesn't meet expected shape requirements.
        """
        assert Y_data_real.ndim == 3, "Y Real data array must be 3-dimensional"
        assert Y_data_real.shape[2] == 4, "Y Real data .shape[2] must be 4"
        assert prev_day_close.ndim == 1, "Prev Close data array must be 1-dimensional"

    def _get_model_file_path(self) -> Path:
        """Get the full path to the model file.

        Returns:
            Path to the model file.

        Raises:
            FileNotFoundError: If the model file doesn't exist.
        """
        model_path = Path(self.model_location_type.value) / self.model_file_name
        if not model_path.exists():
            error_msg = f"Model file not found at: {model_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        return model_path

    def _print_start_of_evaluation_message(self) -> None:
        """Print a message indicating the start of evaluation."""
        if self.test_size > 0:
            logger.info("Evaluating TRAINING data")
        else:
            separator = "_" * 80
            logger.info(f"\n\n{separator}\n\nEvaluating VALIDATION data")

    def _preprocess_data(self) -> None:
        """Preprocess the data before evaluation."""
        self._fill_gaps_in_y_real_data()

    def load_model(
        self,
        custom_objects: Optional[dict[str, Any]] = None,
        print_summary: bool = False,
        print_detailed_config: bool = False,
    ) -> Model:
        """Load the Keras model from file.

        Args:
            custom_objects: Dictionary mapping custom object names to their implementations.
            print_summary: Whether to print model summary.
            print_detailed_config: Whether to print detailed model configuration.

        Returns:
            The loaded Keras model.
        """
        if custom_objects is None:
            custom_objects = {}

        logger.info(f"Loading model from {self.model_file_path}")

        with custom_object_scope(custom_objects):
            model = load_model(self.model_file_path)

        if print_summary:
            model.summary()

        if print_detailed_config:
            model_config = model.get_config()
            pprint.pprint(model_config)

        return model

    def _fill_gaps_in_y_real_data(self) -> None:
        """Fill gaps in the real data.

        This method ensures continuity in the time series data by filling
        gaps between adjacent ticks.
        """
        logger.info("Filling gaps in Y Real data")

        # Make a copy to avoid modifying the original
        y_data_real_copy = deepcopy(self.y_data_real)

        # Process each day and tick
        for day in range(y_data_real_copy.shape[0]):
            for tick in range(1, y_data_real_copy.shape[1]):  # Skip first tick
                prev_tick_min = y_data_real_copy[day, tick - 1, 2]
                prev_tick_max = y_data_real_copy[day, tick - 1, 1]

                next_tick_min = y_data_real_copy[day, tick, 2]
                next_tick_max = y_data_real_copy[day, tick, 1]

                # Check if there's no gap
                if next_tick_min <= prev_tick_min <= next_tick_max or next_tick_min <= prev_tick_max <= next_tick_max:
                    continue

                # Gap filling logic would go here if needed in the future
                # Current implementation just identifies gaps without filling them

        # Only assign back if changes were made
        # self.y_data_real = y_data_real_copy

    def generate_win_graph(
        self,
        max_pred: NDArray[np.float64],
        min_pred: NDArray[np.float64],
        buy_order_pred: NDArray[np.bool_],
    ) -> tuple[bool, bool]:
        """Generate a win graph from prediction data.

        Args:
            max_pred: Maximum price predictions.
            min_pred: Minimum price predictions.
            buy_order_pred: Buy order predictions (True for buy, False for sell).

        Returns:
            Tuple of (is_model_worth_saving, is_model_worth_double_saving)
        """
        logger.info(f"Generating win graph for {self.model_file_name}")

        # Create WinGraph instance
        with np.errstate(divide="ignore", invalid="ignore"):  # Handle division warnings
            win_graph = WinGraph(
                max_pred=max_pred,
                min_pred=min_pred,
                order_type_buy=buy_order_pred,
                y_real=self.y_data_real,
            )

        # Copy relevant attributes from WinGraph to this instance
        self._copy_attributes_from_win_graph(win_graph)

        # Log results
        self._log_evaluation_results()

        return self.is_model_worth_saving, self.is_model_worth_double_saving

    def _copy_attributes_from_win_graph(self, win_graph: WinGraph) -> None:
        """Copy relevant attributes from WinGraph to this instance.

        Args:
            win_graph: The WinGraph instance to copy attributes from.
        """
        attributes_to_copy = [
            "is_model_worth_saving",
            "is_model_worth_double_saving",
            "win_250_days",
            "win_pred_capture_percent",
            "simulation_250_days",
            "all_simulations_max_250_days",
        ]

        for attr in attributes_to_copy:
            setattr(self, attr, getattr(win_graph, attr))

    def _log_evaluation_results(self) -> None:
        """Log the evaluation results."""
        logger.info(f"File name: {self.model_file_name}")
        logger.info(f"Win 250 days: {self.win_250_days}%")
        logger.info(f"Win prediction capture: {self.win_pred_capture_percent}%")
        logger.info(f"Simulation 250 days: {self.simulation_250_days}%")
        logger.info(f"All simulations max 250 days: {self.all_simulations_max_250_days}%")

        if self.is_model_worth_double_saving:
            logger.info("Model is worth double saving \033[92m++++\033[0m")
        elif self.is_model_worth_saving:
            logger.info("Model is worth saving \033[92m+++\033[0m")

    def get_evaluation_metrics(self) -> dict[str, Union[float, bool]]:
        """Get all evaluation metrics as a dictionary.

        Returns:
            Dictionary containing all evaluation metrics.
        """
        return {
            "is_model_worth_saving": self.is_model_worth_saving,
            "is_model_worth_double_saving": self.is_model_worth_double_saving,
            "win_250_days": self.win_250_days,
            "win_pred_capture_percent": self.win_pred_capture_percent,
            "simulation_250_days": self.simulation_250_days,
            "all_simulations_max_250_days": self.all_simulations_max_250_days,
        }

    def validate_real_data(self) -> bool:
        """Validate that all real data values meet expected constraints.

        Returns:
            True if all data is valid, False otherwise.
        """
        logger.info("Validating real data...")

        valid = True

        # Check that high >= low for all data points
        high_vs_low = np.all(self.y_data_real[:, :, 1] >= self.y_data_real[:, :, 2])
        if not high_vs_low:
            logger.error("Validation failed: high values should be >= low values")
            valid = False

        # Check that open and close are between high and low
        open_valid = np.all(
            (self.y_data_real[:, :, 0] <= self.y_data_real[:, :, 1])
            & (self.y_data_real[:, :, 0] >= self.y_data_real[:, :, 2]),
        )
        if not open_valid:
            logger.error("Validation failed: open values should be between high and low")
            valid = False

        close_valid = np.all(
            (self.y_data_real[:, :, 3] <= self.y_data_real[:, :, 1])
            & (self.y_data_real[:, :, 3] >= self.y_data_real[:, :, 2]),
        )
        if not close_valid:
            logger.error("Validation failed: close values should be between high and low")
            valid = False

        return valid


# task - check for all real data for tickers
# that all open and close are inside min and max
# and that max is greater than min
