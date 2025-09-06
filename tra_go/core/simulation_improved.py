"""
Improved trading simulation module with better structure and performance.
"""

from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

import numpy as np
from core.config import settings
from core.logger import log_performance_metric, log_warning
from numpy.typing import NDArray
from scipy import stats
from scipy.stats import kurtosis, skew
from utils.functions import round_num_str

from database.enums import ProcessedDataType


# Configuration constants
class SimulationConfig:
    PERCENT_250_DAYS_MIN_THRESHOLD: int = -100
    PERCENT_250_DAYS_WORTH_SAVING: int = 5
    MIN_REWARD_THRESHOLD_PERCENT: float = 0.05
    SPECIAL_TRADE_THRESHOLD: float = 70.0  # % trades taken
    SPECIAL_EXPECTED_THRESHOLD: float = 50.0  # % expected trades


class TradeType(Enum):
    BUY = "buy"
    SELL = "sell"


class TradeResult(NamedTuple):
    """Result of a single trade simulation."""

    trade_taken: bool
    trade_completed: bool
    stop_loss_hit: bool
    completed_at_closing: bool
    is_expected_trade: bool
    net_reward: float
    closing_reward: float


@dataclass
class SimulationMetrics:
    """Container for simulation metrics."""

    wins_day_wise: NDArray[np.float64]
    invested_day_wise: NDArray[np.float64]
    expected_reward_day_wise: NDArray[np.float64]
    trade_stats: dict


class Simulation:
    """
    Improved trading simulation with better structure and performance.
    """

    def __init__(
        self,
        buy_price_arr: NDArray[np.float64],
        sell_price_arr: NDArray[np.float64],
        order_type_buy_arr: NDArray[np.bool_],
        real_price_arr: NDArray[np.float64],
        print_log_stats_extra: bool = False,
    ):
        self.buy_price_arr = buy_price_arr
        self.sell_price_arr = sell_price_arr
        self.order_type_buy_arr = order_type_buy_arr
        self.real_price_arr = real_price_arr  # shape: (days, ticks, OHLC)
        self.print_log_stats_extra = print_log_stats_extra

        # Initialize results
        self.is_model_worth_saving = False
        self.is_model_worth_double_saving = False
        self.simulation_250_days = 0.0
        self.all_simulations_max_250_days = float("-inf")

        # Analysis data
        self.real_data_for_analysis = np.array([])
        self.stoploss_data_for_analysis = np.array([])
        self.stoploss_rrr_for_analysis = 1.0

        self.real_mean = 0.0
        self.expected_mean = 0.0
        self.actual_full_reward_mean = 0.0

        # Run simulation
        self._run_simulation()
        self._set_real_full_reward_mean()
        self._display_stats()

    def _create_rrr_list(self) -> list[float]:
        """Create list of risk-to-reward ratios to test."""
        rrr_list = [0, 0.33, 0.66, 1, 2, 3, 5, 8, 15]

        if settings.RISK_TO_REWARD_RATIO not in rrr_list:
            rrr_list.append(settings.RISK_TO_REWARD_RATIO)

        return sorted(rrr_list)

    def _calculate_stop_loss(self, buy_price: float, sell_price: float, is_buy_trade: bool, rrr: float) -> float:
        """Calculate stop loss price based on trade type and RRR."""
        expected_reward = sell_price - buy_price

        if is_buy_trade:
            return buy_price - (expected_reward * rrr)
        else:
            return sell_price + (expected_reward * rrr)

    def _simulate_single_day(self, day_idx: int, rrr: float) -> TradeResult:
        """
        Simulate trading for a single day.

        Args:
            day_idx: Day index
            rrr: Risk to reward ratio

        Returns:
            TradeResult with all relevant metrics
        """
        is_buy_trade = self.order_type_buy_arr[day_idx]
        buy_price = self.buy_price_arr[day_idx]
        sell_price = self.sell_price_arr[day_idx]
        expected_reward = sell_price - buy_price

        # Skip trades with minimal reward
        invested_amount = (buy_price + sell_price) / 2
        if expected_reward / invested_amount * 100 < SimulationConfig.MIN_REWARD_THRESHOLD_PERCENT:
            # reward is less than 0.05% of the invested amount
            # so, not worth trading
            return TradeResult(False, False, False, False, False, 0.0, 0.0)

        stop_loss = self._calculate_stop_loss(buy_price, sell_price, is_buy_trade, rrr)

        # Track trade states
        trade_taken = False
        trade_completed = False
        stop_loss_hit = False
        net_reward = 0.0

        # Simulate each tick
        for tick_idx in range(self.real_price_arr.shape[1]):
            tick_low = self.real_price_arr[day_idx, tick_idx, 2]
            tick_high = self.real_price_arr[day_idx, tick_idx, 1]

            if is_buy_trade:
                trade_taken, trade_completed, stop_loss_hit, net_reward = self._process_buy_trade(
                    buy_price,
                    sell_price,
                    stop_loss,
                    tick_low,
                    tick_high,
                    trade_taken,
                    trade_completed,
                )
            else:
                trade_taken, trade_completed, stop_loss_hit, net_reward = self._process_sell_trade(
                    buy_price,
                    sell_price,
                    stop_loss,
                    tick_low,
                    tick_high,
                    trade_taken,
                    trade_completed,
                )

            if trade_completed:
                break

        # Handle trades still open at closing
        closing_reward = 0.0
        completed_at_closing = False

        if trade_taken and not trade_completed:
            completed_at_closing = True
            last_tick_mid = (self.real_price_arr[day_idx, -1, 2] + self.real_price_arr[day_idx, -1, 1]) / 2

            if is_buy_trade:
                net_reward = last_tick_mid - buy_price
            else:
                net_reward = sell_price - last_tick_mid

            closing_reward = net_reward

        # Determine if this was an expected trade
        is_expected_trade = trade_completed and not stop_loss_hit

        return TradeResult(
            trade_taken=trade_taken,
            trade_completed=trade_completed or completed_at_closing,
            stop_loss_hit=stop_loss_hit,
            completed_at_closing=completed_at_closing,
            is_expected_trade=is_expected_trade,
            net_reward=net_reward,
            closing_reward=closing_reward,
        )

    def _process_buy_trade(
        self,
        buy_price: float,
        sell_price: float,
        stop_loss: float,
        tick_low: float,
        tick_high: float,
        trade_taken: bool,
        trade_completed: bool,
    ) -> tuple[bool, bool, bool, float]:
        """Process a buy trade for a single tick."""
        net_reward = 0.0
        stop_loss_hit = False

        # Check if we can enter the trade
        if not trade_taken and tick_low <= buy_price <= tick_high:
            trade_taken = True

        # Check if we can exit the trade
        if trade_taken and not trade_completed:
            if tick_low <= sell_price <= tick_high:
                trade_completed = True
                net_reward = sell_price - buy_price
            elif tick_low <= stop_loss <= tick_high:
                trade_completed = True
                stop_loss_hit = True
                net_reward = stop_loss - buy_price

        return trade_taken, trade_completed, stop_loss_hit, net_reward

    def _process_sell_trade(
        self,
        buy_price: float,
        sell_price: float,
        stop_loss: float,
        tick_low: float,
        tick_high: float,
        trade_taken: bool,
        trade_completed: bool,
    ) -> tuple[bool, bool, bool, float]:
        """Process a sell trade for a single tick."""
        net_reward = 0.0
        stop_loss_hit = False

        # Check if we can enter the trade
        if not trade_taken and tick_low <= sell_price <= tick_high:
            trade_taken = True

        # Check if we can exit the trade
        if trade_taken and not trade_completed:
            if tick_low <= buy_price <= tick_high:
                trade_completed = True
                net_reward = sell_price - buy_price
            elif tick_low <= stop_loss <= tick_high:
                trade_completed = True
                stop_loss_hit = True
                net_reward = sell_price - stop_loss

        return trade_taken, trade_completed, stop_loss_hit, net_reward

    def _run_simulation(self) -> None:
        """Run the complete simulation across all RRR values."""
        print("Simulation started...")

        number_of_days = self.real_price_arr.shape[0]
        rrr_list = self._create_rrr_list()

        for rrr in rrr_list:
            metrics = self._simulate_rrr(rrr, number_of_days)
            self._analyze_rrr_results(rrr, metrics, number_of_days)

    def _simulate_rrr(self, rrr: float, number_of_days: int) -> SimulationMetrics:
        """Simulate trading for a specific RRR value."""
        # Initialize tracking arrays
        wins_day_wise = np.zeros(number_of_days)
        invested_day_wise = np.zeros(number_of_days)
        expected_reward_day_wise = np.zeros(number_of_days)

        # Track trade statistics
        trade_stats = {
            "trades_taken": 0,
            "trades_completed": 0,
            "stop_losses_hit": 0,
            "completed_at_closing": 0,
            "expected_trades": 0,
            "win_trades": 0,
            "closing_rewards": [],
        }

        # Simulate each day
        for day_idx in range(number_of_days):
            result = self._simulate_single_day(day_idx, rrr)

            # Update arrays
            wins_day_wise[day_idx] = result.net_reward

            if result.trade_taken:
                invested_day_wise[day_idx] = (
                    self.buy_price_arr[day_idx] if self.order_type_buy_arr[day_idx] else self.sell_price_arr[day_idx]
                )
                expected_reward = self.sell_price_arr[day_idx] - self.buy_price_arr[day_idx]
                expected_reward_day_wise[day_idx] = expected_reward / invested_day_wise[day_idx] * 100

            # Update statistics
            if result.trade_taken:
                trade_stats["trades_taken"] += 1
            if result.trade_completed:
                trade_stats["trades_completed"] += 1
            if result.stop_loss_hit:
                trade_stats["stop_losses_hit"] += 1
            if result.completed_at_closing:
                trade_stats["completed_at_closing"] += 1
                trade_stats["closing_rewards"].append(result.closing_reward)
            if result.is_expected_trade:
                trade_stats["expected_trades"] += 1
            if result.net_reward > 0:
                trade_stats["win_trades"] += 1

        return SimulationMetrics(
            wins_day_wise=wins_day_wise,
            invested_day_wise=invested_day_wise,
            expected_reward_day_wise=expected_reward_day_wise,
            trade_stats=trade_stats,
        )

    def _analyze_rrr_results(self, rrr: float, metrics: SimulationMetrics, number_of_days: int) -> None:
        """Analyze and log results for a specific RRR value."""
        # Calculate performance metrics
        safe_invested = np.where(metrics.invested_day_wise == 0, 1, metrics.invested_day_wise)
        returns_percent = (metrics.wins_day_wise / safe_invested) * 100
        avg_return = np.mean(returns_percent / 100)

        # Calculate 250-day performance
        days_250_performance = (pow(1 + avg_return, 250) - 1) * 100

        # Format output
        percent_str = f"{days_250_performance:.2f}%"
        display_str = (
            percent_str if days_250_performance > SimulationConfig.PERCENT_250_DAYS_MIN_THRESHOLD else "   --"
        )

        # Log performance
        print(f"\t\tRisk To Reward Ratio: {rrr:.2f}\t250 Days (avg): {display_str}", end="")

        if days_250_performance > SimulationConfig.PERCENT_250_DAYS_WORTH_SAVING:
            print(" \033[92m++\033[0m")
            self.is_model_worth_saving = True

            if rrr <= settings.RISK_TO_REWARD_RATIO:
                self.is_model_worth_double_saving = True
        else:
            print()

        # Track maximum performance
        self.all_simulations_max_250_days = max(self.all_simulations_max_250_days, float(days_250_performance))

        # Store data for the configured RRR
        if rrr == settings.RISK_TO_REWARD_RATIO:
            self._store_analysis_data(
                metrics,
                returns_percent,
                float(days_250_performance),
                number_of_days,
            )

            """# graph making

            # plt.figure(figsize=(16, 9))

            # x = np.arange(0, number_of_days, 1)
            # plt.scatter(x, arr_real, color="orange", s=50)

            # plt.plot(arr_real)

            # arr2 = np.array(stop_loss_hit_list) * (-0.2)
            # plt.plot(arr2)

            # filename = f"training/graphs/{self.y_type} - {self.now_datetime} - Splot - sf={self.SAFETY_FACTOR} - model_{self.model_num}.png"
            # if self.test_size == 0:
            #     filename = filename[:-4] + "- valid.png"

            # plt.savefig(filename, dpi=300, bbox_inches="tight")
            """

    def _store_analysis_data(
        self,
        metrics: SimulationMetrics,
        returns_percent: NDArray,
        days_250_performance: float,
        number_of_days: int,
    ) -> None:
        """Store analysis data for the configured RRR value."""
        self.real_data_for_analysis = returns_percent
        self.stoploss_data_for_analysis = metrics.expected_reward_day_wise.copy()
        self.stoploss_rrr_for_analysis = 1.0
        self.simulation_250_days = round(days_250_performance, 2)

        # Log detailed statistics
        self._log_detailed_trade_stats(metrics, number_of_days)

    def _log_detailed_trade_stats(self, metrics: SimulationMetrics, number_of_days: int) -> None:
        """Log detailed trading statistics."""
        stats = metrics.trade_stats

        if stats["trades_taken"] == 0:
            return

        # Safeguard for division by zero (matching original logic)
        safe_trades_taken = max(stats["trades_taken"], 1)

        # Calculate percentages
        pct_trades_taken = stats["trades_taken"] / number_of_days * 100
        pct_trades_completed = stats["trades_completed"] / safe_trades_taken * 100
        pct_stop_losses = stats["stop_losses_hit"] / safe_trades_taken * 100
        pct_closing = stats["completed_at_closing"] / safe_trades_taken * 100
        pct_expected = stats["expected_trades"] / safe_trades_taken * 100
        pct_wins = stats["win_trades"] / safe_trades_taken * 100

        # Calculate closing trades contribution (missing from improved version)
        safe_invested = np.where(metrics.invested_day_wise == 0, 1, metrics.invested_day_wise)
        closing_rewards_array = np.array(stats["closing_rewards"]) if stats["closing_rewards"] else np.array([])

        # Create a proper array for closing contribution calculation
        completed_at_closing_reward = np.zeros(number_of_days, dtype=np.float32)
        closing_idx = 0
        for day_idx in range(number_of_days):
            if closing_idx < len(closing_rewards_array):
                completed_at_closing_reward[day_idx] = closing_rewards_array[closing_idx]
                closing_idx += 1

        closing_arr_percent_avg_win_per_day = np.mean((completed_at_closing_reward / safe_invested) * 100)

        # Special condition for high-performance models
        if (
            pct_trades_taken > SimulationConfig.SPECIAL_TRADE_THRESHOLD
            and pct_expected > SimulationConfig.SPECIAL_EXPECTED_THRESHOLD
        ):
            self.is_model_worth_saving = True
            log_performance_metric(
                "special_condition_triggered",
                1.0,
                trades_taken=pct_trades_taken,
                expected=pct_expected,
            )

        # Log all statistics with exact formatting from original
        print(f"\n\t\t\t Percent Trade Taken\t\t\t{pct_trades_taken:.2f} %")
        print(
            f"\t\t\t Percent Trade Taken And Out\t\t{(stats['trades_completed'] / number_of_days * 100):.2f} % \t | \t"
            f"{pct_trades_completed:.2f} %",
        )
        print(
            f"\t\t\t Percent Stop Loss Hit\t\t\t{(stats['stop_losses_hit'] / number_of_days * 100):.2f} % \t | \t"
            f"{pct_stop_losses:.2f} %",
        )
        print(
            f"\t\t\t Percent Completed At Closing\t\t{(stats['completed_at_closing'] / number_of_days * 100):.2f} % \t | \t"
            f"{pct_closing:.2f} %",
        )
        print(f"\t\t\t Closing Trades Per Day contribution\t{closing_arr_percent_avg_win_per_day:.2f}")
        print(
            f"\t\t\t Percent Expected Trades\t\t{(stats['expected_trades'] / number_of_days * 100):.2f} % \t | \t"
            f"{pct_expected:.2f} %",
        )
        print(
            f"\n\t\t\t Percent Win Trades\t\t\t{(stats['win_trades'] / number_of_days * 100):.2f} % \t | \t"
            f"{pct_wins:.2f} %\n",
        )

    def _set_real_full_reward_mean(self) -> None:
        """Calculate the mean of maximum possible rewards."""
        number_of_days = self.real_price_arr.shape[0]
        full_rewards = np.zeros(number_of_days)

        # Track real order type based on timing (missing from improved version)
        real_order_type_buy = np.zeros(number_of_days, dtype=bool)

        for day_idx in range(number_of_days):
            day_lows = self.real_price_arr[day_idx, :, 2]
            day_highs = self.real_price_arr[day_idx, :, 1]

            min_price = np.min(day_lows)
            max_price = np.max(day_highs)

            full_reward = max_price - min_price

            # buy trend is max comes after then min (original logic)
            real_order_type_buy[day_idx] = np.argmax(self.real_price_arr[day_idx, :, 1]) > np.argmax(
                self.real_price_arr[day_idx, :, 0],
            )

            invested_amount = (
                self.buy_price_arr[day_idx] if real_order_type_buy[day_idx] else self.sell_price_arr[day_idx]
            )

            full_rewards[day_idx] = full_reward / invested_amount * 100

        self.actual_full_reward_mean = float(np.mean(full_rewards))

    def _display_stats(self) -> None:
        """Display comprehensive statistics."""
        print(f"\n\n\n{'-' * 30}")
        print(f"Real End of Day Stats (per day %), RRR = {settings.RISK_TO_REWARD_RATIO}\n")
        self._log_statistics(self.real_data_for_analysis, ProcessedDataType.REAL)

        print(f"\n\n\n{'-' * 30}")
        print(f"Stop Loss Data Stats (per day %), RRR = {self.stoploss_rrr_for_analysis}\n")
        self._log_statistics(self.stoploss_data_for_analysis, ProcessedDataType.EXPECTED_REWARD)

        if self.actual_full_reward_mean > 0:
            capture_percent = self.real_mean / self.actual_full_reward_mean * 100
            print(f"\n\nCapture Return Percent:\t\t{capture_percent:.2f} %")

    def _log_statistics(self, arr: NDArray, data_type: ProcessedDataType) -> None:
        """Log statistical analysis of the data."""
        if len(arr) == 0:
            log_warning("Empty array provided for statistical analysis")
            return

        sorted_arr = np.sort(arr)
        mean_val = np.mean(sorted_arr)

        if data_type == ProcessedDataType.REAL:
            self.real_mean = float(mean_val)
        elif data_type == ProcessedDataType.EXPECTED_REWARD:
            self.expected_mean = float(mean_val)

            # Special condition: if expected mean is too high, model might be skewed
            if self.expected_mean > 3 or np.isnan(self.expected_mean):
                self.is_model_worth_saving = False
                self.is_model_worth_double_saving = False
                log_warning("Model rejected due to high expected mean", expected_mean=self.expected_mean)

        median_val = np.median(sorted_arr)
        min_val = np.min(sorted_arr)
        max_val = np.max(sorted_arr)

        print(f"Mean: \t\t\t\t{round_num_str(mean_val, 2)}")
        print(f"Median: \t\t\t{round_num_str(median_val, 2)}")
        print(f"Min: \t\t\t\t{round_num_str(min_val, 2)}")
        print(f"Max: \t\t\t\t{round_num_str(max_val, 2)}")

        if self.print_log_stats_extra:
            self._log_statistics_extra(sorted_arr)

    def _log_statistics_extra(self, sorted_arr: NDArray) -> None:
        """Log additional statistical measures."""
        try:
            # Statistical measures
            z_scores = stats.zscore(sorted_arr)
            stats.rankdata(sorted_arr)
            kurtosis_val = kurtosis(sorted_arr)
            skew_val = skew(sorted_arr)

            # Z-scores display (matching original format)
            z_scores_array = np.array(z_scores)
            if len(z_scores_array) > 6:
                z_scores_small = np.concatenate([z_scores_array[:3], z_scores_array[-3:]])
                z_list = [round_num_str(x, 3) for x in z_scores_small]
                z_list.insert(3, " ... ")
                print("Z-Scores: \t\t\t", z_list)
            else:
                print("Z-Scores: \t\t\t", [round_num_str(x, 3) for x in z_scores_array])

            print(f"Kurtosis: \t\t\t{round_num_str(kurtosis_val, 2)}")
            print(f"Skewness: \t\t\t{round_num_str(skew_val, 2)}")

            # Normality tests
            if len(sorted_arr) >= 3:  # Minimum required for Shapiro-Wilk
                shapiro_test = stats.shapiro(sorted_arr)
                print(f"Shapiro-Wilk Test: \t\t{shapiro_test}")

                # Add Kolmogorov-Smirnov test (missing from improved version)
                kolmogorov_smirnov_test = stats.kstest(sorted_arr, "norm")
                print(f"Kolmogorov-Smirnov Test: \t{kolmogorov_smirnov_test}")

            # Outlier detection (matching original format)
            outliers = sorted_arr[np.abs(z_scores_array) > 3]
            anomalies = sorted_arr[np.abs(z_scores_array) > 2]

            outlier_pct = np.size(outliers) / np.size(sorted_arr) * 100
            anomaly_pct = np.size(anomalies) / np.size(sorted_arr) * 100

            print(f"Anomalies: \t\t\t{round_num_str(anomaly_pct, 2)} %")
            print(f"Outliers: \t\t\t{round_num_str(outlier_pct, 2)} %")

        except Exception as e:
            log_warning("Failed to compute extended statistics", error=str(e))

    def get_model_worthiness(self) -> tuple[bool, bool]:
        """
        Returns a tuple indicating the worthiness of the model.

        Returns:
            Tuple of (is_worth_saving, is_worth_double_saving)
        """
        return self.is_model_worth_saving, self.is_model_worth_double_saving
