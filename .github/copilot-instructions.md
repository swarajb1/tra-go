# TRA-GO AI Coding Agent Instructions

## Project Overview

TRA-GO is a machine learning trading system that trains neural networks to predict stock price movements and simulates trading strategies. The system processes financial data from NSE/Indian markets using TensorFlow/Keras for model training and comprehensive backtesting simulations.

## Architecture & Key Components

### Core Data Flow

1. **Data Sources**: Raw market data from Zerodha API (`download_data_zerodha.py`) and Yahoo Finance (`download_data_yf.py`)
2. **Data Processing**: Raw → Cleaned (`data_clean.py`) → Training format (`data_loader.py`, `data_loader_tf.py`)
3. **Model Training**: Multi-band neural networks (`band_*/keras_model_*.py`) with custom metrics
4. **Simulation**: Trading strategy backtesting (`core/simulation_improved.py`) with risk-reward analysis
5. **Evaluation**: Model performance assessment (`core/evaluate_models.py`) with automated model saving

### Band-Based Architecture

- **Bands** represent different prediction strategies (e.g., `BAND_1_CLOSE`, `BAND_2_1`, `BAND_4`)
- Each band has its own model architecture in `tra_go/band_*/` directories
- Input/Output data points are configurable (150/150 points) via `training_zero.py` constants
- Models predict price ranges (min/max) rather than single values
- **Zone-based time processing**: `TOTAL_POINTS_IN_ONE_DAY = 375` with configurable zone splits

### Configuration System

- **Pydantic-based settings** in `core/config.py` with environment-specific configs
- **Required env vars**: `ZERODHA_ID`, `PASSWORD`, `API_KEY`, `ACCESS_TOKEN`, `RISK_TO_REWARD_RATIO`
- **Model params**: `NUMBER_OF_EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, `NUMBER_OF_NEURONS`
- **Data loader toggle**: `USE_OPTIMIZED_DATA_LOADER` switches between tf.data and pandas
- Use `.env.template` as reference for required environment variables

## Development Workflows

### Training Models

```bash
# Single ticker training (default: 4 CPU cores)
make run model=train

# Parallel training across multiple tickers
make run model=train_parallel

# Evaluate newly trained models
make run model=training_new
```

### Model Evaluation Pipeline

```bash
# Evaluate saved models (6 models by default, no file movement)
make run model=saved num=6 move=false

# Evaluate with file movement (promotes good models)
make run model=saved_double num=10 move=true

# Evaluate triple-tier models
make run model=saved_triple num=5

# Clean up discarded models
make run model=discarded
```

### Data Management

```bash
# Download new market data from Yahoo Finance
make new-yf-data

# Clean and process data
make get_clean_data

# Clean training logs
make clean_logs

# Setup project structure
make create_data_folders
```

## Critical Code Patterns

### Model Training Structure

- Main training loop in `main_training.py` with ticker list iteration
- **Optimized data loading**: Toggle `USE_OPTIMIZED_LOADER` for tf.data vs traditional pandas loading
- **Custom callbacks**: TensorBoard logging, model checkpointing, NaN termination
- **Memory management**: Explicit garbage collection between model training sessions
- **Multi-core support**: Default 4-core parallel training with `main_training_4_cores()`

### Simulation Engine

- `simulation_improved.py` implements sophisticated tick-by-tick trading simulation with comprehensive backtesting
- **Risk-reward ratios**: Tests multiple RRR values (0, 0.33, 0.66, 1, 2, 3, 5, 8, 15)
- **Performance thresholds**: Models saved if 250-day performance > 5% OR special conditions met
- **Trade execution**: Handles both BUY/SELL orders with stop-loss and closing scenarios
- **Statistical analysis**: Includes kurtosis, skew, and advanced performance metrics

#### Simulation Architecture

- **Tick-by-tick execution**: Processes OHLC data at minute-level granularity
- **Trade lifecycle management**: Entry conditions, stop-loss triggers, profit targets, and end-of-day closings
- **Real-time decision making**: Evaluates price movements within each trading tick to determine optimal entry/exit
- **Multi-RRR testing**: Automatically tests various risk-reward ratios to find optimal trading parameters

#### Trade Logic Patterns

- **Entry conditions**: Trades triggered when predicted buy/sell prices fall within tick's OHLC range
- **Exit strategies**: Three exit scenarios - profit target hit, stop-loss triggered, or end-of-day closing
- **Order types**: Supports both BUY (long) and SELL (short) positions with appropriate stop-loss calculations
- **Minimal reward filtering**: Skips trades with expected returns < 0.05% of invested amount

#### Performance Evaluation

- **250-day compound returns**: Primary metric for model worthiness (>5% threshold)
- **Trade statistics**: Tracks completion rates, stop-loss hits, expected vs actual trades
- **Capture percentage**: Measures how much of theoretical maximum profit is captured
- **Special conditions**: High-frequency models (>70% trades taken, >50% expected) get priority saving
- **Statistical robustness**: Uses scipy for kurtosis, skew analysis of return distributions

### Data Loading Patterns

- **Zone-based processing**: Market hours divided into time zones with configurable offsets
- **OHLC format**: [Open, High, Low, Close, Volume] + real_close for training data
- **Path conventions**: `data_training/`, `data_cleaned/`, `training/models*/`
- **Ticker management**: Predefined list in `main_training.py` with NSE stock symbols

### Custom Metrics

- **Multi-objective loss functions** combining RMSE, absolute error, and domain-specific metrics
- **Capture percentage**: Measures how much of the theoretical maximum profit is captured
- **Expected vs actual trade analysis** with detailed logging

## File Organization Conventions

```text
tra_go/
├── core/           # Core business logic (config, simulation, evaluation)
├── band_*/         # Band-specific model architectures
├── decorators/     # Common decorators (timing, etc.)
├── utils/          # Utility functions (scaling, time formatting)
├── *.py            # Main modules (training, data loading, etc.)
data_*/             # Data directories (training, cleaned, raw)
training/           # Model storage with hierarchical organization
└── models*/        # Different quality tiers (saved, saved_double, etc.)
```

## Integration Points

### External APIs

- **Zerodha KiteConnect**: Real-time trading data with authentication flow
- **Yahoo Finance**: Historical data fallback and validation
- **TensorBoard**: Training metrics visualization at `training/logs/`

### Model Lifecycle

1. **Training**: Models saved to `training/models/`
2. **Evaluation**: Performance-based promotion to `models_saved/`, `models_saved_double/`
3. **Archival**: Poor models moved to `models_z_old/` or `models_zz_discarded/`

## Development Setup

```bash
make create-venv      # Create virtual environment
make install          # Install dependencies via Poetry
make create_data_folders  # Initialize required directory structure
```

Environment setup requires careful attention to TensorFlow platform-specific dependencies (macOS vs Linux) as defined in `pyproject.toml`.

## General Instructions for Copilot

- Any docs file created should be in the `docs/` directory.
- Don't use MCP Server. Create a python file in `temp/` directory of the project and share the code block there. And execute that file in terminal using `.venv/bin/python` command. Delete/clear the file after use.
