# TRA-GO

Machine learning trading system for NSE/Indian markets. Trains neural networks to predict price bands and evaluates them via a tick-by-tick backtesting simulator.

## Quick start

1. Create a virtual environment and install dependencies

    ```bash
    make setup
    ```

2. Prepare data folders (idempotent)

    ```bash
    make create-data-folders
    ```

3. Populate `.env` from template and edit values as needed

    ```bash
    cp .env.template .env
    # edit .env with your values
    ```

4. Run a training session or evaluate models

    ```bash
    # Train models
    make train

    # Evaluate newly trained models
    make eval-new

    # Evaluate saved models (promote or discard with MOVE=true)
    make eval-saved NUM=6 MOVE=false
    ```

5. Launch TensorBoard

    ```bash
    make tensorboard
    ```

## Notable settings

Edit `.env` to control behavior. Key variables:

- NUMBER_OF_EPOCHS, BATCH_SIZE, LEARNING_RATE, TEST_SIZE
- NUMBER_OF_NEURONS, NUMBER_OF_LAYERS, INITIAL_DROPOUT_PERCENT
- USE_OPTIMIZED_DATA_LOADER=true to enable tf.data pipelines
- RISK_TO_REWARD_RATIO, SAFETY_FACTOR

On macOS, TensorFlow uses `tensorflow-macos` and `tensorflow-metal` as defined in `pyproject.toml`.

## Directory highlights

- `tra_go/core` — config, evaluation, simulation, logging
- `tra_go/band_*` — model architectures and metrics per band
- `tra_go/data_loader*.py` — data split, tf.data loader
- `training/` — logs, models, graphs, and tiered model folders

See `docs/` for deeper analysis and environment configuration notes.
