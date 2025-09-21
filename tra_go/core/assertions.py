import logging

from core.config import settings

logger = logging.getLogger(__name__)


def assert_env_vals() -> None:
    """Asserts the values of environment variables."""
    # Safety Factor validation
    assert (
        settings.SAFETY_FACTOR >= 1
    ), f"Safety Factor should be greater than or equal to 1, got {settings.SAFETY_FACTOR}"

    # Test size validation
    assert 0 < settings.TEST_SIZE <= 0.5, f"Test Size should be between 0 and 0.5, got {settings.TEST_SIZE}"

    # Learning rate validation
    assert 0 < settings.LEARNING_RATE <= 1, f"Learning rate should be between 0 and 1, got {settings.LEARNING_RATE}"

    # Batch size validation
    assert settings.BATCH_SIZE > 0, f"Batch size should be positive, got {settings.BATCH_SIZE}"

    # Number of epochs validation
    assert settings.NUMBER_OF_EPOCHS > 0, f"Number of epochs should be positive, got {settings.NUMBER_OF_EPOCHS}"

    # Number of neurons validation
    assert settings.NUMBER_OF_NEURONS > 0, f"Number of neurons should be positive, got {settings.NUMBER_OF_NEURONS}"

    # Number of layers validation
    assert settings.NUMBER_OF_LAYERS > 0, f"Number of layers should be positive, got {settings.NUMBER_OF_LAYERS}"

    # Dropout validation
    assert 0 <= settings.INITIAL_DROPOUT <= 1, f"Dropout should be between 0 and 1, got {settings.INITIAL_DROPOUT}"


def print_settings() -> None:
    """Prints the values of all environment variables."""
    print("\n" + "=" * 80)
    print("CURRENT CONFIGURATION SETTINGS")
    print("=" * 80)

    special_settings = {"ZERODHA_ID", "NUMBER_OF_EPOCHS", "NUMBER_OF_NEURONS", "RISK_TO_REWARD_RATIO", "DEBUG"}

    # Get all settings as dict
    settings_dict = settings.model_dump()

    for key, value in settings_dict.items():
        if key in special_settings:
            print("")
        # Mask sensitive information
        display_value = "***REDACTED***" if key in {"PASSWORD", "API_KEY", "ACCESS_TOKEN"} else value
        print(f"{key:<50} {display_value}")

    print("=" * 80)


def main():
    """Asserts and prints the settings."""
    try:
        assert_env_vals()
        print_settings()
    except AssertionError as e:
        logger.error(e)


if __name__ == "__main__":
    main()
