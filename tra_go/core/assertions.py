from core.config import settings
from core.logger import log_exceptions, logger


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


@log_exceptions()
def main():
    """Asserts and prints the settings."""
    logger.info("Validating configuration settings")
    print_settings()


if __name__ == "__main__":
    main()
