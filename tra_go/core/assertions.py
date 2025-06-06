from typing import Any

from core.config import settings


def assert_env_vals() -> None:
    """Asserts the values of environment variables.

    Raises:
        AssertionError: If the safety factor is less than 1 or the test size is greater than 0.5.

    Prints the values of all environment variables.

    Returns:
        None
    """

    assert (
        settings.SAFETY_FACTOR >= 1
    ), f"Safety Factor should be greater than or equal to 1 == {settings.SAFETY_FACTOR}"

    assert settings.TEST_SIZE <= 0.5, f"Test Size should be less than or equal to 0.5 == {settings.TEST_SIZE}"

    print("\n")

    # Type annotation: settings.dict().items() returns List[Tuple[str, Any]]
    items: list[tuple[str, Any]] = list(settings.model_dump().items())
    items.sort(key=lambda x: x[0])

    for item_name, item_value in items:
        if item_name in ["ZERODHA_ID", "NUMBER_OF_EPOCHS", "NUMBER_OF_NEURONS", "RISK_TO_REWARD_RATIO", "DEBUG"]:
            print("")

        padding: int = max(1, 50 - len(item_name))
        print(f"{item_name}:", " " * padding, f"{item_value}")

    return
