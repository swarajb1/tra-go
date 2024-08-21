from core.config import settings


def assert_env_vals() -> None:
    assert (
        settings.SAFETY_FACTOR >= 1
    ), f"Safety Factor should be greater than or equal to 1 == {settings.SAFETY_FACTOR}"

    assert settings.TEST_SIZE <= 0.5, f"Test Size should be less than or equal to 0.5 == {settings.TEST_SIZE}"

    print("\n")
    for item in settings:
        if item[0] in ["NUMBER_OF_EPOCHS", "NUMBER_OF_NEURONS", "RISK_TO_REWARD_RATIO", "DEBUG"]:
            print("")

        print(f"{item[0]}:", " " * (50 - len(item[0])), f"{item[1]}")

    del item

    return
