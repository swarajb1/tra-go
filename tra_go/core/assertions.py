import os

SAFETY_FACTOR: float = float(os.getenv("SAFETY_FACTOR"))
TEST_SIZE: float = float(os.getenv("TEST_SIZE"))

NUMBER_OF_NEURONS: int = int(os.getenv("NUMBER_OF_NEURONS"))
NUMBER_OF_LAYERS: int = int(os.getenv("NUMBER_OF_LAYERS"))
INITIAL_DROPOUT_PERCENT: int = int(os.getenv("INITIAL_DROPOUT_PERCENT"))

NUMBER_OF_EPOCHS: int = int(os.getenv("NUMBER_OF_EPOCHS"))
BATCH_SIZE: int = int(os.getenv("BATCH_SIZE"))
LEARNING_RATE: float = float(os.getenv("LEARNING_RATE"))


def assert_env_vals() -> None:
    assert SAFETY_FACTOR >= 1, f"Safety Factor should be greater than or equal to 1 == {SAFETY_FACTOR}"

    assert TEST_SIZE <= 0.5, f"Test Size should be less than or equal to o.5 == {TEST_SIZE}"

    print("\nSAFETY_FACTOR:", " " * (50 - len("SAFETY_FACTOR")), f"{SAFETY_FACTOR}")
    print("TEST_SIZE:", " " * (50 - len("TEST_SIZE")), f"{TEST_SIZE}")

    print("\nNUMBER_OF_NEURONS:", " " * (50 - len("NUMBER_OF_NEURONS")), f"{NUMBER_OF_NEURONS}")
    print("NUMBER_OF_LAYERS:", " " * (50 - len("NUMBER_OF_LAYERS")), f"{NUMBER_OF_LAYERS}")
    print("INITIAL_DROPOUT_PERCENT:", " " * (50 - len("INITIAL_DROPOUT_PERCENT")), f"{INITIAL_DROPOUT_PERCENT}")

    print("\nNUMBER_OF_EPOCHS:", " " * (50 - len("NUMBER_OF_EPOCHS")), f"{NUMBER_OF_EPOCHS}")
    print("BATCH_SIZE:", " " * (50 - len("BATCH_SIZE")), f"{BATCH_SIZE}")
    print("LEARNING_RATE:", " " * (50 - len("LEARNING_RATE")), f"{LEARNING_RATE}")

    return
