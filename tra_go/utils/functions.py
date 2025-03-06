import sys


def min_max_scaler(val: float, min_val: float, max_val: float) -> float:
    if max_val == min_val and max_val == 0:
        return 0

    if max_val == min_val:
        return 1

    scaled_data = (val - min_val) / (max_val - min_val)

    return scaled_data


def round_num_str(val, number_of_decimals) -> str:
    return "{:.{}f}".format(val, number_of_decimals)


def with_leverage(val: float) -> float:
    per_day_percent: float = (pow(1 + val / 100, 1 / 250) - 1) * 100

    return round((pow(per_day_percent * 5 / 100 + 1, 250) - 1) * 100)


def from_per_day_to_year(val: float) -> tuple[float, float]:
    print("\n(250_days, 250_days_5)")

    return (
        round((pow(val / 100 + 1, 250) - 1) * 100),
        round((pow(val * 5 / 100 + 1, 250) - 1) * 100),
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "with_leverage":
            print("\n", with_leverage(float(sys.argv[2])))

        elif sys.argv[1] == "from_per_day_to_year":
            print("\n", from_per_day_to_year(float(sys.argv[2])), sep="")

        else:
            print("\n", "Invalid 1st argument")
