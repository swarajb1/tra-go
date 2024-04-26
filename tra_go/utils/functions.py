def min_max_scaler(val: float, min_val: float, max_val: float) -> float:
    if max_val == min_val and max_val == 0:
        return 0

    if max_val == min_val:
        return 1

    scaled_data = (val - min_val) / (max_val - min_val)

    return scaled_data
