def get_initial_index_offset(points_in_zone_2nd: int) -> int:
    if points_in_zone_2nd == 132:
        return 47
    elif points_in_zone_2nd == 150:
        return 36
    else:
        raise ValueError("Unsupported number of points in zone 2nd")
