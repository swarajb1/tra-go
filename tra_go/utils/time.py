def format_time(seconds: float):
    days = seconds // 86400
    hours = (seconds % 86400) // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    parts: list[str] = []

    if int(days) > 0:
        parts.append(f"{int(days)} days")
    if int(hours) > 0:
        parts.append(f"{int(hours)} hours")
    if int(minutes) > 0:
        parts.append(f"{int(minutes)} minutes")
    if round(secs, 2) > 0 or not parts:
        parts.append(f"{round(secs, 2)} seconds")

    return ", ".join(parts)
