import os

from tabulate import tabulate

from database.enums import ModelLocationType


def parse_model_file_name(file_name: str) -> tuple[str, str] | None:
    """
    Parse the model file name to extract x_type and y_type strings.

    Expected format: "something - interval - x_type - y_type - ticker.keras"
    """
    if not file_name.endswith(".keras"):
        return None

    # Remove .keras extension
    file_name_1 = file_name[: -len(".keras")]

    parts = file_name_1.split(" - ")
    if len(parts) < 5:
        return None

    x_type_str = parts[2]
    y_type_str = parts[3]

    return x_type_str, y_type_str


def get_comprehensive_distribution():
    """
    Get comprehensive distribution including counts and types per folder.
    """
    results = {}

    for location_type in ModelLocationType:
        location_path = location_type.value
        location_name = location_type.name

        if not os.path.exists(location_path):
            results[location_name] = {"count": 0, "x_types": {}, "y_types": {}}
            continue

        try:
            files = [f for f in os.listdir(location_path) if f.endswith(".keras")]
            count = len(files)

            x_types = {}
            y_types = {}

            for file_name in files:
                parsed = parse_model_file_name(file_name)
                if parsed:
                    x_type, y_type = parsed
                    x_types[x_type] = x_types.get(x_type, 0) + 1
                    y_types[y_type] = y_types.get(y_type, 0) + 1

            results[location_name] = {"count": count, "x_types": x_types, "y_types": y_types}

        except OSError:
            results[location_name] = {"count": 0, "x_types": {}, "y_types": {}}

    return results


def format_types(types_dict):
    """Format types dictionary as string."""
    if not types_dict:
        return "-"
    return ", ".join([f"{k}:{v}" for k, v in sorted(types_dict.items(), key=lambda x: x[1], reverse=True)])


if __name__ == "__main__":
    print("COMPREHENSIVE MODEL DISTRIBUTION ANALYSIS")
    print("=" * 80)

    results = get_comprehensive_distribution()
    total_models = sum(data["count"] for data in results.values())

    print(f"\nTOTAL MODELS: {total_models}\n")

    # Prepare table data
    table_data = []
    headers = ["Folder", "Total", "%", "X_Type", "X_Count", "Y_Type", "Y_Count", "Description"]

    folder_descriptions = {
        "TRAINED_NEW": "Newly trained models",
        "SAVED": "Single-tier saved models",
        "SAVED_DOUBLE": "Double-tier saved models",
        "SAVED_TRIPLE": "Triple-tier saved models",
        "OLD": "Older models",
        "DISCARDED": "Discarded/poor performing models",
    }

    for location, data in results.items():
        count = data["count"]
        percentage = f"{(count / total_models * 100):.1f}%" if total_models > 0 else "0.0%"
        description = folder_descriptions.get(location, "Unknown")

        # Add total row
        table_data.append([f"{location} (Total)", count, percentage, "-", "-", "-", "-", description])

        # Add detailed rows for each X-Y type combination
        x_types = data["x_types"]
        y_types = data["y_types"]

        if x_types and y_types:
            # Try to match X and Y types by count (assuming they correspond)
            used_y_types = set()

            for x_type, x_count in sorted(x_types.items(), key=lambda x: x[1], reverse=True):
                # Find corresponding Y type with same count
                corresponding_y = None
                for y_type, y_count in y_types.items():
                    if y_count == x_count and y_type not in used_y_types:
                        corresponding_y = y_type
                        used_y_types.add(y_type)
                        break

                if corresponding_y:
                    table_data.append(
                        [f"{location} ({x_type})", "-", "-", x_type, x_count, corresponding_y, x_count, ""],
                    )
                else:
                    # If no direct match, show X type only
                    table_data.append([f"{location} ({x_type})", "-", "-", x_type, x_count, "-", "-", ""])

            # Show any remaining Y types that weren't matched
            for y_type, y_count in y_types.items():
                if y_type not in used_y_types:
                    table_data.append([f"{location} ({y_type})", "-", "-", "-", "-", y_type, y_count, ""])

        elif x_types:
            # Only X types
            for x_type, x_count in x_types.items():
                table_data.append([f"{location} ({x_type})", "-", "-", x_type, x_count, "-", "-", ""])
        elif y_types:
            # Only Y types
            for y_type, y_count in y_types.items():
                table_data.append([f"{location} ({y_type})", "-", "-", "-", "-", y_type, y_count, ""])

    # Print the table using tabulate
    print(
        tabulate(
            table_data,
            headers=headers,
            tablefmt="grid",
            colalign=("left", "right", "right", "left", "right", "left", "right", "left"),
        ),
    )

    # Global summary
    print("\n\nGLOBAL TYPE SUMMARY:")
    print("-" * 50)

    global_x = {}
    global_y = {}

    for data in results.values():
        for x_type, count in data["x_types"].items():
            global_x[x_type] = global_x.get(x_type, 0) + count
        for y_type, count in data["y_types"].items():
            global_y[y_type] = global_y.get(y_type, 0) + count

    print(f"\nX_TYPES (Total: {sum(global_x.values())}):")
    for x_type, count in sorted(global_x.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total_models * 100) if total_models > 0 else 0
        print(f"  {x_type}: {count} ({pct:.1f}%)")

    print(f"\nY_TYPES (Total: {sum(global_y.values())}):")
    for y_type, count in sorted(global_y.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total_models * 100) if total_models > 0 else 0
        print(f"  {y_type}: {count} ({pct:.1f}%)")

    print("\n" + "=" * 50)
