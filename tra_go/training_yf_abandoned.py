def custom_evaluate_safety_factor_hl(
    model,
    X_test,
    Y_test,
    y_type,
    now_datetime,
    safety_factor,
):
    y_pred = model.predict(X_test)

    list_min_pred = []
    list_max_pred = []
    list_min_actual = []
    list_max_actual = []

    for i in range(y_pred.shape[0]):
        # i  -> day
        # for 1st day
        min_pred = y_pred[i, 1]
        max_pred = y_pred[i, 0]

        min_actual = Y_test[i, 1]
        max_actual = Y_test[i, 0]

        list_min_actual.append(min_actual)
        list_max_actual.append(max_actual)

        average_pred = (min_pred + max_pred) / 2
        min_t = average_pred + (min_pred - average_pred) * safety_factor
        max_t = average_pred + (max_pred - average_pred) * safety_factor

        list_min_pred.append(min_t)
        list_max_pred.append(max_t)

    function_make_win_graph(
        list_max_actual=list_max_actual,
        list_min_actual=list_min_actual,
        list_max_pred=list_max_pred,
        list_min_pred=list_min_pred,
        y_type=y_type,
        now_datetime=now_datetime,
    )
    return


def custom_evaluate_safety_factor_band(
    X_test,
    Y_test,
    y_type,
    now_datetime,
    safety_factor,
):
    with custom_object_scope(
        {
            "custom_loss_band": km.custom_loss_band,
            "metric_rmse": km.metric_rmse,
        }
    ):
        model = keras.models.load_model(f"training/models/model - {y_type} - {now_datetime}")
        model.summary()

    y_pred = model.predict(X_test)

    list_min_pred = []
    list_max_pred = []
    list_min_actual = []
    list_max_actual = []

    for i in range(y_pred.shape[0]):
        # i -> day
        all_y_pred_l = y_pred[i, 0 : y_pred.shape[1], 0].tolist()
        all_y_pred_h = y_pred[i, 0 : y_pred.shape[1], 1].tolist()

        all_y_pred_l.sort(reverse=True)
        all_y_pred_h.sort()

        min_pred = all_y_pred_l[int(len(all_y_pred_h) * 0.75) - 1]
        max_pred = all_y_pred_h[int(len(all_y_pred_h) * 0.75) - 1]

        min_actual = min(Y_test[i, :, 0])
        max_actual = max(Y_test[i, :, 1])
        list_min_actual.append(min_actual)
        list_max_actual.append(max_actual)

        average_pred = (min_pred + max_pred) / 2
        min_t = average_pred + (min_pred - average_pred) * safety_factor
        max_t = average_pred + (max_pred - average_pred) * safety_factor

        list_min_pred.append(min_t)
        list_max_pred.append(max_t)

    function_error_132_graph(y_pred=y_pred, y_test=Y_test, now_datetime=now_datetime, y_type=y_type)

    function_make_win_graph(
        list_max_actual=list_max_actual,
        list_min_actual=list_min_actual,
        list_max_pred=list_max_pred,
        list_min_pred=list_min_pred,
        y_type=y_type,
        now_datetime=now_datetime,
    )

    return


def custom_evaluate_safety_factor_band_2(
    X_test,
    Y_test,
    testsize,
    now_datetime,
):
    safety_factor: float = 0.8
    y_type: str = "band"

    with custom_object_scope(
        {
            "custom_loss_band_2": km.custom_loss_band_2,
            "metric_rmse": km.metric_rmse,
            "metric_band_error_average": km.metric_band_error_average,
            "metric_band_hl_correction": km.metric_band_hl_correction,
            "metric_band_inside_range": km.metric_band_inside_range,
        }
    ):
        model = keras.models.load_model(f"training/models/model - {now_datetime} - band")
        model.summary()

    y_pred = model.predict(X_test)
    # [0] is low, [1] is high
    # y_pred shape is (days, minutes, features)

    first_percentile_exclude: float = 0.01
    first_point_taken: int = int(y_pred.shape[0] * first_percentile_exclude)
    # make the y_pred, such that we first percentile array elements is same as the first precentile value
    for i in range(y_pred.shape[0]):
        for j in range(first_point_taken):
            y_pred[i, j, 0] = y_pred[i, first_point_taken, 0]
            y_pred[i, j, 1] = y_pred[i, first_point_taken, 1]
    zeros = np.zeros((y_pred.shape[0], y_pred.shape[1], 2))
    y_pred = np.concatenate((y_pred, zeros), axis=2)

    list_min_pred = []
    list_max_pred = []
    list_min_actual = []
    list_max_actual = []

    for i in range(y_pred.shape[0]):
        # i -> day
        min_actual = min(Y_test[i, :, 0])
        max_actual = max(Y_test[i, :, 1])

        list_min_actual.append(min_actual)
        list_max_actual.append(max_actual)

    prev_val = -1
    for safety_factor_i in [j / 20 for j in range(5, 21)]:
        # safety_factor_i ranges from 0.25 to 0.95

        for i in range(y_pred.shape[0]):
            # i  -> day

            all_y_pred_l = y_pred[i, :, 0]
            all_y_pred_h = y_pred[i, :, 1]

            min_pred = min(all_y_pred_l)
            max_pred = max(all_y_pred_h)

            average_pred = (min_pred + max_pred) / 2
            min_t = average_pred + (min_pred - average_pred) * safety_factor_i
            max_t = average_pred + (max_pred - average_pred) * safety_factor_i

            list_min_pred.append(min_t)
            list_max_pred.append(max_t)

        val = function_make_win_graph(
            list_max_actual=list_max_actual,
            list_min_actual=list_min_actual,
            list_max_pred=list_max_pred,
            list_min_pred=list_min_pred,
            testsize=testsize,
            y_type=y_type,
            max_percentile_found=False,
            now_datetime=now_datetime,
        )
        if val > 0:
            print("sf:", safety_factor_i, "{:0.6f}".format(val))
        if val > prev_val:
            safety_factor = safety_factor_i
            prev_val = val

        list_min_pred.clear()
        list_max_pred.clear()

    # safety factor found
    if prev_val == 0:
        safety_factor = 1

    for i in range(y_pred.shape[0]):
        # i  -> day

        all_y_pred_l = y_pred[i, :, 0]
        all_y_pred_h = y_pred[i, :, 1]

        min_pred = min(all_y_pred_l)
        max_pred = max(all_y_pred_h)

        average_pred = (min_pred + max_pred) / 2
        min_t = average_pred + (min_pred - average_pred) * safety_factor
        max_t = average_pred + (max_pred - average_pred) * safety_factor

        list_min_pred.append(min_t)
        list_max_pred.append(max_t)

    function_error_132_graph(y_pred=y_pred, y_test=Y_test, now_datetime=now_datetime, y_type=y_type)

    print("\nmax_safety_factor\t", safety_factor, "\n")

    function_make_win_graph(
        list_max_actual=list_max_actual,
        list_min_actual=list_min_actual,
        list_max_pred=list_max_pred,
        list_min_pred=list_min_pred,
        testsize=testsize,
        y_type=y_type,
        max_percentile_found=True,
        now_datetime=now_datetime,
    )

    return


def custom_evaluate_safety_factor_2_mods(
    X_test_h,
    Y_test_h,
    X_test_l,
    Y_test_l,
    testsize,
    now_datetime,
):
    y_type: str = "2_mods"

    with custom_object_scope({"custom_loss_2_mods_high": km.custom_loss_2_mods_high, "metric_rmse": km.metric_rmse}):
        model_h = keras.models.load_model(f"training/models/model - {now_datetime} - 2_mods - high")
        model_h.summary()

    with custom_object_scope({"custom_loss_2_mods_low": km.custom_loss_2_mods_low, "metric_rmse": km.metric_rmse}):
        model_l = keras.models.load_model(f"training/models/model - {now_datetime} - 2_mods - low")

    y_pred_h = model_h.predict(X_test_h)
    y_pred_l = model_l.predict(X_test_l)

    list_min_pred = []
    list_max_pred = []
    list_min_actual = []
    list_max_actual = []

    for i in range(y_pred_h.shape[0]):
        # i  -> day
        min_actual = Y_test_l[i, 0, 0]
        max_actual = Y_test_h[i, 0, 0]

        for j in range(y_pred_h.shape[1]):
            min_actual = min(min_actual, Y_test_l[i, j, 0])
            max_actual = max(max_actual, Y_test_h[i, j, 0])

        list_min_actual.append(min_actual)
        list_max_actual.append(max_actual)

    prev_val = -1
    max_percentile = 1
    for percentile in [i / 20 for i in range(5, 21)]:
        # percentile ranges from 0.1 to 1
        for safety_factor_i in [j / 20 for j in range(5, 20)]:
            # safety_factor_i ranges from 0.3 to 0.9
            for i in range(y_pred_h.shape[0]):
                # i  -> day
                all_y_pred_h = y_pred_l[i, :, 0]
                all_y_pred_l = y_pred_h[i, :, 0]

                # for j in range(y_pred_h.shape[1]):
                #     # j -> time
                #     all_y_pred_l.append(y_pred_l[i, j, 0])
                #     all_y_pred_h.append(y_pred_h[i, j, 0])

                all_y_pred_l = sorted(all_y_pred_l, reverse=True)
                all_y_pred_h.sort()

                min_pred = all_y_pred_l[int(len(all_y_pred_h) * percentile) - 1]
                max_pred = all_y_pred_h[int(len(all_y_pred_h) * percentile) - 1]

                average_pred = (min_pred + max_pred) / 2
                min_t = average_pred + (min_pred - average_pred) * safety_factor_i
                max_t = average_pred + (max_pred - average_pred) * safety_factor_i

                list_min_pred.append(min_t)
                list_max_pred.append(max_t)

            val = function_make_win_graph(
                list_max_actual=list_max_actual,
                list_min_actual=list_min_actual,
                list_max_pred=list_max_pred,
                list_min_pred=list_min_pred,
                testsize=testsize,
                y_type=y_type,
                max_percentile_found=False,
                now_datetime=now_datetime,
            )
            if val > 0:
                print("sf:", safety_factor_i, "percentile:", percentile, "{:0.6f}".format(val))
            if val > prev_val:
                max_percentile = percentile
                safety_factor = safety_factor_i
                prev_val = val

            list_min_pred.clear()
            list_max_pred.clear()

    # percentile found
    # safety factor found
    if prev_val == 0:
        max_percentile = 1
        safety_factor = 1

    for i in range(y_pred_h.shape[0]):
        # i  -> day
        all_y_pred_h = []
        all_y_pred_l = []

        for j in range(y_pred_h.shape[1]):
            # j -> time
            all_y_pred_l.append(y_pred_l[i, j, 0])
            all_y_pred_h.append(y_pred_h[i, j, 0])

        all_y_pred_l.sort(reverse=True)
        all_y_pred_h.sort()

        min_pred = all_y_pred_l[int(len(all_y_pred_h) * max_percentile) - 1]
        max_pred = all_y_pred_h[int(len(all_y_pred_h) * max_percentile) - 1]

        average_pred = (min_pred + max_pred) / 2
        min_t = average_pred + (min_pred - average_pred) * safety_factor
        max_t = average_pred + (max_pred - average_pred) * safety_factor

        list_min_pred.append(min_t)
        list_max_pred.append(max_t)

    y_pred = np.concatenate((y_pred_l, y_pred_h), axis=-1)
    Y_test = np.concatenate((Y_test_l, Y_test_h), axis=-1)

    function_error_132_graph(y_pred=y_pred, y_test=Y_test, now_datetime=now_datetime, y_type=y_type)

    print("\nmax_percentile\t", max_percentile, "max_safety_factor\t", safety_factor, "\n")

    function_make_win_graph(
        list_max_actual=list_max_actual,
        list_min_actual=list_min_actual,
        list_max_pred=list_max_pred,
        list_min_pred=list_min_pred,
        testsize=testsize,
        y_type=y_type,
        max_percentile_found=True,
        now_datetime=now_datetime,
    )

    return


def function_make_win_graph(
    list_max_actual: list[float],
    list_min_actual: list[float],
    list_max_pred: list[float],
    list_min_pred: list[float],
    testsize: float,
    max_percentile_found: bool,
    y_type: str,
    now_datetime: str,
):
    """
    Calculates various statistics based on the given lists of actual and predicted values.

    Parameters:
    - list_max_actual (list): A list of maximum actual values.
    - list_min_actual (list): A list of minimum actual values.
    - list_max_pred (list): A list of maximum predicted values.
    - list_min_pred (list): A list of minimum predicted values.
    - y_type (str): The type of y values.
    - safety_factor (float): A safety factor for the calculations.
    - now_datetime (str): The current date and time.

    Returns:
    - float: The percentage of winning days.
    """
    list_pred_avg = []

    res_win = []
    valid_pred = []
    valid_act = []
    valid_max = []
    valid_min = []
    is_average_in = []

    for i in range(len(list_max_actual)):
        min_pred = list_min_pred[i]
        max_pred = list_max_pred[i]
        min_actual = list_min_actual[i]
        max_actual = list_max_actual[i]

        average_pred = (min_pred + max_pred) / 2

        list_pred_avg.append(average_pred)

        win = max_pred < max_actual and min_pred > min_actual and max_pred > min_pred

        valid_pred.append(max_pred > min_pred)
        valid_act.append(max_actual > min_actual)
        valid_max.append(max_pred < max_actual and max_pred > min_actual)
        valid_min.append(min_pred > min_actual and min_pred < max_actual)
        is_average_in.append(average_pred < max_actual and average_pred > min_actual)

        res_win.append(win)

    pred_num: int = 0
    for i in valid_pred:
        if i:
            pred_num += 1

    act_num: int = 0
    for i in valid_act:
        if i:
            act_num += 1

    max_num: int = 0
    for i in valid_max:
        if i:
            max_num += 1

    min_num: int = 0
    for i in valid_min:
        if i:
            min_num += 1

    average_in_num: int = 0
    for i in is_average_in:
        if i:
            average_in_num += 1

    res: dict[str:float] = {}

    average_in_perc: float = round(average_in_num / len(valid_min) * 100, 2)
    y_min: float = min(min(list_min_actual), min(list_min_pred))
    y_max: float = max(max(list_max_actual), max(list_max_pred))

    x: List[int] = [i + 1 for i in range(len(list_max_actual))]

    if max_percentile_found:
        fig = plt.figure(figsize=(16, 9))

        ax = fig.add_subplot(111)

        plt.axvline(x=int(len(list_max_actual) * (1 - testsize)) - 0.5, color="blue")

        plt.fill_between(x, list_min_actual, list_max_actual, color="yellow")

        # plt.scatter(x, list_min_actual, color="orange", s=50)
        # plt.scatter(x, list_max_actual, color="orange", s=50)

        plt.plot(x, list_pred_avg, linestyle="dashed", c="red")

    wins = 0
    total_capture = 0
    pred_capture = 0
    all_days_pro = 1

    for i in range(len(res_win)):
        total_capture += list_max_actual[i] / list_min_actual[i] - 1
        if res_win[i]:
            all_days_pro *= list_max_pred[i] / list_min_pred[i]
            pred_capture += list_max_pred[i] / list_min_pred[i] - 1

            wins += 1
            if max_percentile_found:
                plt.scatter(
                    x[i], y_min - (y_max - y_min) / 100, c="yellow", linewidths=2, marker="^", edgecolor="red", s=125
                )

    win_percent = round((wins / len(res_win)) * 100, 2)
    cdgr = (pow(all_days_pro, 1 / len(res_win)) - 1) * 100

    pred_capture_percent = round((pred_capture / total_capture) * 100, 2)

    avg_captured = 0
    if wins != 0:
        avg_captured = "{:.4f}".format(pred_capture / wins * 100)
    pro_250 = pow(cdgr / 100 + 1, 250) - 1
    pro_250_str = "{:.4f}".format(pro_250)
    pro_250_5 = "{:.4f}".format(pow(cdgr * 5 / 100 + 1, 250) - 1)

    res["pro_250"] = pro_250
    res["win_percent"] = win_percent
    res["pred_capture_percent"] = pred_capture_percent

    if max_percentile_found:
        for i in range(len(list_min_pred)):
            if valid_pred[i]:
                plt.vlines(
                    x=x[i],
                    ymin=list_min_pred[i],
                    ymax=list_max_pred[i],
                    colors="green",
                )

        ax.set_xlabel("days", fontsize=15)
        ax.set_ylabel("fraction of prev close", fontsize=15)

        print("valid_act\t", round(act_num / len(valid_act) * 100, 2), " %")
        print("valid_pred\t", round(pred_num / len(valid_pred) * 100, 2), " %")
        print("max_inside\t", round(max_num / len(valid_max) * 100, 2), " %")
        print("min_inside\t", round(min_num / len(valid_min) * 100, 2), " %\n")
        print("average_in\t", average_in_perc, " %\n")

        print("win_days_perc\t", win_percent, " %")
        print("pred_capture\t", pred_capture_percent, " %")
        print("per_day\t\t", avg_captured, " %")
        print("250 days:\t", pro_250_str)
        print("\nleverage:\t", pro_250_5)
        print("datetime:\t", now_datetime)

        ax.set_title(
            f" name: {now_datetime} \n\n wins: {win_percent}% || average_in: {average_in_perc}% || pred_capture: {pred_capture_percent}% || 250 days: {pro_250_str}",
            fontsize=20,
        )

        filename = f"training/graphs/{y_type} - {now_datetime} - Splot.png"

        plt.savefig(filename, dpi=1500, bbox_inches="tight")

        plt.show()  # temp_now

        print("\n\nNUMBER_OF_NEURONS\t\t", km.NUMBER_OF_NEURONS)
        print("NUMBER_OF_LAYERS\t\t", km.NUMBER_OF_LAYERS)
        print("INITIAL_DROPOUT\t\t\t", km.INITIAL_DROPOUT)

        if y_type != "band" or y_type != "band_2":
            print("ERROR_AMPLIFICATION_FACTOR\t", km.ERROR_AMPLIFICATION_FACTOR, end="\n\n")

    return res


def get_hl_list(y_pred: np.ndarray, i: int) -> Tuple[List[float], List[float]]:
    all_y_pred_l: List[float] = []
    all_y_pred_h: List[float] = []

    for j in range(y_pred.shape[1]):
        all_y_pred_l.append(y_pred[i, j, 0])
        all_y_pred_h.append(y_pred[i, j, 1])

    return all_y_pred_l, all_y_pred_h


def get_hl_list_2(y_pred: np.ndarray, day: int, band_height: float) -> Tuple[List[float], List[float]]:
    all_y_pred_l: List[float] = []
    all_y_pred_h: List[float] = []

    for minute in range(y_pred.shape[1]):
        all_y_pred_l.append(y_pred[day, minute, 0] - abs(band_height) / 2)
        all_y_pred_h.append(y_pred[day, minute, 0] + abs(band_height) / 2)
    return all_y_pred_l, all_y_pred_h


def get_hl_list_3(y_pred: np.ndarray, day: int, band_height: float) -> Tuple[float, float]:
    min_val: float = min(y_pred[day, :, 0]) - band_height / 2
    max_val: float = max(y_pred[day, :, 0]) + band_height / 2

    return min_val, max_val


def custom_evaluate_safety_factor_band_2_2(
    X_test,
    Y_test,
    y_type: str,
    testsize: float = 0,
    now_datetime: str = "2020-01-01 00-00",
):
    # convert y_test to same format as y_pred
    with custom_object_scope(
        {
            "metric_new_idea_2": km.metric_new_idea_2,
            "metric_rmse": km.metric_rmse,
            "metric_band_average": km.metric_band_average,
            "metric_band_height": km.metric_band_height,
            "metric_band_hl_wrongs_percent": km.metric_band_hl_wrongs_percent,
            "metric_loss_comp_2": km.metric_loss_comp_2,
            "metric_pred_capture_percent": km.metric_pred_capture_percent,
            "metric_band_height_percent": km.metric_band_height_percent,
        }
    ):
        model = keras.models.load_model(f"training/models/model - {now_datetime} - {y_type}")
        model.summary()

    y_pred = model.predict(X_test)

    # h = km.metric_loss_comp_2_1212(y_pred, Y_test)

    # # transform y data from (0, 1) = (avg, band) to (low, high)
    # y_pred_temp = np.zeros_like(y_pred)
    # y_pred_temp[:, :, 0] = y_pred[:, :, 0] - y_pred[:, :, 1] / 2
    # y_pred_temp[:, :, 1] = y_pred[:, :, 0] + y_pred[:, :, 1] / 2
    # y_pred = y_pred_temp

    # taking only first 2 columns of y_test
    Y_test = Y_test[:, :, :2]

    # Y_test_temp = np.zeros_like(Y_test)
    # Y_test_temp[:, :, 0] = Y_test[:, :, 0] - Y_test[:, :, 1] / 2
    # Y_test_temp[:, :, 1] = Y_test[:, :, 0] + Y_test[:, :, 1] / 2
    # Y_test = Y_test_temp

    # [0] is low, [1] is high
    # y_pred shape is (days, minutes, features)

    # first_percentile_exclude: float = 0.01
    # first_point_taken: int = int(y_pred.shape[0] * first_percentile_exclude)
    # # make the y_pred, such that we first percentile array elements is same as the first precentile value
    # for i in range(y_pred.shape[0]):
    #     for j in range(first_point_taken):
    #         y_pred[i, j, 0] = y_pred[i, first_point_taken, 0]
    #         y_pred[i, j, 1] = y_pred[i, first_point_taken, 1]

    # #  making y_pred, of (,,2) to (,,4)
    # zeros = np.zeros((y_pred.shape[0], y_pred.shape[1], 2))
    # y_pred = np.concatenate((y_pred, zeros), axis=2)

    list_min_pred: list[float] = []
    list_max_pred: list[float] = []
    list_min_actual: list[float] = []
    list_max_actual: list[float] = []

    for i in range(y_pred.shape[0]):
        # i -> day
        # min_actual = min(Y_test[i, :, 0])
        # max_actual = max(Y_test[i, :, 1])

        min_actual = min(Y_test[i, :, 0] - Y_test[i, :, 1] / 2)
        max_actual = max(Y_test[i, :, 0] + Y_test[i, :, 1] / 2)

        list_min_actual.append(min_actual)
        list_max_actual.append(max_actual)

    prev_val: float = -1
    max_band_percentile_sorted: float = 0.5

    for band_percentile in [j / 20 for j in range(21)]:
        for i in range(y_pred.shape[0]):
            # trying all band_height s in that day. given the average for that day is remains the same.
            band_heights: list[float] = y_pred[i, :, 1]

            sorted_band_heights = sorted(band_heights)

            # all_y_pred_l, all_y_pred_h = get_hl_list_2(
            #     y_pred=y_pred,
            #     day=i,
            #     band_height=np.percentile(sorted_band_heights, band_percentile * 100),
            # )
            # list_min_pred.append(min(all_y_pred_l))
            # list_max_pred.append(max(all_y_pred_h))

            min_pred_val, max_pred_val = get_hl_list_3(
                y_pred=y_pred,
                day=i,
                band_height=np.percentile(sorted_band_heights, band_percentile * 100),
            )
            list_min_pred.append(min_pred_val)
            list_max_pred.append(max_pred_val)

        temp = function_make_win_graph(
            list_max_actual=list_max_actual,
            list_min_actual=list_min_actual,
            list_max_pred=list_max_pred,
            list_min_pred=list_min_pred,
            testsize=testsize,
            y_type=y_type,
            max_percentile_found=False,
            now_datetime=now_datetime,
        )
        val = temp["pro_250"]
        if val > 0:
            print("band_percentile:\t", band_percentile, "{:0.6f}".format(val))
            if val > prev_val:
                max_band_percentile_sorted = band_percentile
                prev_val = val

        list_min_pred.clear()
        list_max_pred.clear()

    for i in range(y_pred.shape[0]):
        band_heights: list[float] = y_pred[i, :, 1]

        sorted_band_heights = sorted(band_heights)

        # all_y_pred_l, all_y_pred_h = get_hl_list_2(
        #     y_pred=y_pred,
        #     day=i,
        #     band_height=np.percentile(sorted_band_heights, max_band_percentile_sorted * 100),
        # )
        # list_min_pred.append(min(all_y_pred_l))
        # list_max_pred.append(max(all_y_pred_h))

        # min, max for that day _pred
        min_pred_val, max_pred_val = get_hl_list_3(
            y_pred=y_pred,
            day=i,
            band_height=np.percentile(sorted_band_heights, max_band_percentile_sorted * 100),
        )
        list_min_pred.append(min_pred_val)
        list_max_pred.append(max_pred_val)

    # prev_val = -1
    # for safety_factor_i in [j / 20 for j in range(5, 21)]:
    #     # safety_factor_i ranges from 0.25 to 0.95

    #     for i in range(y_pred.shape[0]):
    #         # i  -> day
    #         all_y_pred_l, all_y_pred_h = get_hl_list(y_pred=y_pred, i=i)

    #         min_pred = min(all_y_pred_l)
    #         max_pred = max(all_y_pred_h)

    #         average_pred = (min_pred + max_pred) / 2
    #         min_t = average_pred + (min_pred - average_pred) * safety_factor_i
    #         max_t = average_pred + (max_pred - average_pred) * safety_factor_i

    #         list_min_pred.append(min_t)
    #         list_max_pred.append(max_t)

    #     val = function_make_win_graph(
    #         list_max_actual=list_max_actual,
    #         list_min_actual=list_min_actual,
    #         list_max_pred=list_max_pred,
    #         list_min_pred=list_min_pred,
    #         testsize=testsize,
    #         y_type=y_type,
    #         max_percentile_found=False,
    #         now_datetime=now_datetime,
    #     )
    #     if val > 0:
    #         print("sf:\t", safety_factor_i, "{:0.6f}".format(val))
    #         if val > prev_val:
    #             safety_factor = safety_factor_i
    #             prev_val = val

    #     list_min_pred.clear()
    #     list_max_pred.clear()

    # # safety factor found
    # if prev_val == 0:
    #     safety_factor = 1

    # for i in range(y_pred.shape[0]):
    #     # i  -> day

    #     # all_y_pred_l = y_pred[i, :, 0]
    #     # all_y_pred_h = y_pred[i, :, 1]

    #     # removing pairs that have hl error
    #     all_y_pred_l, all_y_pred_h = get_hl_list(y_pred=y_pred, i=i)

    #     min_pred = min(all_y_pred_l)
    #     max_pred = max(all_y_pred_h)

    #     average_pred = (min_pred + max_pred) / 2
    #     min_t = average_pred + (min_pred - average_pred) * safety_factor
    #     max_t = average_pred + (max_pred - average_pred) * safety_factor

    #     list_min_pred.append(min_t)
    #     list_max_pred.append(max_t)

    # function_error_132_graph(y_pred=y_pred, y_test=Y_test, now_datetime=now_datetime, y_type=y_type)

    function_make_win_graph_2(
        y_true=Y_test,
        y_pred=y_pred,
        testsize=testsize,
        y_type=y_type,
        now_datetime=now_datetime,
        make_graph=True,
    )

    return
