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
