import multiprocessing
import time
from datetime import datetime

import keras_model as km
import numpy as np
import training_yf as an
from keras.callbacks import TensorBoard, TerminateOnNaN
from keras.utils import custom_object_scope
from sklearn.model_selection import train_test_split
from tensorflow import keras

# 2_mods = 2 hl models
# terminal command: tensorboard --logdir=training/logs/


NUMBER_OF_EPOCHS: int = 2000
BATCH_SIZE: int = 128
LEARNING_RATE: float = 0.0001
TEST_SIZE: float = 0.2

Y_TYPE: str = "band_2"

# Y_TYPE = "2_mods" / "band" / band_2


TICKER: str = "CCI"
INTERVAL: str = "1m"

IS_TRAINING_MODEL: bool = True

PREV_MODEL_TRAINING: bool = False


def main():
    df = an.get_data_all_df(ticker=TICKER, interval=INTERVAL)

    prev_model: str = "2023-11-18 16-40"

    num_cores: int = multiprocessing.cpu_count()
    # total cores = 8 in this mac.
    num_workers: int = 1

    if Y_TYPE == "2_mods":
        data_dict = {}

        if IS_TRAINING_MODEL and not PREV_MODEL_TRAINING:
            now_datetime = datetime.now().strftime("%Y-%m-%d %H-%M")
        else:
            now_datetime = prev_model

        # # get previous log folders
        # parent_folder_path = os.getcwd()
        # for item in os.listdir(os.path.join(parent_folder_path, "training/logs")):
        #     item_path = os.path.join(parent_folder_path, "training/logs", item)
        #     if os.path.isdir(item_path):
        #         shutil.rmtree(item_path)
        #         time.sleep(31)

        for key in ["high", "low"]:
            data_x, data_y = an.get_x_y_individual_data(
                data_df=df,
                interval=INTERVAL,
                columns=[key],
            )
            # TODOO: write a custom shuffle function.

            X_train, X_test, Y_train, Y_test = train_test_split(
                data_x,
                data_y,
                test_size=TEST_SIZE,
                shuffle=False,
            )

            data_dict[key] = {}
            data_dict[key]["train_x"] = X_train
            data_dict[key]["train_y"] = Y_train
            data_dict[key]["test_x"] = X_test
            data_dict[key]["test_y"] = Y_test

            if IS_TRAINING_MODEL:
                if PREV_MODEL_TRAINING:
                    with custom_object_scope(
                        {
                            "custom_loss_2_mods_high": km.custom_loss_2_mods_high,
                            "custom_loss_2_mods_low": km.custom_loss_2_mods_low,
                            "metric_rmse": km.metric_rmse,
                        },
                    ):
                        model = keras.models.load_model(
                            f"training/models_saved/model - {prev_model} - {Y_TYPE} - {key}",
                        )
                        model.summary()
                else:
                    model: keras.Model = km.get_untrained_model(
                        X_train=X_train,
                        y_type=Y_TYPE,
                    )

                print("training data shape\t", X_train.shape)
                print("training element shape\t", X_train[0].shape)

                print("model output shape\t", model.output_shape)

                log_dir: str = f"training/logs/{now_datetime}-{key}"
                tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
                loss_diff_callback = km.LossDifferenceCallback(log_dir=log_dir)

                optimizer = km.get_optimiser(learning_rate=LEARNING_RATE)

                if key == "high":
                    model.compile(
                        optimizer=optimizer,
                        loss=km.custom_loss_2_mods_high,
                        metrics=["mae", km.metric_rmse],
                    )

                elif key == "low":
                    model.compile(
                        optimizer=optimizer,
                        loss=km.custom_loss_2_mods_low,
                        metrics=["mae", km.metric_rmse],
                    )

                callbacks = [tensorboard_callback]
                # callbacks = [tensorboard_callback, loss_diff_callback]
                X_train = X_train.astype(np.float32)
                Y_train = Y_train.astype(np.float32)

                model.fit(
                    x=X_train,
                    y=Y_train,
                    epochs=NUMBER_OF_EPOCHS,
                    batch_size=BATCH_SIZE,
                    workers=num_workers,
                    use_multiprocessing=True,
                    validation_data=(X_test, Y_test),
                    callbacks=callbacks,
                )

                model.save(f"training/models/model - {now_datetime} - {Y_TYPE} - {key}")

                print(f"\nmodel - {key} : training done. \n")

            X_test_h = np.append(
                data_dict["high"]["train_x"],
                data_dict["high"]["test_x"],
                axis=0,
            )
            Y_test_h = np.append(
                data_dict["high"]["train_y"],
                data_dict["high"]["test_y"],
                axis=0,
            )
            X_test_l = np.append(
                data_dict["low"]["train_x"],
                data_dict["low"]["test_x"],
                axis=0,
            )
            Y_test_l = np.append(
                data_dict["low"]["train_y"],
                data_dict["low"]["test_y"],
                axis=0,
            )

            an.custom_evaluate_safety_factor_2_mods(
                X_test_h=X_test_h,
                Y_test_h=Y_test_h,
                X_test_l=X_test_l,
                Y_test_l=Y_test_l,
                testsize=TEST_SIZE,
                now_datetime=now_datetime,
            )

    elif Y_TYPE == "band":
        (X_train, Y_train), (X_test, Y_test) = an.train_test_split(
            data_df=df,
            test_size=TEST_SIZE,
            y_type=Y_TYPE,
            interval=INTERVAL,
        )

        if IS_TRAINING_MODEL and not PREV_MODEL_TRAINING:
            now_datetime = datetime.now().strftime("%Y-%m-%d %H-%M")
        else:
            now_datetime = prev_model

        if IS_TRAINING_MODEL:
            model = km.get_untrained_model(X_train=X_train, y_type=Y_TYPE)

            print("training data shape\t", X_train.shape)
            print("training element shape\t", X_train[0].shape)

            print("model output shape\t", model.output_shape)

            # Directory where you want to store the TensorBoard logs
            log_dir = f"training/logs/{now_datetime} - {Y_TYPE}"

            tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
            loss_diff_callback = km.LossDifferenceCallback(log_dir=log_dir)

            optimizer = km.get_optimiser(learning_rate=LEARNING_RATE)

            loss = km.custom_loss_band_2

            model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=[
                    "mae",
                    km.metric_rmse,
                    km.metric_band_error_average,
                    km.metric_band_hl_correction,
                    km.metric_band_inside_range,
                ],
            )

            # callbacks = [tensorboard_callback, loss_diff_callback]
            callbacks = [tensorboard_callback]

            model.fit(
                x=X_train,
                y=Y_train,
                epochs=NUMBER_OF_EPOCHS,
                batch_size=BATCH_SIZE,
                workers=num_workers,
                use_multiprocessing=True,
                validation_data=(X_test, Y_test),
                callbacks=callbacks,
            )

            model.save(f"training/models/model - {now_datetime} - {Y_TYPE}")

            print("\nmodel : training done. \n")

            X_test = np.append(X_train, X_test, axis=0)
            Y_test = np.append(Y_train, Y_test, axis=0)
            zeros = np.zeros((Y_test.shape[0], Y_test.shape[1], 2))
            Y_test = np.concatenate((Y_test, zeros), axis=2)

            print(f"\n\nnow_datetime:\t{now_datetime}\n\n")
            print("-" * 30)

            an.custom_evaluate_safety_factor_band_2(
                X_test=X_test,
                Y_test=Y_test,
                testsize=TEST_SIZE,
                now_datetime=now_datetime,
            )

    if Y_TYPE == "band_2":
        (X_train, Y_train), (X_test, Y_test), x_close = an.train_test_split(
            data_df=df,
            test_size=TEST_SIZE,
            y_type=Y_TYPE,
            interval=INTERVAL,
        )

        if IS_TRAINING_MODEL and not PREV_MODEL_TRAINING:
            now_datetime = datetime.now().strftime("%Y-%m-%d %H-%M")
        else:
            now_datetime = prev_model

        if IS_TRAINING_MODEL:
            model = km.get_untrained_model(X_train=X_train, y_type=Y_TYPE)

            print("training data shape\t", X_train.shape)
            print("training element shape\t", X_train[0].shape)

            print("model output shape\t", model.output_shape)

            optimizer = km.get_optimiser(learning_rate=LEARNING_RATE)

            loss = km.metric_new_idea_2

            model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=[
                    km.metric_band_base_percent,
                    km.metric_loss_comp_2,
                    km.metric_band_hl_wrongs_percent,
                    km.metric_band_avg_correction_percent,
                    km.metric_band_average_percent,
                    km.metric_band_height_percent,
                    km.metric_win_percent,
                    km.metric_pred_capture_percent,
                    km.metric_win_pred_capture_percent,
                ],
            )

            log_dir: str = f"training/logs/{now_datetime} - {Y_TYPE}"

            tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
            terNan = TerminateOnNaN()

            callbacks = [tensorboard_callback, terNan]

            history = model.fit(
                x=X_train,
                y=Y_train,
                epochs=NUMBER_OF_EPOCHS,
                batch_size=BATCH_SIZE,
                workers=num_workers,
                use_multiprocessing=True,
                validation_data=(X_test, Y_test),
                callbacks=callbacks,
            )

            model.save(f"training/models/model - {now_datetime} - {Y_TYPE}")

            print("\nmodel : training done. \n")

            X_test, Y_test = an.append_test_train_arr(X_train, Y_train, X_test, Y_test)

            zeros = np.zeros((Y_test.shape[0], Y_test.shape[1], 2))
            Y_test = np.concatenate((Y_test, zeros), axis=2)

            print(f"\n\nnow_datetime:\t{now_datetime}\n\n")
            print("-" * 30)

            an_2.custom_evaluate_safety_factor(
                X_test=X_test,
                Y_test=Y_test,
                x_close=x_close,
                y_type=Y_TYPE,
                test_size=TEST_SIZE,
                now_datetime=now_datetime,
            )


if __name__ == "__main__":
    time_1 = time.time()
    main()
    print(f"\ntime taken = {round(time.time() - time_1, 2)} sec\n")
