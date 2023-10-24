import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint
from time import time
from datetime import datetime
import multiprocessing
import numpy as np
from keras.utils import custom_object_scope
from sklearn.model_selection import train_test_split

import keras_model as km
import training_yf as an

# 2_mods = 2 hl models
# commands
# terminal command: tensorboard --logdir=training/logs/


NUMBER_OF_EPOCHS: int = 400
BATCH_SIZE: int = 128
SAFETY_FACTOR: float = 0.8
LEARNING_RATE: float = 0.0001
TEST_SIZE: float = 0.2

Y_TYPE: str = "2_mods"

TICKER: str = "CCI"
INTERVAL: str = "1m"

IS_TRAINING_MODEL: bool = True

PREV_MODEL_TRAINING: bool = False


def main():
    df = an.get_data_all_df(ticker=TICKER, interval=INTERVAL)

    prev_model: str = "2023-10-22 23-48"

    num_cores: int = multiprocessing.cpu_count()
    # total cores = 8 in this mac.
    num_workers: int = 1

    if Y_TYPE == "2_mods":
        data_dict = {}

        if IS_TRAINING_MODEL and not PREV_MODEL_TRAINING:
            now_datetime = datetime.now().strftime("%Y-%m-%d %H-%M")
        else:
            now_datetime = prev_model

        for key in ["high", "low"]:
            data_x, data_y = an.get_x_y_individual_data(data_df=df, interval=INTERVAL, columns=[key])
            # TODOO: write a custom shuffle function.

            X_train, X_test, Y_train, Y_test = train_test_split(data_x, data_y, test_size=TEST_SIZE, shuffle=False)

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
                        }
                    ):
                        model = keras.models.load_model(
                            f"training/models_saved/model - {prev_model} - {Y_TYPE} - {key}"
                        )
                        model.summary()
                else:
                    model: keras.Model = km.get_untrained_model(X_train=X_train, y_type=Y_TYPE)

                print("training data shape\t", X_train.shape)
                print("training elememt shape\t", X_train[0].shape)

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

        X_test_h = np.append(data_dict["high"]["train_x"], data_dict["high"]["test_x"], axis=0)
        Y_test_h = np.append(data_dict["high"]["train_y"], data_dict["high"]["test_y"], axis=0)
        X_test_l = np.append(data_dict["low"]["train_x"], data_dict["low"]["test_x"], axis=0)
        Y_test_l = np.append(data_dict["low"]["train_y"], data_dict["low"]["test_y"], axis=0)

        an.custom_evaluate_safety_factor_2_mods(
            X_test_h=X_test_h,
            Y_test_h=Y_test_h,
            X_test_l=X_test_l,
            Y_test_l=Y_test_l,
            testsize=TEST_SIZE,
            now_datetime=now_datetime,
            safety_factor=SAFETY_FACTOR,
        )

    elif Y_TYPE == "band":
        (X_train, Y_train), (X_test, Y_test) = an.train_test_split(
            data_df=df, test_size=TEST_SIZE, y_type=Y_TYPE, interval=INTERVAL
        )

        if IS_TRAINING_MODEL:
            now_datetime = datetime.now().strftime("%Y-%m-%d %H-%M")

            model = km.get_untrained_model(X_train=X_train, y_type=Y_TYPE)

            print("training data shape\t", X_train.shape)
            print("training elememt shape\t", X_train[0].shape)

            print("model output shape\t", model.output_shape)

            # Directory where you want to store the TensorBoard logs
            log_dir = f"training/logs/{Y_TYPE} - {now_datetime}"
            tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
            loss_diff_callback = km.LossDifferenceCallback(log_dir=log_dir)

            optimizer = km.get_optimiser(learning_rate=LEARNING_RATE)

            loss = km.custom_loss_band

            model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=["mae", km.metric_rmse],
            )

            callbacks = [tensorboard_callback, loss_diff_callback]

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

            model.save(f"training/models/model - {Y_TYPE} - {now_datetime}")

            print("\nmodel : training done. \n")

        an.custom_evaluate_safety_factor_band(
            X_test=X_test,
            Y_test=Y_test,
            now_datetime=now_datetime,
            y_type=Y_TYPE,
            safety_factor=SAFETY_FACTOR,
        )


if __name__ == "__main__":
    time_1 = time()
    main()
    print(f"\ntime taken = {round(time() - time_1, 2)} sec\n")
