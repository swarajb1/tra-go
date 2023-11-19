import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint
import time
from datetime import datetime
import multiprocessing
import numpy as np
from keras.utils import custom_object_scope
from sklearn.model_selection import train_test_split

import keras_model as km
import training_yf as an


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

    prev_model: str = "2023-11-19 10-18"

    num_cores: int = multiprocessing.cpu_count()
    # total cores = 8 in this mac.
    num_workers: int = 1

    if Y_TYPE == "band_2":
        (X_train, Y_train), (X_test, Y_test) = an.train_test_split(
            data_df=df, test_size=TEST_SIZE, y_type=Y_TYPE, interval=INTERVAL
        )

        if IS_TRAINING_MODEL and not PREV_MODEL_TRAINING:
            now_datetime = datetime.now().strftime("%Y-%m-%d %H-%M")
        else:
            now_datetime = prev_model

        if IS_TRAINING_MODEL:
            model = km.get_untrained_model(X_train=X_train, y_type=Y_TYPE)

            print("training data shape\t", X_train.shape)
            print("training elememt shape\t", X_train[0].shape)

            print("model output shape\t", model.output_shape)

            # Directory where you want to store the TensorBoard logs
            log_dir = f"training/logs/{now_datetime} - {Y_TYPE}"

            tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
            loss_diff_callback = km.LossDifferenceCallback(log_dir=log_dir)

            optimizer = km.get_optimiser(learning_rate=LEARNING_RATE)

            loss = km.custom_loss_band_2_2

            model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=[
                    "mae",
                    km.metric_rmse,
                    km.metric_band_average,
                    km.metric_band_height,
                    km.metric_band_hl_wrongs_percent,
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

        print(f"\n\nnow_datatime:\t{now_datetime}\n\n")
        print("-" * 30)

        an.custom_evaluate_safety_factor_band_2_2(
            X_test=X_test,
            Y_test=Y_test,
            y_type=Y_TYPE,
            testsize=TEST_SIZE,
            now_datetime=now_datetime,
        )


if __name__ == "__main__":
    time_1 = time.time()
    main()
    print(f"\ntime taken = {round(time.time() - time_1, 2)} sec\n")
