import multiprocessing
import time
from datetime import datetime

import band_2.training_yf_band_2 as an_2
import band_4.keras_model_band_4 as km_4
import band_4.training_yf_band_4 as an_4
import keras_model as km
import numpy as np
import training_yf as an
from keras.callbacks import ModelCheckpoint, TensorBoard, TerminateOnNaN

IS_TRAINING_MODEL: bool = True
prev_model: str = "2024-01-08 16-59"


NUMBER_OF_EPOCHS: int = 5000
BATCH_SIZE: int = 256
LEARNING_RATE: float = 0.0001
TEST_SIZE: float = 0.2

Y_TYPE: str = "band_4"

TICKER: str = "CCI"
INTERVAL: str = "1m"

PREV_MODEL_TRAINING: bool = False

# terminal command: tensorboard --logdir=training/logs/


def main():
    df = an.get_data_all_df(ticker=TICKER, interval=INTERVAL)

    # total cores = 8 in this mac.
    num_cores: int = multiprocessing.cpu_count()
    num_workers: int = 1

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
            print("training elememt shape\t", X_train[0].shape)

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

        print(f"\n\nnow_datatime:\t{now_datetime}\n\n")
        print("-" * 30)

        an_2.custom_evaluate_safety_factor(
            X_test=X_test,
            Y_test=Y_test,
            x_close=x_close,
            y_type=Y_TYPE,
            testsize=TEST_SIZE,
            now_datetime=now_datetime,
        )

    elif Y_TYPE == "band_4":
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
            print("training elememt shape\t", X_train[0].shape)

            print("model output shape\t", model.output_shape)

            optimizer = km.get_optimiser(learning_rate=LEARNING_RATE)

            loss = km_4.metric_new_idea

            model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=[
                    km.metric_abs_percent,
                    km.metric_rmse_percent,
                    km_4.metric_loss_comp_2,
                    km_4.metric_win_percent,
                    km_4.metric_pred_capture_percent,
                    km_4.metric_win_pred_capture_percent,
                ],
            )

            log_dir: str = f"training/logs/{now_datetime} - {Y_TYPE}"

            tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
            terNan = TerminateOnNaN()

            mcp_save = ModelCheckpoint(
                f"training/models/model - {now_datetime} - {Y_TYPE} - modelCheckPoint",
                save_best_only=True,
                monitor="val_metric_rmse_percent",
                mode="min",
            )

            callbacks = [tensorboard_callback, terNan, mcp_save]

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

        X_all, Y_all = an.append_test_train_arr(X_train, Y_train, X_test, Y_test)

        print(f"\n\nnow_datatime:\t{now_datetime}\n\n")
        print("-" * 30)

        an_4.custom_evaluate_safety_factor(
            X_data=X_all,
            Y_data=Y_all,
            y_type=Y_TYPE,
            testsize=TEST_SIZE,
            now_datetime=now_datetime,
        )

        print("\n" * 4, "*" * 2000, "\n" * 4)
        print("only validation data now")

        an_4.custom_evaluate_safety_factor(
            X_data=X_test,
            Y_data=Y_test,
            y_type=Y_TYPE,
            testsize=0,
            now_datetime=now_datetime,
        )


if __name__ == "__main__":
    time_1 = time.time()
    main()
    print(f"\ntime taken = {round(time.time() - time_1, 2)} sec\n")
