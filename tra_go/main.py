import time
from datetime import datetime

import band_4.keras_model_band_4_old as km_4
import keras_model as km
import training_zero as an
from band_4.training_yf_band_4 import CustomEvaluation
from keras.callbacks import ModelCheckpoint, TensorBoard, TerminateOnNaN

IS_TRAINING_MODEL: bool = False
prev_model: str = "2024-03-08 11-41"


NUMBER_OF_EPOCHS: int = 3000
BATCH_SIZE: int = 512
LEARNING_RATE: float = 0.0001
TEST_SIZE: float = 0.2

Y_TYPE: str = "band_4"

TICKER: str = "CCI"
INTERVAL: str = "1m"

PREV_MODEL_TRAINING: bool = False


def main():
    df = an.get_data_all_df(ticker=TICKER, interval=INTERVAL)

    if Y_TYPE == "band_4":
        (X_train, Y_train, train_prev_close), (X_test, Y_test, test_prev_close) = an.train_test_split(
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
            # model = km.get_untrained_model(X_train=X_train, y_type=Y_TYPE)
            model = km.get_untrained_model_new(X_train=X_train)

            print("training data shape\t", X_train.shape)
            print("training elememt shape\t", X_train[0].shape)

            print("model output shape\t", model.output_shape)

            optimizer = km.get_optimiser(learning_rate=LEARNING_RATE)

            loss = km_4.metric_new_idea

            model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=[
                    km.metric_rmse_percent,
                    km.metric_abs_percent,
                    km_4.metric_loss_comp_2,
                    km_4.metric_win_percent,
                    km_4.metric_win_pred_capture_percent,
                    km_4.metric_win_checkpoint,
                ],
            )

            log_dir: str = f"training/logs/{now_datetime} - {Y_TYPE}"

            tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
            terNan = TerminateOnNaN()

            mcp_save_1 = ModelCheckpoint(
                f"training/models/model - {now_datetime} - {Y_TYPE} - modelCheckPoint-1.keras",
                save_best_only=True,
                monitor="loss",
                mode="min",
            )

            mcp_save_2 = ModelCheckpoint(
                f"training/models/model - {now_datetime} - {Y_TYPE} - modelCheckPoint-2.keras",
                save_best_only=True,
                monitor="val_loss",
                mode="min",
            )

            mcp_save_3 = ModelCheckpoint(
                f"training/models/model - {now_datetime} - {Y_TYPE} - modelCheckPoint-3.keras",
                save_best_only=True,
                monitor="metric_win_pred_capture_percent",
                mode="max",
            )

            mcp_save_4 = ModelCheckpoint(
                f"training/models/model - {now_datetime} - {Y_TYPE} - modelCheckPoint-4.keras",
                save_best_only=True,
                monitor="val_metric_win_pred_capture_percent",
                mode="max",
            )

            callbacks = [tensorboard_callback, terNan, mcp_save_1, mcp_save_2, mcp_save_3, mcp_save_4]

            history = model.fit(
                x=X_train,
                y=Y_train,
                epochs=NUMBER_OF_EPOCHS,
                batch_size=BATCH_SIZE,
                validation_data=(X_test, Y_test),
                callbacks=callbacks,
            )

            model.save(f"training/models/model - {now_datetime} - {Y_TYPE}.keras")

            print("\nmodel : training done. \n")

        print(f"\n\nnow_datatime:\t{now_datetime}\n\n")
        print("-" * 30)

        print("\n" * 4, "*" * 500, "\n" * 4)
        print("only training data now")

        training_data_custom_evaluation = CustomEvaluation(
            X_data=X_train,
            Y_data=Y_train,
            prev_close=train_prev_close,
            y_type=Y_TYPE,
            test_size=TEST_SIZE,
            now_datetime=now_datetime,
        )

        print("\n" * 4, "*" * 500, "\n" * 4)
        print("only validation data now")

        valid_data_custom_evaluation = CustomEvaluation(
            X_data=X_test,
            Y_data=Y_test,
            prev_close=test_prev_close,
            y_type=Y_TYPE,
            test_size=0,
            now_datetime=now_datetime,
        )


if __name__ == "__main__":
    time_1 = time.time()
    main()
    print(f"\ntime taken = {round(time.time() - time_1, 2)} sec\n")
