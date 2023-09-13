import training_yf as an
from tensorflow import keras
from keras.callbacks import TensorBoard, ModelCheckpoint
from time import time
from datetime import datetime
import keras_model as km


def main():
    NUMBER_OF_EPOCHS = 1000
    BATCH_SIZE = 32
    SAFETY_FACTOR = 0.8
    LEARNING_RATE = 0.0001
    TEST_SIZE = 0.2
    Y_TYPE = "hl"

    TICKER = "ADANIPORTS.NS"
    INTERVAL = "1m"

    load_model_1 = 0
    check_on_test_1 = 0

    df = an.get_data_all_df(ticker=TICKER, interval=INTERVAL)

    (X_train, Y_train), (X_test, Y_test) = an.train_test_split(
        data_df=df, test_size=TEST_SIZE, y_type=Y_TYPE, interval=INTERVAL
    )

    load_model = bool(load_model_1)
    check_on_test = bool(check_on_test_1)
    now_datetime = datetime.now().strftime("%Y-%m-%d %H-%M")

    if load_model:
        model = keras.models.load_model("models/model - band - 2023-09-06 20-52")
        print("\nmodel loaded.\n")
        model.summary()

    else:
        model = km.get_untrained_model(X_train=X_train, y_type=Y_TYPE)

        print("training data shape\t", X_train.shape)
        print("training elememt shape\t", X_train[0].shape)

        print("model output shape\t", model.output_shape)

        log_dir = f"logs/{now_datetime}"  # Directory where you want to store the TensorBoard logs
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        # terminal command: tensorboard --logdir=logs/
        # log directory: logs/train/

        optimizer = km.get_optimiser(learning_rate=LEARNING_RATE)

        loss = keras.losses.MeanSquaredError()

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=["mae", "mse"],
        )

        history = model.fit(
            x=X_train,
            y=Y_train,
            epochs=NUMBER_OF_EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_test, Y_test),
            callbacks=[tensorboard_callback],
        )

        model.save(f"models/model - {Y_TYPE} - {now_datetime}")

        print("\nmodel training done.\n")

    # if check_on_test:
    #     loss = model.evaluate(X_test, Y_test)

    #     win_percent = an.custom_evaluate_full_envelope(
    #         model=model, X_test=X_test, Y_test=Y_test
    #     )
    # else:
    #     loss = model.evaluate(X_train, Y_train)

    #     win_percent = an.custom_evaluate_full_envelope(
    #         model=model, X_test=X_train, Y_test=Y_train
    #     )

    if not check_on_test:
        X_test = X_train
        Y_test = Y_train

    win_percent = an.custom_evaluate_safety_factor(
        model=model,
        X_test=X_test,
        Y_test=Y_test,
        now_datetime=now_datetime,
        y_type=Y_TYPE,
        safety_factor=SAFETY_FACTOR,
    )

    print("now_datetime: ", now_datetime)
    # print(f"win_percent: {win_percent}")


if __name__ == "__main__":
    time_1 = time()
    main()
    print(f"\ntime taken = {round(time() - time_1, 2)} sec\n")
