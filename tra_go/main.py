from training_yf import MyANN
from tensorflow import keras
from keras.layers import Dense, SimpleRNN, Flatten, LSTM, Dropout, GRU
from keras.callbacks import TensorBoard, ModelCheckpoint
from time import time
from datetime import datetime


def main():
    obj = MyANN(ticker="ADANIPORTS.NS", interval="1m")

    (X_train, Y_train), (X_test, Y_test) = obj.train_test_split(test_size=0.2)

    load_model = False
    check_on_test = False

    NUMBER_OF_EPOCHS = 1000
    BATCH_SIZE = 32
    SAFETY_FACTOR = 0.8
    LEARNING_RATE = 0.003

    print(X_train[0].shape)

    if load_model:
        model = keras.models.load_model(
            "models/model - 2023-08-20 15:05:47.670826.keras"
        )
        print("\nmodel loaded.\n")

    else:
        model = keras.Sequential()

        # model.add(
        #     SimpleRNN(
        #         9,
        #         input_shape=(X_train[0].shape),
        #         return_sequences=True,
        #         activation="relu",
        #     )
        # )

        model.add(
            Dense(
                50,
                input_shape=(X_train[0].shape),
                activation="relu",
            )
        )

        model.add(Flatten())

        model.add(
            Dense(
                30,
                activation="relu",
            )
        )
        model.add(
            Dense(
                1000,
                activation="relu",
            )
        )

        # model.add(
        #     Dense(
        #         300,
        #         activation="relu",
        #     )
        # )

        # model.add(Dropout(0.2))

        model.add(Dense(2))

        model.summary()

        print(model.output_shape)

        log_dir = "logs/"  # Directory where you want to store the TensorBoard logs
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        # terminal command: tensorboard --logdir=logs/
        # log directory: logs/train/

        optimizer = keras.optimizers.legacy.Adam(learning_rate=LEARNING_RATE)

        model.compile(
            optimizer=optimizer,
            loss="mean_squared_error",
            metrics=["accuracy", "mse", "mae"],
        )

        # filepath = (
        #     f"models/{datetime.now()} - " + "RNN_Final-{epoch:02d}-{accuracy:.3f}"
        # )
        # # unique file name that will include the epoch and the validation acc for that epoch
        # checkpoint = ModelCheckpoint(
        #     "models/{}.model".format(
        #         filepath, monitor="accuracy", verbose=1, save_best_only=True, mode="max"
        #     )
        # )  # saves only the best ones

        model.fit(
            X_train,
            Y_train,
            epochs=NUMBER_OF_EPOCHS,
            # batch_size=BATCH_SIZE,
            validation_data=(X_test, Y_test),
            callbacks=[tensorboard_callback],
            # callbacks=[tensorboard_callback, checkpoint],
        )

        model.save(f"models/model - {datetime.now()}.keras")

        print("\nmodel training done.\n")

    # if check_on_test:
    #     loss = model.evaluate(X_test, Y_test)

    #     win_percent = obj.custom_evaluate_full_envelope(
    #         model=model, X_test=X_test, Y_test=Y_test
    #     )
    # else:
    #     loss = model.evaluate(X_train, Y_train)

    #     win_percent = obj.custom_evaluate_full_envelope(
    #         model=model, X_test=X_train, Y_test=Y_train
    #     )

    if check_on_test:
        loss = model.evaluate(X_test, Y_test)

        win_percent = obj.custom_evaluate_safety_factor_2(
            model=model, X_test=X_test, Y_test=Y_test, safety_factor=SAFETY_FACTOR
        )
    else:
        loss = model.evaluate(X_train, Y_train)

        win_percent = obj.custom_evaluate_safety_factor_2(
            model=model, X_test=X_train, Y_test=Y_train, safety_factor=SAFETY_FACTOR
        )

    # print(f"\nloss: {loss}\n")
    # print(f"win_percent: {win_percent}")


if __name__ == "__main__":
    time_1 = time()
    main()
    print(f"\ntime taken = {round(time() - time_1, 2)} sec\n")
