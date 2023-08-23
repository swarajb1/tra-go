from training_yf import MyANN
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import TensorBoard
from time import time
from datetime import datetime


def main():
    obj = MyANN(ticker="ADANIPORTS.NS", interval="1m")

    X_train, Y_train, X_test, Y_test = obj.train_test_split(test_size=0.2)

    load_model = True
    check_on_test = False

    if load_model:
        model = keras.models.load_model(
            "models/model - 2023-08-20 15:05:47.670826.keras"
        )
        print("\nmodel loaded.\n")

    else:
        model = keras.Sequential(
            [
                keras.layers.Dense(
                    300,
                    input_shape=(len(X_train[0]), len(X_train[0][0])),
                    activation="relu",
                    kernel_initializer="random_normal",
                ),
                keras.layers.Dense(
                    300,
                    activation="relu",
                    kernel_initializer="random_normal",
                ),
                # keras.layers.Dense(
                #     1800,
                #     activation="relu",
                #     kernel_initializer="random_normal",
                # ),
                # keras.layers.Dropout(0.5),
                keras.layers.Dense(2),
            ]
        )

        print(model.summary())

        log_dir = "logs/"  # Directory where you want to store the TensorBoard logs
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        # terminal command: tensorboard --logdir=logs/

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.003)

        model.compile(
            optimizer=optimizer,
            loss="mean_absolute_error",
            metrics=["accuracy", "mse", "mae"],
        )

        model.fit(
            X_train,
            Y_train,
            epochs=100,
            # batch_size=9,
            callbacks=[tensorboard_callback],
        )

        print("\nmodel training done.\n")

        model.save(f"models/model - {datetime.now()}.keras")

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

        win_percent = obj.custom_evaluate_safety_factor(
            model=model, X_test=X_test, Y_test=Y_test, safety_factor=0.8
        )
    else:
        loss = model.evaluate(X_train, Y_train)

        win_percent = obj.custom_evaluate_safety_factor(
            model=model, X_test=X_train, Y_test=Y_train, safety_factor=0.8
        )

    print(f"\nloss: {loss}\n")
    print(f"win_percent: {win_percent}")


if __name__ == "__main__":
    time_1 = time()
    main()
    print(f"\ntime taken = {round(time() - time_1, 2)} sec\n")
