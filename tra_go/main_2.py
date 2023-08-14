from training_yf import MyANN
from tensorflow import keras
from keras.callbacks import TensorBoard
from time import time

from datetime import datetime


def main():
    obj = MyANN(ticker="ADANIPORTS.NS", interval="1m")

    X_train, Y_train, X_test, Y_test = obj.train_test_split(test_size=0.2)

    model = keras.Sequential(
        [
            keras.layers.Dense(
                264,
                input_shape=(len(X_train[0]), len(X_train[0][0])),
                activation="relu",
            ),
            keras.layers.Dropout(0.5),
            # keras.layers.Dense(300, activation="relu"),
            # keras.layers.Dropout(0.5),
            keras.layers.Dense(2),
        ]
    )

    print(model.summary())

    log_dir = "logs/"  # Directory where you want to store the TensorBoard logs
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    # terminal command: tensorboard --logdir=logs/

    model.compile(
        optimizer="adam",
        loss="mean_absolute_error",
        metrics=["accuracy", "mse", "mae"],
    )

    model.fit(X_train, Y_train, epochs=1000, callbacks=[tensorboard_callback])

    print("\nmodel training done.\n")

    model.save(f"models/model - {datetime.now()}.keras")

    check_on_test = False

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