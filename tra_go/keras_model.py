from tensorflow import keras
from keras.layers import Dense, Flatten, LSTM, Dropout
from keras import backend as K
import tensorflow as tf

# PYTHONPATH = /Users/bisane.s/my_files/my_codes/tra-go/.venv/bin/python

# keep total neurons below 2700 (700 * 3)

NUMBER_OF_NEURONS = 850
NUMBER_OF_LAYERS = 3
INITIAL_DROPOUT = 20

ERROR_AMPLIFICATION_FACTOR = 0


def get_untrained_model(X_train, y_type):
    model = keras.Sequential()

    model.add(
        LSTM(
            units=NUMBER_OF_NEURONS,
            input_shape=(X_train[0].shape),
            return_sequences=True,
            activation="relu",
        )
    )
    model.add(Dropout(INITIAL_DROPOUT / 100))

    for i in range(NUMBER_OF_LAYERS - 1):
        model.add(
            LSTM(
                units=NUMBER_OF_NEURONS,
                return_sequences=True,
                activation="relu",
            )
        )
        model.add(Dropout(pow(INITIAL_DROPOUT, 1 / (i + 2)) / 100))
        #  dropout value decreases in exponential fashion.

    if y_type == "hl":
        model.add(Flatten())

    if y_type in ["band", "hl"]:
        model.add(Dense(2))

    if y_type == "2_mods":
        model.add(Dense(1))

    model.summary()

    return model


def get_optimiser(learning_rate: float):
    return keras.optimizers.legacy.Adam(learning_rate=learning_rate)


def custom_loss_2_mods_high(y_true, y_pred):
    """
    Calculates a custom loss function for high predictions.

    Args:
        y_true (tensor): The true values.
        y_pred (tensor): The predicted values.

    Returns:
        tensor: The calculated loss value.

    Notes:
        - The predicted values (y_pred) should be higher than the true values (y_true).
        - The predicted values should approach the true values from above.

    """
    error = y_true - y_pred
    # y_pred should be higher that y_true
    # y_pred to approach to y_true from up

    negative_error = K.maximum(-error, 0)
    positive_error = K.maximum(error, 0)

    return K.sqrt(K.mean(K.square(error) + K.square(positive_error) * ERROR_AMPLIFICATION_FACTOR))


def custom_loss_2_mods_low(y_true, y_pred):
    """
    Calculates a custom loss function for a regression model with 2 modifications.

    Parameters:
        y_true (tensor): The true values of the target variable.
        y_pred (tensor): The predicted values of the target variable.

    Returns:
        tensor: The custom loss value.

    Notes:
        - The custom loss is calculated as the square root of the mean of the sum of squares of the difference between y_true and y_pred, and the square of the maximum of -error and 0 multiplied by the error amplification factor.
        - The error amplification factor determines the weight of the negative error in the loss calculation.
        - The y_pred values should be lower than the y_true values.
        - The y_pred values should approach the y_true values from below.
    """
    error = y_true - y_pred
    # y_pred should be lower that y_true
    # y_pred to approach to y_true from down

    negative_error = K.maximum(-error, 0)
    positive_error = K.maximum(error, 0)

    return K.sqrt(K.mean(K.square(error) + K.square(negative_error) * ERROR_AMPLIFICATION_FACTOR))


def custom_loss_band(y_true, y_pred):
    error = y_true - y_pred
    error_l = y_true[..., 0] - y_pred[..., 0]
    error_h = y_true[..., 1] - y_pred[..., 1]

    positive_error_h = K.maximum(error_h, 0)
    negative_error_l = K.maximum(-error_l, 0)

    error_amplified = (positive_error_h + negative_error_l) * ERROR_AMPLIFICATION_FACTOR
    error_amplified = K.expand_dims(error_amplified, axis=-1)  # Reshape error_amplified

    return K.sqrt(K.mean(K.square(error) + error_amplified))


def metric_rmse(y_true, y_pred):
    """
    Calculate the root mean squared error (RMSE) between the true values and the predicted values.

    Parameters:
        y_true (array-like): The true values.
        y_pred (array-like): The predicted values.

    Returns:
        float: The root mean squared error.
    """
    error = y_true - y_pred
    return K.sqrt(K.mean(K.square(error)))


class LossDifferenceCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir):
        super(LossDifferenceCallback, self).__init__()
        self.previous_loss = None
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(self.log_dir)

    def on_train_begin(self, logs=None):
        self.loss_difference_values = []

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get("loss")
        if self.previous_loss is not None:
            loss_difference = current_loss - self.previous_loss
            self.loss_difference_values.append(loss_difference)
            self.update_tensorboard(epoch, loss_difference)

        self.previous_loss = current_loss

    def update_tensorboard(self, epoch, loss_difference):
        with self.writer.as_default():
            tf.summary.scalar("Loss Difference", loss_difference, step=epoch)
