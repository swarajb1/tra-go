from tensorflow import keras
from keras.layers import Dense, Flatten, LSTM, Dropout
from keras import backend as K
import tensorflow as tf

# PYTHONPATH = /Users/bisane.s/my_files/my_codes/tra-go/.venv/bin/python

# keep total neurons below 2700 (700 * 3)

NUMBER_OF_NEURONS = 128
NUMBER_OF_LAYERS = 3
INITIAL_DROPOUT = 0

ERROR_AMPLIFICATION_FACTOR = 0.6


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

    if y_type in ["band", "hl", "band_2"]:
        model.add(Dense(NUMBER_OF_NEURONS))
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

    positive_error_h = K.square(K.maximum(error_h, 0))
    negative_error_l = K.square(K.maximum(-error_l, 0))

    error_amplified = (positive_error_h + negative_error_l) * ERROR_AMPLIFICATION_FACTOR
    error_amplified = K.expand_dims(error_amplified, axis=-1)  # Reshape error_amplified

    return K.sqrt(K.mean(K.square(error) + error_amplified))


def custom_loss_band_2(y_true, y_pred):
    # list of all approaches:
    # approach pred_average to true_average = average_error
    # pred_high to approach true_high from below = h_more_than_ture_error
    # pred_low to approach true_low from below = l_less_than_ture_error
    # making sure that pred_high is always greater than pred_low = error_hl_correction

    return (
        metric_rmse(y_true, y_pred)
        + metric_band_inside_range(y_true, y_pred) * 4
        + metric_band_error_average(y_true, y_pred) * 5
        + metric_band_hl_correction(y_true, y_pred) * 5
    ) / 10


def metric_band_inside_range(y_true, y_pred):
    average_true = (y_true[..., 0] + y_true[..., 1]) / 2

    # approach - :
    h_error_1 = y_true[..., 1] - y_pred[..., 1]
    l_error_1 = y_true[..., 0] - y_pred[..., 0]

    # h should be less than true_h
    h_more_than_ture_error = K.sqrt(K.mean(K.square(K.maximum(-h_error_1, 0))))
    # l should be more than true_l
    l_less_than_ture_error = K.sqrt(K.mean(K.square(K.maximum(l_error_1, 0))))

    h_error_2 = y_true[..., 1] - average_true
    l_error_2 = y_true[..., 0] - average_true

    # h should be more than true_avg
    h_less_than_ture_avg_error = K.sqrt(K.mean(K.square(K.maximum(-h_error_2, 0))))
    # l should be less than true_avg
    l_more_than_ture_avg_error = K.sqrt(K.mean(K.square(K.maximum(l_error_2, 0))))

    # approach - :
    error_inside_range = (
        h_more_than_ture_error + l_less_than_ture_error + h_less_than_ture_avg_error + l_more_than_ture_avg_error
    )

    error_inside_range_2 = (
        h_more_than_ture_error
        + l_less_than_ture_error
        + h_less_than_ture_avg_error * 4
        + l_more_than_ture_avg_error * 4
    ) / 10

    return error_inside_range_2


def metric_band_error_average(y_true, y_pred):
    # list of all approaches:
    # approach pred_average to true_average = average_error
    # pred_high to approach true_high from below = h_more_than_ture_error
    # pred_low to approach true_low from below = l_less_than_ture_error
    # making sure that pred_high is always greater than pred_low = error_hl_correction

    # approach - :
    average_true = (y_true[..., 0] + y_true[..., 1]) / 2
    average_pred = (y_pred[..., 0] + y_pred[..., 1]) / 2
    average_error = average_true - average_pred

    error_avg = K.sqrt(K.mean(K.square(average_error)))

    return error_avg


def metric_band_hl_correction(y_true, y_pred):
    hl_correction_error_val = y_pred[..., 1] - y_pred[..., 0]
    error_hl_correction = K.sqrt(K.mean(K.square(K.maximum(-hl_correction_error_val, 0))))

    # at starting : error_rsme=1, error_inside_range=1, error_avg=1, error_hl_correction=0
    return error_hl_correction


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


def custom_loss_band_2_2(y_true, y_pred):
    # list of all approaches:
    # approach pred_average to true_average = average_error
    # band to be inside true band
    # band cannot be negative

    return (
        metric_rmse(y_true, y_pred)
        + metric_band_inside_range_2(y_true, y_pred) * 4
        + metric_band_error_average_2(y_true, y_pred) * 2
        + metric_band_hl_correction_2(y_true, y_pred) * 5
    ) / 6


def metric_band_inside_range_2(y_true, y_pred):
    # band height cannot be more than true band height
    error = y_true[..., 1] - y_pred[..., 1]

    return K.sqrt(K.mean(K.square(K.maximum(-error, 0))))


def metric_band_error_average_2(y_true, y_pred):
    # average should approach true average
    average_error = y_true[..., 0] - y_pred[..., 0]

    error_avg = K.sqrt(K.mean(K.square(average_error)))

    return error_avg


def metric_band_hl_correction_2(y_true, y_pred):
    # band height cannot be negative
    hl_correction_error_val = y_pred[..., 1]

    error_hl_correction = K.sqrt(K.mean(K.square(K.maximum(-hl_correction_error_val, 0))))

    return error_hl_correction


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
