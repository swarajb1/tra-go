from tensorflow import keras
from keras.layers import Dense, Flatten, LSTM, Dropout
from keras import backend as K
import tensorflow as tf


# PYTHONPATH = /Users/bisane.s/my_files/my_codes/tra-go/.venv/bin/python

# keep total neurons below 2700 (700 * 3)

NUMBER_OF_NEURONS = 512
NUMBER_OF_LAYERS = 3
INITIAL_DROPOUT = 0


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


class CustomLossCallback(tf.keras.callbacks.Callback):
    def __init__(self, first_loss, second_loss, switch_epoch):
        super().__init__()
        self.first_loss = first_loss
        self.second_loss = second_loss
        self.switch_epoch = switch_epoch

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.switch_epoch:
            self.model.loss = self.first_loss
        else:
            self.model.loss = self.second_loss


def metric_rmse(y_true, y_pred):
    # Calculate the root mean squared error (RMSE) between the true values and the predicted values.

    error = y_true - y_pred

    return K.sqrt(K.mean(K.square(error)))


def metric_band_hl_correction_2(y_true, y_pred):
    # band height cannot be negative
    bandwidth_array = y_pred[..., 1]

    error_hl_correction = K.sqrt(K.mean(K.square(K.maximum(-bandwidth_array, 0))))

    return error_hl_correction


def metric_band_hl_wrongs_percent(y_true, y_pred):
    bandwidth_array = y_pred[..., 1]

    negative_hl_count = tf.reduce_sum(tf.cast(tf.less(bandwidth_array, 0), tf.float32))

    total_count = K.cast(K.shape(bandwidth_array)[1], dtype=K.floatx()) * K.cast(
        K.shape(bandwidth_array)[0], dtype=K.floatx()
    )

    return negative_hl_count / total_count * 100


def metric_band_height(y_true, y_pred):
    # band height to approach true band height
    error = y_true[..., 1] - y_pred[..., 1]

    return K.sqrt(K.mean(K.square(error)))


def metric_band_height_percent(y_true, y_pred):
    return metric_band_height(y_true, y_pred) / K.mean(y_true[..., 1]) * 100


def metric_band_average(y_true, y_pred):
    # average should approach true average
    error = y_true[..., 0] - y_pred[..., 0]

    error_avg = K.sqrt(K.mean(K.square(error)))

    return error_avg


def metric_band_average_percent(y_true, y_pred):
    return metric_band_average(y_true, y_pred) / K.mean(y_true[..., 0]) * 100


def support_new_idea_1(y_true, y_pred):
    min_pred = K.min(y_pred[:, :, 0] - y_pred[:, :, 1] / 2, axis=1)
    max_pred = K.max(y_pred[:, :, 0] + y_pred[:, :, 1] / 2, axis=1)

    min_true = K.min(y_true[:, :, 0] - y_true[:, :, 1] / 2, axis=1)
    max_true = K.max(y_true[:, :, 0] + y_true[:, :, 1] / 2, axis=1)

    wins = K.all(
        [
            max_true >= max_pred,
            max_pred >= min_pred,
            min_pred >= min_true,
        ],
        axis=0,
    )

    return min_pred, max_pred, min_true, max_true, wins


def support_new_idea_2(min_pred, max_pred, min_true, max_true, wins):
    cond_1 = K.all([max_true >= max_pred], axis=0)
    cond_2 = K.all([max_pred >= min_pred], axis=0)
    cond_3 = K.all([min_pred >= min_true], axis=0)

    z_1 = K.mean((1 - K.cast(cond_1, dtype=K.floatx())) * K.abs(max_true - max_pred))

    z_2 = K.mean((1 - K.cast(cond_2, dtype=K.floatx())) * K.abs(max_pred - min_pred))

    z_3 = K.mean((1 - K.cast(cond_3, dtype=K.floatx())) * K.abs(min_pred - min_true))

    win_amt_true = K.mean((1 - K.cast(wins, dtype=K.floatx())) * K.abs(max_true - min_true))

    return z_1, z_2, z_3, win_amt_true


def metric_new_idea_2(y_true, y_pred):
    min_pred, max_pred, min_true, max_true, wins = support_new_idea_1(y_true, y_pred)

    z_1, z_2, z_3, win_amt_true = support_new_idea_2(min_pred, max_pred, min_true, max_true, wins)

    pred_capture = K.sum(((max_pred / min_pred - 1)) * K.cast(wins, dtype=K.floatx()))

    total_capture_possible = K.sum((max_true / min_true - 1))

    return (
        metric_band_average(y_true, y_pred) * 3
        + (metric_band_height(y_true, y_pred) + metric_band_hl_correction_2(y_true, y_pred)) * 100
        + (metric_band_height_percent(y_true, y_pred) + metric_band_hl_wrongs_percent(y_true, y_pred)) / 10
        + (
            (z_1 + z_2 + z_3 + win_amt_true)
            + (1 - pred_capture / total_capture_possible) * K.mean(K.abs(max_true - min_true))
        )
        * 100
    )


def metric_loss_comp_2(y_true, y_pred):
    min_pred, max_pred, min_true, max_true, wins = support_new_idea_1(y_true, y_pred)

    z_1, z_2, z_3, win_amt_true = support_new_idea_2(min_pred, max_pred, min_true, max_true, wins)

    return z_1 + z_2 + z_3 + win_amt_true


def metric_pred_capture_percent(y_true, y_pred):
    min_pred, max_pred, min_true, max_true, wins = support_new_idea_1(y_true, y_pred)

    pred_capture = K.sum(((max_pred / min_pred - 1)) * K.cast(wins, dtype=K.floatx()))

    total_capture_possible = K.sum((max_true / min_true - 1))

    return pred_capture / total_capture_possible * 100


def metric_win_percent(y_true, y_pred):
    min_pred, max_pred, min_true, max_true, wins = support_new_idea_1(y_true, y_pred)

    win_percent = K.mean((K.cast(wins, dtype=K.floatx())))

    return win_percent * 100
