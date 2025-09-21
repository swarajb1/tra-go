import tensorflow as tf


def get_optimiser(learning_rate: float):
    return tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)


def metric_rmse(y_true, y_pred):
    # Calculate the root mean squared error (RMSE)

    error = y_true - y_pred

    return tf.sqrt(tf.reduce_mean(tf.square(error)))


def metric_abs(y_true, y_pred):
    # Calculate the absolute mean error (MAE)

    error = y_true - y_pred

    return tf.reduce_mean(tf.abs(error))


def metric_abs_percent(y_true, y_pred):
    error = y_true - y_pred

    return tf.reduce_mean(tf.abs(error)) / tf.reduce_mean(tf.abs(y_true)) * 100


def metric_rmse_percent(y_true, y_pred):
    error = y_true - y_pred

    return tf.sqrt(tf.reduce_mean(tf.square(error))) / tf.reduce_mean(tf.abs(y_true)) * 100
