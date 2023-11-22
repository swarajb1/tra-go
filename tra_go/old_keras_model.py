from tensorflow import keras
from keras.layers import Dense, Flatten, LSTM, Dropout
from keras import backend as K
import tensorflow as tf


ERROR_AMPLIFICATION_FACTOR = 0.6


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


def metric_band_hl_correction(y_true, y_pred):
    hl_correction_error_val = y_pred[..., 1] - y_pred[..., 0]
    error_hl_correction = K.sqrt(K.mean(K.square(K.maximum(-hl_correction_error_val, 0))))

    # at starting : error_rsme=1, error_inside_range=1, error_avg=1, error_hl_correction=0
    return error_hl_correction


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
