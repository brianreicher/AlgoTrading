from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def validate(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    mae_past = mean_absolute_error(actual, predicted)
    mse_past = mean_squared_error(actual, predicted)
    rmse_past = np.sqrt(mse_past)
    r2_past = r2_score(actual, predicted)
    mape = mean_absolute_percentage_error(actual, predicted)
    print("Metrics for Predicted Past YTD:")
    print("MAE:", mae_past)
    print("MSE:", mse_past)
    print("RMSE:", rmse_past)
    print("Mean Absolute Percentage Error:", mape)
    print("R-squared:", r2_past)