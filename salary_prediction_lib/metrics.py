from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def calculate_regression_metrics(y_true, y_pred):
    """
    Calculates common regression metrics: RMSE, MAE, R^2, MAPE, SMAPE.

    Parameters:
    - y_true (array-like): True target values.
    - y_pred (array-like): Predicted target values.

    Returns:
    - dict: A dictionary containing the calculated metrics.
    """
    # RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # MAE
    mae = mean_absolute_error(y_true, y_pred)

    # R^2 Score
    r2 = r2_score(y_true, y_pred)

    # MAPE
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # SMAPE
    smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R^2": r2,
        "MAPE (%)": mape,
        "SMAPE (%)": smape
    }
