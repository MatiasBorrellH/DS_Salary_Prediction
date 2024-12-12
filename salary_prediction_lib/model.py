import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import numpy as np

def train_and_predict_lgb(X_train, y_train, X_test, y_test, params=None, num_boost_round=100):
    """
    Trains a LightGBM model and makes predictions on test data.

    Parameters:
    - X_train (pd.DataFrame or np.array): Training features.
    - y_train (pd.Series or np.array): Training target.
    - X_test (pd.DataFrame or np.array): Test features.
    - y_test (pd.Series or np.array): Test target.
    - params (dict): Dictionary of LightGBM hyperparameters.
    - num_boost_round (int): Number of boosting iterations.

    Returns:
    - model (lgb.Booster): Trained LightGBM model.
    - y_pred (np.array): Predictions on the test set.
    - rmse (float): Root Mean Squared Error of the predictions.
    """
    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    params = {
    'objective': 'regression', 
    'metric': 'rmse',         
    'boosting_type': 'gbdt',   
    'learning_rate': 0.1,
    'num_leaves': 31,
    'max_depth': -1,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'seed': 42
}

    # Train the model
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, test_data],
        num_boost_round=num_boost_round,
    )

    # Predict on test data
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    
    return y_pred

