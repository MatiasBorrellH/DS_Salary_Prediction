import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import random

def random_search_lgb(X_train, y_train, X_test, y_test, n_iter=20, seed=42):
    """
    Performs random search for LightGBM hyperparameter tuning.

    Parameters:
    - X_train, y_train: Training features and labels.
    - X_test, y_test: Testing features and labels.
    - n_iter: Number of random combinations to try.
    - seed: Random seed for reproducibility.

    Returns:
    - best_params: Best hyperparameters found.
    - best_rmse: RMSE of the best model.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Parameter grid
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'num_leaves': [15, 31, 63, 127],
        'max_depth': [-1, 5, 10, 15],
        'feature_fraction': [0.6, 0.7, 0.8, 0.9],
        'bagging_fraction': [0.6, 0.7, 0.8, 0.9],
        'bagging_freq': [1, 5, 10],
        'lambda_l1': [0.0, 0.1, 0.5, 1.0],
        'lambda_l2': [0.0, 0.1, 0.5, 1.0]
    }

    # Random search
    best_rmse = float('inf')
    best_params = None

    for i in range(n_iter):
        # Sample random parameters
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'seed': seed,
            'learning_rate': random.choice(param_grid['learning_rate']),
            'num_leaves': random.choice(param_grid['num_leaves']),
            'max_depth': random.choice(param_grid['max_depth']),
            'feature_fraction': random.choice(param_grid['feature_fraction']),
            'bagging_fraction': random.choice(param_grid['bagging_fraction']),
            'bagging_freq': random.choice(param_grid['bagging_freq']),
            'lambda_l1': random.choice(param_grid['lambda_l1']),
            'lambda_l2': random.choice(param_grid['lambda_l2']),
        }

        # Train model
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        model = lgb.train(
            params,
            train_data,
            valid_sets=[test_data],
            num_boost_round=100,
        )

        # Evaluate model
        y_pred = model.predict(X_test, num_iteration=model.best_iteration)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f"Iteration {i+1}/{n_iter}: RMSE = {rmse:.4f}, Params = {params}")

        # Update best parameters if this model is better
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params

    print("\nBest Parameters:", best_params)
    print("Best RMSE:", best_rmse)
    return best_params, best_rmse