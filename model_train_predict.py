import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def train_model(train_df, target):

    features = [
        'hour_sin','hour_cos','is_weekday',
        'borrow_lag_1','borrow_lag_2','borrow_lag_3','borrow_lag_24',
        'return_lag_1','return_lag_2','return_lag_3','return_lag_24',
        'borrow_roll3','return_roll3',
        'zone_0','zone_1','zone_2'
    ]

    X = train_df[features]
    y = train_df[target]

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8
    )

    model.fit(X, y)

    return model, features


def predict(test_df, model, features, base_col):
    X = test_df[features]
    pred_res = model.predict(X)

    pred = pred_res + test_df[base_col]

    return pred


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred)/(y_true+1e-5)))

    return mae, rmse, mape