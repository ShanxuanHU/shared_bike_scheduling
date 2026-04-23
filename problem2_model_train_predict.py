import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def train_model(train_df, target, features):

    X = train_df[features].copy()
    y = train_df[target].copy()

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=1
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
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    nonzero_mask = y_true != 0
    if np.any(nonzero_mask):
        mape = np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask]))
    else:
        mape = np.nan

    return mae, rmse, mape
