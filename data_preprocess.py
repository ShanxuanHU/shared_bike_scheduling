import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


BASE_DIR = Path(__file__).resolve().parent
STATION_FILE = BASE_DIR / "附件1_站点基础信息.csv"
TRIP_FILE = BASE_DIR / "附件2_每小时借还记录.csv"

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

FORECAST_FEATURES = [
    "hour_sin",
    "hour_cos",
    "is_weekday",
    "borrow_lag_24",
    "return_lag_24",
    "borrow_base",
    "return_base",
    "zone_0",
    "zone_1",
    "zone_2",
]


def load_raw_data():
    trip_df = pd.read_csv(TRIP_FILE)
    station_df = pd.read_csv(STATION_FILE)

    trip_df.columns = trip_df.columns.str.strip()
    station_df.columns = station_df.columns.str.strip()

    trip_df = trip_df.rename(
        columns={
            "日期": "date",
            "小时(0-23)": "hour",
            "站点编号": "station_id",
            "借出量": "borrow",
            "归还量": "return",
        }
    )
    station_df = station_df.rename(
        columns={
            "站点编号": "station_id",
            "经度": "lon",
            "纬度": "lat",
        }
    )

    trip_df["date"] = pd.to_datetime(trip_df["date"]).dt.normalize()
    trip_df["hour"] = pd.to_numeric(trip_df["hour"], errors="coerce").astype(int)
    trip_df["borrow"] = pd.to_numeric(trip_df["borrow"], errors="coerce")
    trip_df["return"] = pd.to_numeric(trip_df["return"], errors="coerce")
    trip_df = trip_df.dropna(subset=["borrow", "return"])
    trip_df["time"] = trip_df["date"] + pd.to_timedelta(trip_df["hour"], unit="h")

    merged = trip_df.merge(
        station_df[["station_id", "lon", "lat"]],
        on="station_id",
        how="left",
    )
    return merged


def add_time_features(df):
    df = df.copy()
    df["weekday"] = df["time"].dt.weekday
    df["is_weekday"] = (df["weekday"] < 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


def add_spatial_features(df):
    df = df.copy()
    station_coords = df[["station_id", "lon", "lat"]].drop_duplicates().reset_index(drop=True)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    station_coords["zone"] = kmeans.fit_predict(station_coords[["lon", "lat"]])

    df = df.merge(station_coords[["station_id", "zone"]], on="station_id", how="left")
    zone_dummies = pd.get_dummies(df["zone"], prefix="zone", dtype=int)
    df = pd.concat([df, zone_dummies], axis=1)

    for zone_col in ("zone_0", "zone_1", "zone_2"):
        if zone_col not in df.columns:
            df[zone_col] = 0

    return df


def add_history_features(df):
    df = df.copy().sort_values(["station_id", "time"]).reset_index(drop=True)
    df["borrow_lag_24"] = df.groupby("station_id")["borrow"].shift(24)
    df["return_lag_24"] = df.groupby("station_id")["return"].shift(24)
    return df


def add_train_only_baseline(full_df, train_mask):
    baseline = (
        full_df.loc[train_mask]
        .groupby(["station_id", "hour"])[["borrow", "return"]]
        .mean()
        .rename(columns={"borrow": "borrow_base", "return": "return_base"})
        .reset_index()
    )

    enriched = full_df.merge(baseline, on=["station_id", "hour"], how="left")
    return enriched


def build_model_datasets(target_date):
    target_date = pd.Timestamp(target_date).normalize()

    df = load_raw_data()
    df = add_time_features(df)
    df = add_spatial_features(df)
    df = add_history_features(df)

    train_mask = df["date"] < target_date
    test_mask = df["date"] == target_date

    if not train_mask.any():
        raise ValueError(f"目标日期 {target_date.date()} 之前没有可用训练数据。")
    if not test_mask.any():
        raise ValueError(f"目标日期 {target_date.date()} 在数据中不存在。")

    df = add_train_only_baseline(df, train_mask)
    df["borrow_res"] = df["borrow"] - df["borrow_base"]
    df["return_res"] = df["return"] - df["return_base"]

    required_cols = FORECAST_FEATURES + ["borrow", "return", "borrow_res", "return_res"]
    model_df = df.dropna(subset=required_cols).copy()

    train_df = model_df[model_df["date"] < target_date].copy()
    test_df = model_df[model_df["date"] == target_date].copy()

    if test_df.empty:
        raise ValueError(
            f"目标日期 {target_date.date()} 的特征构造失败，通常是因为缺少前一日同小时数据。"
        )

    return train_df, test_df, FORECAST_FEATURES
