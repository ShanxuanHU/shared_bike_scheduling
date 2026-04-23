"""
问题3配置文件
加载所有数据，计算距离矩阵，设置参数
"""

from math import radians, sin, cos, sqrt, asin
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
STATION_FILE = BASE_DIR / "附件1_站点基础信息.csv"
TRIP_FILE = BASE_DIR / "附件2_每小时借还记录.csv"
PARAM_FILE = BASE_DIR / "附件3_调度成本参数.csv"
PROBLEM2_OUTPUT_DIR = BASE_DIR / "outputs" / "problem2"
PROBLEM3_OUTPUT_DIR = BASE_DIR / "outputs" / "problem3"


def clean_columns(df):
    """
    清理 DataFrame 列名。
    """
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace("\ufeff", "", regex=False)
    df.columns = df.columns.str.replace("﻿", "", regex=False)
    return df


def convert_to_numeric(value):
    """
    将字符串转换为数字，处理中文单位。
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        value = value.replace("元/公里", "").replace("元/辆", "").replace("元/(桩·小时)", "")
        value = value.replace("公里/小时", "").replace("辆/次", "").replace("%", "")
        value = value.strip()
        try:
            return float(value)
        except ValueError:
            return 0.0
    return float(value)


def haversine_distance(lon1, lat1, lon2, lat2):
    """
    计算两个经纬度点之间的直线距离（单位：公里）。
    """
    radius = 6371
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return radius * c


def load_data():
    """
    加载站点与成本参数。
    """
    stations = clean_columns(pd.read_csv(STATION_FILE))
    stations["总桩位数"] = pd.to_numeric(stations["总桩位数"], errors="coerce")
    stations["当前库存量"] = pd.to_numeric(stations["当前库存量"], errors="coerce")

    params = clean_columns(pd.read_csv(PARAM_FILE))
    param_dict = {}
    for _, row in params.iterrows():
        param_dict[row["参数名称"]] = convert_to_numeric(row["数值"])

    return stations, param_dict


def compute_distance_matrix(stations, speed):
    """
    计算站点间距离矩阵与调度时间矩阵。
    """
    n = len(stations)
    dist_matrix = np.zeros((n, n))
    time_matrix = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            lon1, lat1 = stations.iloc[i][["经度", "纬度"]]
            lon2, lat2 = stations.iloc[j][["经度", "纬度"]]
            dist = haversine_distance(lon1, lat1, lon2, lat2)
            dist_matrix[i, j] = dist
            time_matrix[i, j] = int(np.floor(dist / speed + 0.5))

    return dist_matrix, time_matrix


def normalize_prediction_columns(pred_df):
    pred_df = clean_columns(pred_df).rename(
        columns={
            "日期": "date",
            "小时(0-23)": "hour",
        }
    ).copy()
    pred_df["hour"] = pd.to_numeric(pred_df["hour"], errors="coerce").astype(int)
    pred_df["借出量"] = pd.to_numeric(pred_df["借出量"], errors="coerce").fillna(0)
    pred_df["归还量"] = pd.to_numeric(pred_df["归还量"], errors="coerce").fillna(0)
    return pred_df


def load_demand_predictions(file_path=None):
    """
    加载问题3需求预测数据。
    """
    if file_path:
        explicit_path = Path(file_path)
        if not explicit_path.is_absolute():
            explicit_path = BASE_DIR / explicit_path
        if explicit_path.exists():
            pred_df = normalize_prediction_columns(pd.read_csv(explicit_path))
            print(f"  [OK] 成功加载指定预测文件: {explicit_path}")
            return pred_df

    candidate_files = sorted(PROBLEM2_OUTPUT_DIR.glob("predictions_*.csv"))
    if not candidate_files:
        candidate_files = sorted(BASE_DIR.glob("predictions_*.csv"))

    if candidate_files:
        latest_file = candidate_files[-1]
        pred_df = normalize_prediction_columns(pd.read_csv(latest_file))
        print(f"  [OK] 找到预测文件: {latest_file}")
        print(f"    - 数据行数: {len(pred_df)}")
        print(f"    - 覆盖站点: {pred_df['站点编号'].nunique()}")
        return pred_df

    print("  未找到预测文件，使用实际数据代替")
    df = clean_columns(pd.read_csv(TRIP_FILE))
    test_data = df[df["日期"] == "2025-04-09"].copy()
    if len(test_data) == 0:
        print("  警告: 未找到 2025-04-09 的数据，使用 2025-04-07 数据代替")
        test_data = df[df["日期"] == "2025-04-07"].copy()

    test_data = test_data[["日期", "小时(0-23)", "站点编号", "借出量", "归还量"]].copy()
    test_data = normalize_prediction_columns(test_data)
    print(f"  已加载 {len(test_data)} 条需求记录")
    print(f"  时间跨度: {test_data['hour'].min():.0f}-{test_data['hour'].max():.0f} 小时")
    print(f"  站点数: {test_data['站点编号'].nunique()} 个")
    return test_data


def get_station_mapping(stations):
    """
    创建站点编号到索引的映射。
    """
    station_to_idx = {row["站点编号"]: idx for idx, row in stations.iterrows()}
    idx_to_station = {idx: row["站点编号"] for idx, row in stations.iterrows()}
    return station_to_idx, idx_to_station


def get_config():
    """
    返回完整配置。
    """
    stations, params = load_data()
    speed = float(params.get("调度车速", 40))
    dist_matrix, time_matrix = compute_distance_matrix(stations, speed)
    station_to_idx, idx_to_station = get_station_mapping(stations)

    config = {
        "stations": stations,
        "n_stations": len(stations),
        "station_to_idx": station_to_idx,
        "idx_to_station": idx_to_station,
        "dist_matrix": dist_matrix,
        "time_matrix": time_matrix,
        "capacity": stations["总桩位数"].values.astype(float),
        "init_inventory": stations["当前库存量"].values.astype(float),
        "safe_inventory": stations["总桩位数"].values.astype(float) * 0.5,
        "c_km": float(params.get("每公里运输成本", 3.5)),
        "c_move": float(params.get("每辆单次搬运成本", 2.0)),
        "alpha": float(params.get("满桩惩罚系数", 8.0)),
        "beta": float(params.get("空桩惩罚系数", 5.0)),
        "truck_capacity": int(float(params.get("货车最大载车量", 15))),
        "speed": speed,
        "time_window_start": 6,
        "time_window_end": 22,
        "future_horizon": 2,
        "future_lambda": 0.5,
        "output_dir": PROBLEM3_OUTPUT_DIR,
    }

    return config
