"""
问题3配置文件
加载所有数据，计算距离矩阵，设置参数
"""

import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, asin


def clean_columns(df):
    """
    清理DataFrame的列名：去除空格、BOM头、统一格式
    """
    df.columns = df.columns.str.strip()  # 去除首尾空格
    df.columns = df.columns.str.replace('\ufeff', '')  # 去除BOM头
    df.columns = df.columns.str.replace('﻿', '')  # 去除其他隐藏字符
    return df


def convert_to_numeric(value):
    """
    将字符串转换为数字，处理中文单位
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # 去除可能的中文单位
        value = value.replace('元/公里', '').replace('元/辆',
                                                  '').replace('元/(桩·小时)', '')
        value = value.replace('公里/小时', '').replace('辆/次', '').replace('%', '')
        value = value.strip()
        try:
            return float(value)
        except:
            return 0.0
    return float(value)


def haversine_distance(lon1, lat1, lon2, lat2):
    """
    计算两个经纬度点之间的直线距离（单位：公里）
    使用Haversine公式
    """
    R = 6371  # 地球半径（公里）

    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))

    return R * c


def load_data():
    """
    加载所有数据文件
    """
    # 加载站点信息
    stations = pd.read_csv('附件1_站点基础信息.csv')
    stations = clean_columns(stations)

    # 确保数值列为数字类型
    stations['总桩位数'] = pd.to_numeric(stations['总桩位数'], errors='coerce')
    stations['当前库存量'] = pd.to_numeric(stations['当前库存量'], errors='coerce')

    # 加载调度参数
    params = pd.read_csv('附件3_调度成本参数.csv')
    params = clean_columns(params)

    # 提取参数并转换为数值
    param_dict = {}
    for _, row in params.iterrows():
        param_name = row['参数名称']
        param_value = convert_to_numeric(row['数值'])
        param_dict[param_name] = param_value

    return stations, param_dict


def compute_distance_matrix(stations):
    """
    计算站点间的距离矩阵（30x30）
    返回：距离矩阵(公里)，调度时间矩阵(小时，四舍五入取整)
    """
    n = len(stations)
    dist_matrix = np.zeros((n, n))
    time_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                dist_matrix[i, j] = 0
                time_matrix[i, j] = 0
            else:
                lon1, lat1 = stations.iloc[i]['经度'], stations.iloc[i]['纬度']
                lon2, lat2 = stations.iloc[j]['经度'], stations.iloc[j]['纬度']
                dist = haversine_distance(lon1, lat1, lon2, lat2)
                dist_matrix[i, j] = dist
                # 调度时间 = 距离 / 速度(40km/h)，四舍五入取整
                time_matrix[i, j] = round(dist / 40)

    return dist_matrix, time_matrix


def load_demand_predictions(file_path=None):
    """
    加载问题3的预测需求数据

    优先级：
    1. 如果指定file_path，使用指定文件
    2. 否则自动寻找 predictions_*.csv (工作日预测文件)
    3. 如果没有，使用附件2的实际数据（2025-04-07）
    """
    import os
    import glob

    # 如果指定了文件路径
    if file_path and os.path.exists(file_path):
        pred_df = pd.read_csv(file_path)
        pred_df = clean_columns(pred_df)
        print(f"  ✓ 成功加载指定预测文件: {file_path}")
        return pred_df

    # 自动寻找 predictions_*.csv 文件（工作日预测）
    pred_files = glob.glob('predictions_*.csv')
    if pred_files:
        file_path = pred_files[0]  # 使用找到的第一个预测文件
        print(f"  ✓ 找到预测文件: {file_path}")
        pred_df = pd.read_csv(file_path)
        pred_df = clean_columns(pred_df)

        # 重命名列名以匹配 parse_demand 的期望
        pred_df = pred_df.rename(columns={
            '日期': 'date',
            '小时(0-23)': 'hour',
            '站点编号': '站点编号',
            '借出量': '借出量',
            '归还量': '归还量'
        })

        print(f"    - 数据行数: {len(pred_df)}")
        print(f"    - 覆盖站点: {pred_df['站点编号'].nunique()}")
        return pred_df

    # 如果没有预测文件，使用实际数据
    print(f"  未找到预测文件，使用实际数据代替")
    df = pd.read_csv('附件2_每小时借还记录.csv')
    df = clean_columns(df)

    # 选择工作日数据（2025-04-09 周三）
    test_data = df[df['日期'] == '2025-04-09'].copy()

    if len(test_data) == 0:
        print("  警告: 未找到2025-04-09的数据，使用2025-04-07的数据代替")
        test_data = df[df['日期'] == '2025-04-07'].copy()

    # 确保列名正确 - 使用原始列名
    test_data = test_data[['日期', '小时(0-23)', '站点编号', '借出量', '归还量']].copy()

    # 重命名为统一格式
    test_data = test_data.rename(columns={
        '日期': 'date',
        '小时(0-23)': 'hour',
        '站点编号': '站点编号',
        '借出量': '借出量',
        '归还量': '归还量'
    })

    # 确保数值列为数字类型
    test_data['hour'] = pd.to_numeric(
        test_data['hour'], errors='coerce').astype(int)
    test_data['借出量'] = pd.to_numeric(
        test_data['借出量'], errors='coerce').fillna(0)
    test_data['归还量'] = pd.to_numeric(
        test_data['归还量'], errors='coerce').fillna(0)

    print(f"  已加载 {len(test_data)} 条需求记录")
    print(
        f"  时间跨度: {test_data['hour'].min():.0f}-{test_data['hour'].max():.0f} 小时")
    print(f"  站点数: {test_data['站点编号'].nunique()} 个")

    return test_data


def get_station_mapping(stations):
    """
    创建站点编号到索引的映射
    """
    station_to_idx = {row['站点编号']: idx for idx, row in stations.iterrows()}
    idx_to_station = {idx: row['站点编号'] for idx, row in stations.iterrows()}
    return station_to_idx, idx_to_station


def get_config():
    """
    返回完整配置
    """
    stations, params = load_data()
    dist_matrix, time_matrix = compute_distance_matrix(stations)
    station_to_idx, idx_to_station = get_station_mapping(stations)

    # 确保所有参数都是数字类型
    config = {
        'stations': stations,
        'n_stations': len(stations),
        'station_to_idx': station_to_idx,
        'idx_to_station': idx_to_station,
        'dist_matrix': dist_matrix,
        'time_matrix': time_matrix,
        'capacity': stations['总桩位数'].values.astype(float),  # C_i
        'init_inventory': stations['当前库存量'].values.astype(float),  # S_i0
        # R_i = 0.5 * C_i
        'safe_inventory': stations['总桩位数'].values.astype(float) * 0.5,
        'c_km': float(params.get('每公里运输成本', 3.5)),
        'c_move': float(params.get('每辆单次搬运成本', 2.0)),
        'alpha': float(params.get('满桩惩罚系数', 8.0)),  # 满桩惩罚系数
        'beta': float(params.get('空桩惩罚系数', 5.0)),   # 空桩惩罚系数
        'truck_capacity': int(float(params.get('货车最大载车量', 15))),
        'speed': float(params.get('调度车速', 40)),
        'time_window_start': 6,   # 调度开始时间（6:00）
        'time_window_end': 22,    # 调度结束时间（22:00）
        'future_horizon': 2,      # 未来预测视野（小时）
        'future_lambda': 0.5,     # 未来需求权重
    }

    return config
