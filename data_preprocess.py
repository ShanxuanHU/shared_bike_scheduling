import pandas as pd
import numpy as np

def load_data():
    # ===== 读取数据（用真实文件名）=====
    df = pd.read_csv('附件2_每小时借还记录.csv')
    station = pd.read_csv('附件1_站点基础信息.csv')

    # ===== 去除列名空格（防炸）=====
    df.columns = df.columns.str.strip()
    station.columns = station.columns.str.strip()

    # ===== 重命名列 =====
    df = df.rename(columns={
        '日期': 'date',
        '小时(0-23)': 'hour',
        '站点编号': 'station_id',
        '借出量': 'borrow',
        '归还量': 'return'
    })

    station = station.rename(columns={
        '站点编号': 'station_id',
        '经度': 'lon',
        '纬度': 'lat'
    })

    # ===== 构造时间列 =====
    df['time'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['hour'], unit='h')

    # ===== 合并站点信息 =====
    df = df.merge(station[['station_id','lon','lat']], on='station_id', how='left')

    return df


def add_time_features(df):
    df['weekday'] = df['time'].dt.weekday
    df['is_weekday'] = (df['weekday'] < 5).astype(int)

    # 周期编码
    df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
    df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)

    return df


def add_lag_features(df):
    df = df.sort_values(['station_id','time'])

    # lag特征（正确）
    for lag in [1,2,3,24]:
        df[f'borrow_lag_{lag}'] = df.groupby('station_id')['borrow'].shift(lag)
        df[f'return_lag_{lag}'] = df.groupby('station_id')['return'].shift(lag)

    # rolling（修正后的版本）
    df['borrow_roll3'] = df.groupby('station_id')['borrow'] \
        .transform(lambda x: x.shift(1).rolling(3).mean())

    df['return_roll3'] = df.groupby('station_id')['return'] \
        .transform(lambda x: x.shift(1).rolling(3).mean())

    return df


from sklearn.cluster import KMeans

def add_spatial_features(df):
    # 只对站点做一次聚类（避免重复计算）
    station_coords = df[['station_id','lon','lat']].drop_duplicates()

    kmeans = KMeans(n_clusters=3, random_state=42)
    station_coords['zone'] = kmeans.fit_predict(station_coords[['lon','lat']])

    # 合并回原数据
    df = df.merge(station_coords[['station_id','zone']], on='station_id', how='left')

    # one-hot编码
    zone_dummies = pd.get_dummies(df['zone'], prefix='zone')
    df = pd.concat([df, zone_dummies], axis=1)

    return df

def build_baseline(df):
    baseline = df.groupby(['station_id','hour'])[['borrow','return']].mean().reset_index()

    baseline = baseline.rename(columns={
        'borrow': 'borrow_base',
        'return': 'return_base'
    })

    df = df.merge(baseline, on=['station_id','hour'], how='left')

    df['borrow_res'] = df['borrow'] - df['borrow_base']
    df['return_res'] = df['return'] - df['return_base']

    return df


def preprocess_all():
    df = load_data()

    # 删除缺失
    df = df.dropna(subset=['borrow','return'])

    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_spatial_features(df)
    df = build_baseline(df)

    # 删除lag产生的NaN
    df = df.dropna()

    return df