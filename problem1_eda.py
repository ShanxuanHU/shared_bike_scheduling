import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 数据加载（附件3好像用不上）
stations = pd.read_csv('附件1_站点基础信息.csv', encoding='utf-8')
df = pd.read_csv('附件2_每小时借还记录.csv', encoding='utf-8')

# 日期处理
df['日期'] = pd.to_datetime(df['日期'])
df['星期'] = df['日期'].dt.weekday  # 0=周一 6=周日
df['是否为工作日'] = df['星期'].apply(lambda x: 1 if x < 5 else 0)

# 新增净归还量
df['净归还量'] = df['归还量'] - df['借出量']

# 合并站点信息
df = df.merge(stations[['站点编号', '站点名称', '经度', '纬度', '总桩位数']],
              on='站点编号', how='left')

'''
区分站点的信息：'站点编号', '站点名称', '经度', '纬度', '总桩位数'
区分工作日与休息日的信息：'是否为工作日' (由此废弃'日期', '星期')
区分时段的信息：'小时(0-23)'
用于具体计算的数据：'归还量', '借出量', '净归还量'
'''

# ===========聚合数据集===========

# 辅助函数：站点类型分类


def get_station_type(name):
    if any(x in name for x in ['宁波大学', '诺丁汉大学', '高教园区']):
        return '大学区'
    elif any(x in name for x in ['万达', '广场', '商城', '商业中心']):
        return '商业区'
    elif any(x in name for x in ['地铁', '火车', '南站', '机场']):
        return '交通枢纽'
    elif any(x in name for x in ['宿舍', '社区', '公园', '中心站', '开发区']):
        return '住宅/偏远区'
    else:
        return '其他'


# 1. 按工作日/休息日分类
df['站点类型'] = df['站点名称'].apply(get_station_type)
weekday_df = df[df['是否为工作日'] == 1].copy()
weekend_df = df[df['是否为工作日'] == 0].copy()

# 2. 时段聚合（区分工作日/休息日）
hourly_avg = df.groupby(['是否为工作日', '小时(0-23)']
                        )[['借出量', '归还量']].mean().reset_index()
weekday_hourly_avg = hourly_avg[hourly_avg['是否为工作日'] == 1].reset_index(
    drop=True)
weekend_hourly_avg = hourly_avg[hourly_avg['是否为工作日'] == 0].reset_index(
    drop=True)

weekday_hourly = weekday_df.groupby('小时(0-23)')[['借出量', '归还量', '净归还量']].mean()
weekend_hourly = weekend_df.groupby('小时(0-23)')[['借出量', '归还量', '净归还量']].mean()

# 3. 站点总量统计
station_summary = df.groupby(['站点编号', '站点名称', '经度', '纬度']).agg({
    '借出量': 'sum',
    '归还量': 'sum'
}).reset_index()
station_summary = station_summary.sort_values('借出量', ascending=False)
station_total_b = station_summary.copy()  # 借出量排序
station_total_r = station_summary.sort_values(
    '归还量', ascending=False).copy()  # 归还量排序

# 4. Top5 和 Bottom5 站点
top5_stations_b = station_total_b.head(5)['站点编号'].tolist()
bottom5_stations_b = station_total_b.tail(5)['站点编号'].tolist()

# 5. 站点热力矩阵（按小时）
heatmap_data_b = df.pivot_table(index='站点名称', columns='小时(0-23)',
                                values='借出量', aggfunc='mean')
heatmap_data_r = df.pivot_table(index='站点名称', columns='小时(0-23)',
                                values='归还量', aggfunc='mean')

# 6. 站点级别工作日/休息日差异
station_diff = pd.DataFrame({
    '站点名称': df['站点名称'].unique()
})
station_work = weekday_df.groupby('站点名称')['借出量'].mean().rename('工作日_借出')
station_weekend = weekend_df.groupby('站点名称')['借出量'].mean().rename('休息日_借出')
station_diff = station_diff.merge(station_work, on='站点名称', how='left')
station_diff = station_diff.merge(station_weekend, on='站点名称', how='left')
station_diff['差异'] = station_diff['工作日_借出'] - station_diff['休息日_借出']

# 7. 站点类型聚合
type_work = weekday_df.groupby('站点类型')['借出量'].mean()
type_weekend = weekend_df.groupby('站点类型')['借出量'].mean()

# 8. 日均借出量（工作日与休息日）- 用于空间分布图
weekday_station_avg = (weekday_df.groupby(['站点编号', '站点名称', '经度', '纬度'])['借出量']
                       .sum() / 5).reset_index(name='日均借出量')  # 假设5个工作日
weekend_station_avg = (weekend_df.groupby(['站点编号', '站点名称', '经度', '纬度'])['借出量']
                       .sum() / 2).reset_index(name='日均借出量')  # 假设2个休息日

# 9. 工作日/休息日的合并空间差异
merged_station_diff = weekday_station_avg.merge(weekend_station_avg,
                                                on=['站点编号', '站点名称', '经度', '纬度'],
                                                suffixes=('_work', '_weekend'))
merged_station_diff['差异'] = merged_station_diff['日均借出量_work'] - \
    merged_station_diff['日均借出量_weekend']

# 10. 变异性统计（借出量标准差）
var_work = weekday_df.groupby('小时(0-23)')['借出量'].std()
var_weekend = weekend_df.groupby('小时(0-23)')['借出量'].std()
var_df = pd.DataFrame({'工作日变异性': var_work, '休息日变异性': var_weekend})

# 11. 总量统计与比例
weekday_total_b = weekday_df['借出量'].sum()
weekend_total_b = weekend_df['借出量'].sum()
weekday_total_r = weekday_df['归还量'].sum()
weekend_total_r = weekend_df['归还量'].sum()
ratio_work = weekday_total_b / weekday_total_r
ratio_weekend = weekend_total_b / weekend_total_r

# 12. Top5 和 Bottom5 的小时聚合（用于折线图）
df_top5 = df[df['站点编号'].isin(top5_stations_b)]
hourly_top5_b = df_top5.groupby(
    ['站点名称', '小时(0-23)'])['借出量'].mean().reset_index()
hourly_top5_r = df_top5.groupby(
    ['站点名称', '小时(0-23)'])['归还量'].mean().reset_index()

df_bottom5 = df[df['站点编号'].isin(bottom5_stations_b)]
hourly_bottom5_b = df_bottom5.groupby(
    ['站点名称', '小时(0-23)'])['借出量'].mean().reset_index()
hourly_bottom5_r = df_bottom5.groupby(
    ['站点名称', '小时(0-23)'])['归还量'].mean().reset_index()

# 13. 打印统计信息
print("\n(2) 借出量最多/最少的5个站点：")
print("借出最多：")
print(station_total_b.head(5)[['站点名称', '借出量']].to_string(index=False))
print("\n借出最少：")
print(station_total_b.tail(5)[['站点名称', '借出量']].to_string(index=False))
print("\n(2) 归还量最多/最少的5个站点：")
print("归还最多：")
print(station_total_r.head(5)[['站点名称', '归还量']].to_string(index=False))
print("\n归还最少：")
print(station_total_r.tail(5)[['站点名称', '归还量']].to_string(index=False))

# （3）工作日 vs 休息日差异
print(
    f"\n(3) 工作日总借出量：{weekday_total_b}，休息日总借出量：{weekend_total_b}，工作日是休息日的 {weekday_total_b/weekend_total_b:.2f} 倍")
print(
    f"\n(3) 工作日总归还量：{weekday_total_r}，休息日总归还量：{weekend_total_r}，工作日是休息日的 {weekday_total_r/weekend_total_r:.2f} 倍")

# ==========数据可视化==========
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 图1：站点总借出量柱状图
plt.figure(figsize=(14, 8))
ax = sns.barplot(data=station_total_b, y='站点名称', x='借出量',
                 hue='站点名称',
                 palette='YlOrRd_r',
                 legend=False,
                 dodge=False)
plt.title('各站点7天总借出量')
plt.xlabel('总借出量')
plt.ylabel('站点')
for container in ax.containers:
    ax.bar_label(container, fmt='%d', padding=3)
plt.tight_layout()
plt.savefig('问题1_总体_站点总借出量柱状图.png', dpi=300, bbox_inches='tight')
plt.close()

# 图2.1：各站点每小时平均借出量热力图
plt.figure(figsize=(14, 10))
sns.heatmap(heatmap_data_b, cmap='YlOrRd', linewidths=0.5, annot=False)
plt.title('各站点每小时平均借出量热力图')
plt.xlabel('小时 (0-23)')
plt.ylabel('站点名称')
plt.tight_layout()
plt.savefig('问题1_总体_借出量热力图.png', dpi=300, bbox_inches='tight')
plt.close()

# 图2.2：各站点每小时平均归还量热力图
plt.figure(figsize=(14, 10))
sns.heatmap(heatmap_data_r, cmap='YlOrRd', linewidths=0.5, annot=False)
plt.title('各站点每小时平均归还量热力图')
plt.xlabel('小时 (0-23)')
plt.ylabel('站点名称')
plt.tight_layout()
plt.savefig('问题1_总体_归还量热力图.png', dpi=300, bbox_inches='tight')
plt.close()

# 图3：空间散点图
plt.figure(figsize=(10, 8))
plt.scatter(station_total_b['经度'], station_total_b['纬度'], s=station_total_b['借出量']
            * 0.5, c=station_total_b['借出量'], cmap='YlOrRd', alpha=0.7)
for i, row in station_total_b.iterrows():
    plt.text(row['经度']+0.001, row['纬度'], row['站点名称'], fontsize=8)
plt.title('各站点借出量空间分布（点的大小及颜色深浅表示借出量）')
plt.xlabel('经度')
plt.ylabel('纬度')
plt.grid(True)
plt.savefig('问题1_总体_空间散点图.png', dpi=300, bbox_inches='tight')
plt.close()

# 图4.1：总借出量Top5站点的每小时平均借出量曲线（这里先不区分工作日/休息日）
plt.figure(figsize=(14, 8))
print(
    f"借出量Top5站点：{station_total_b.head(5)[['站点名称', '借出量']].to_string(index=False)}")
sns.lineplot(data=hourly_top5_b, x='小时(0-23)', y='借出量',
             hue='站点名称',  # 不同站点不同颜色
             marker='o',      # 数据点用圆圈标记
             linewidth=2.5,   # 线条粗细
             markersize=6)    # 标记点大小
plt.title('借出量Top5站点的每小时平均借出量分布', fontsize=16, fontweight='bold')
plt.xlabel('小时 (0-23)', fontsize=12)
plt.ylabel('平均借出量（次/小时）', fontsize=12)
plt.legend(title='站点名称', bbox_to_anchor=(1.05, 1), loc='upper left')  # 图例放在右侧
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(range(0, 24, 2))  # 每2小时显示一个刻度
plt.tight_layout()
plt.savefig('问题1_Top5站点_每小时借出量曲线.png', dpi=300, bbox_inches='tight')
plt.close()

# 图4.2：总借出量Top5站点的每小时平均归还量曲线（这里先不区分工作日/休息日）
plt.figure(figsize=(14, 8))
sns.lineplot(data=hourly_top5_r, x='小时(0-23)', y='归还量',
             hue='站点名称',  # 不同站点不同颜色
             marker='o',      # 数据点用圆圈标记
             linewidth=2.5,   # 线条粗细
             markersize=6)    # 标记点大小
plt.title('借出量Top5站点的每小时平均归还量分布', fontsize=16, fontweight='bold')
plt.xlabel('小时 (0-23)', fontsize=12)
plt.ylabel('平均归还量（次/小时）', fontsize=12)
plt.legend(title='站点名称', bbox_to_anchor=(1.05, 1), loc='upper left')  # 图例放在右侧
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(range(0, 24, 2))  # 每2小时显示一个刻度
plt.tight_layout()
plt.savefig('问题1_Top5站点_每小时归还量曲线.png', dpi=300, bbox_inches='tight')
plt.close()

# 图4.3：总借出量Bottom5站点的每小时平均借出量曲线（这里先不区分工作日/休息日）
plt.figure(figsize=(14, 8))
print(
    f"借出量Bottom5站点：{station_total_b.tail(5)[['站点名称', '借出量']].to_string(index=False)}")
sns.lineplot(data=hourly_bottom5_b, x='小时(0-23)', y='借出量',
             hue='站点名称',  # 不同站点不同颜色
             marker='o',      # 数据点用圆圈标记
             linewidth=2.5,   # 线条粗细
             markersize=6)    # 标记点大小
plt.title('借出量Bottom5站点的每小时平均借出量分布', fontsize=16, fontweight='bold')
plt.xlabel('小时 (0-23)', fontsize=12)
plt.ylabel('平均借出量（次/小时）', fontsize=12)
plt.legend(title='站点名称', bbox_to_anchor=(1.05, 1), loc='upper left')  # 图例放在右侧
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(range(0, 24, 2))  # 每2小时显示一个刻度
plt.tight_layout()
plt.savefig('问题1_Bottom5站点_每小时借出量曲线.png', dpi=300, bbox_inches='tight')
plt.close()

# 图4.4：总借出量Bottom5站点的每小时平均归还量曲线（这里先不区分工作日/休息日）
plt.figure(figsize=(14, 8))
sns.lineplot(data=hourly_bottom5_r, x='小时(0-23)', y='归还量',
             hue='站点名称',  # 不同站点不同颜色
             marker='o',      # 数据点用圆圈标记
             linewidth=2.5,   # 线条粗细
             markersize=6)    # 标记点大小
plt.title('借出量Bottom5站点的每小时平均归还量分布', fontsize=16, fontweight='bold')
plt.xlabel('小时 (0-23)', fontsize=12)
plt.ylabel('平均归还量（次/小时）', fontsize=12)
plt.legend(title='站点名称', bbox_to_anchor=(1.05, 1), loc='upper left')  # 图例放在右侧
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(range(0, 24, 2))  # 每2小时显示一个刻度
plt.tight_layout()
plt.savefig('问题1_Bottom5站点_每小时归还量曲线.png', dpi=300, bbox_inches='tight')
plt.close()

# 图5：工作日与休息日借还时段对比图
plt.figure(figsize=(12, 6))
# 工作日借出（实线）
plt.plot(weekday_hourly_avg['小时(0-23)'], weekday_hourly_avg['借出量'],
         'r-o', linewidth=1.5, markersize=6, label='工作日借出')
# 工作日归还（虚线）
plt.plot(weekday_hourly_avg['小时(0-23)'], weekday_hourly_avg['归还量'],
         'r-.s', linewidth=1.5, markersize=6, label='工作日归还')
# 休息日借出（实线）
plt.plot(weekend_hourly_avg['小时(0-23)'], weekend_hourly_avg['借出量'],
         'c-o', linewidth=1.5, markersize=6, label='休息日借出')
# 休息日归还（虚线）
plt.plot(weekend_hourly_avg['小时(0-23)'], weekend_hourly_avg['归还量'],
         'c-.s', linewidth=1.5, markersize=6, label='休息日归还')
plt.title('工作日与休息日借还时段对比图', fontsize=14)
plt.xlabel('小时 (0-23)', fontsize=12)
plt.ylabel('平均车辆数', fontsize=12)
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.savefig('问题1_差异_工作日与休息日借还时段对比图.png', dpi=300, bbox_inches='tight')
plt.close()

# 图6: 总量柱状对比
plt.figure(figsize=(8, 5))
totals = pd.DataFrame({
    '日类型': ['工作日', '休息日'],
    '总借出量': [weekday_total_b, weekend_total_b]
})
sns.barplot(data=totals, x='日类型', y='总借出量', palette='Set2')
plt.title('工作日 vs 休息日总借出量对比')
plt.ylabel('总借出量')
plt.savefig('问题1_差异_总量柱状图.png', dpi=300, bbox_inches='tight')
plt.close()

# 图7.1: 空间差异散点图（工作日 vs 休息日借出量，点大小=差异）
plt.figure(figsize=(14, 6))

# 绘图
plt.subplot(1, 2, 1)
plt.scatter(weekday_station_avg['经度'],
            weekday_station_avg['纬度'],
            s=weekday_station_avg['日均借出量'] * 0.8,   # 调整 0.8 这个缩放系数，让点大小合适
            c='red', alpha=0.7, edgecolors='black', linewidth=0.3)
plt.title('工作日各站点日均借出量空间分布')
plt.xlabel('经度')
plt.ylabel('纬度')

plt.subplot(1, 2, 2)
plt.scatter(weekend_station_avg['经度'],
            weekend_station_avg['纬度'],
            s=weekend_station_avg['日均借出量'] * 0.8,   # 同样缩放
            c='blue', alpha=0.7, edgecolors='black', linewidth=0.3)
plt.title('休息日各站点日均借出量空间分布')
plt.xlabel('经度')

plt.tight_layout()
plt.savefig('问题1_差异_空间对比.png', dpi=300, bbox_inches='tight')
plt.close()

# 图7.2：空间差异图（合并）
plt.figure(figsize=(8, 6))
sc = plt.scatter(merged_station_diff['经度'], merged_station_diff['纬度'],
                 s=merged_station_diff['差异'].abs() * 2,      # 差异越大点越大
                 c=merged_station_diff['差异'],                # 正负用颜色区分
                 cmap='RdBu_r', vmin=-max(abs(merged_station_diff['差异'])), vmax=max(abs(merged_station_diff['差异'])),
                 alpha=0.8, edgecolors='black')
plt.colorbar(sc, label='工作日 - 休息日 日均借出量差异')
plt.title('工作日与休息日各站点日均借出量差异空间分布')
plt.xlabel('经度')
plt.ylabel('纬度')
plt.tight_layout()
plt.savefig('问题1_差异_空间差异图（合并）.png', dpi=300, bbox_inches='tight')
plt.close()

# 图8: 站点借出量差异热力图（工作日-休息日）
diff_heatmap = station_diff.set_index('站点名称')[['差异']].T
plt.figure(figsize=(14, 8))
sns.heatmap(diff_heatmap, cmap='RdBu', center=0,
            annot=True, fmt='.1f', linewidths=0.5)
plt.title('各站点借出量差异（工作日 - 休息日，正值=工作日更高）')
plt.xlabel('站点名称')
plt.savefig('问题1_差异_站点差异热力图.png', dpi=300, bbox_inches='tight')
plt.close()

# 图9: 站点类型差异柱状图
plt.figure(figsize=(10, 6))
type_df = pd.DataFrame({'工作日': type_work, '休息日': type_weekend})
type_df.plot(kind='bar', color=['orange', 'skyblue'])
plt.title('不同站点类型借出量对比')
plt.ylabel('平均每小时借出量')
plt.xticks(rotation=0)
plt.legend(title='日类型')
plt.savefig('问题1_差异_站点类型柱状图.png', dpi=300, bbox_inches='tight')
plt.close()

# 图10: 净归还量差异时序图
plt.figure(figsize=(10, 6))
plt.plot(weekday_hourly.index,
         weekday_hourly['净归还量'], label='工作日净归还量', marker='o', color='red')
plt.plot(weekend_hourly.index,
         weekend_hourly['净归还量'], label='休息日净归还量', marker='s', color='blue')
plt.axhline(0, color='gray', linestyle='--')
plt.title('工作日与休息日净归还量对比（净归还量 = 归还量 - 借出量）')
plt.xlabel('小时')
plt.ylabel('平均净归还量')
plt.legend()
plt.grid(True)
plt.savefig('问题1_差异_净归还量时序图.png', dpi=300, bbox_inches='tight')
plt.close()

# 图11: 变异性对比柱状图
var_df.plot(kind='bar', figsize=(12, 6), color=['salmon', 'lightblue'])
plt.title('工作日 vs 休息日小时借出量变异性（标准差）')
plt.ylabel('标准差')
plt.savefig('问题1_差异_变异性柱状图.png', dpi=300, bbox_inches='tight')
plt.close()

# 图12: 借还平衡差异（借出/归还比例）
plt.figure(figsize=(6, 6))
plt.bar(['工作日', '休息日'], [ratio_work, ratio_weekend],
        color=['#FF6B6B', '#4ECDC4'])
plt.title('借出/归还比例（>1 表示借>还）')
plt.ylabel('借出量 / 归还量')
plt.savefig('问题1_差异_借还平衡柱状图.png', dpi=300, bbox_inches='tight')
plt.close()
