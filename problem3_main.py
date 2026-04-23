"""
问题3第(2)小题 - 调度优化对比完整解决方案

核心思路：
1. 多轮调度 - 每小时内根据容量进行多次调度循环
2. 未来需求预测 - 考虑未来2-3小时的需求缺口
3. 距离加权优先级 - priority = deficit_j × surplus_i / distance_ij
4. 离散时延到站 - 调度在 t 时刻发车，在 t + round(d_ij / v) 时刻到达

使用的文件：
- problem3_config.py (配置加载)
- problem3_simulation.py (模拟器)
- problem3_main.py (主程序)
"""

import numpy as np
import pandas as pd


def main():
    """
    主函数 - 问题3第(2)小题完整运行流程
    """
    from problem3_config import get_config, load_demand_predictions
    from problem3_simulation import StationSimulator

    print("="*80)
    print("问题3 第(2)小题：共享单车调度优化方案")
    print("="*80)

    # 1. 加载配置
    print("\n[第1步] 加载配置和基础数据...")
    try:
        config = get_config()
        print(f"  [OK] 成功加载配置")
        print(f"    - 站点总数: {config['n_stations']} 个")
        print(f"    - 货车最大容量: {config['truck_capacity']} 辆/次")
        print(
            f"    - 调度时间窗: {config['time_window_start']}:00-{config['time_window_end']}:00")
        print("    - 调度时间: t_ij = round(d_ij / v)")
        print(f"    - 满桩惩罚(α): {config['alpha']:.1f} 元/(桩·小时)")
        print(f"    - 空桩惩罚(β): {config['beta']:.1f} 元/(桩·小时)")
        print(
            f"    - 运输成本: {config['c_km']:.1f} 元/公里 + {config['c_move']:.1f} 元/辆")
    except Exception as e:
        print(f"  [ERROR] 加载配置失败: {e}")
        return

    config['output_dir'].mkdir(parents=True, exist_ok=True)

    # 2. 加载需求数据
    print("\n[第2步] 加载需求预测数据...")
    try:
        demand_df = load_demand_predictions()
        print(f"  [OK] 成功加载需求数据")
        print(f"    - 数据行数: {len(demand_df)}")
        print(f"    - 覆盖站点: {demand_df['站点编号'].nunique()} 个")
        print(
            f"    - 时间跨度: {demand_df['hour'].min():.0f}-{demand_df['hour'].max():.0f} 小时")
    except Exception as e:
        print(f"  [ERROR] 加载需求数据失败: {e}")
        return

    # 3. 无调度模拟
    print("\n[第3步] 模式1: 无调度模拟...")
    try:
        simulator_no = StationSimulator(config, demand_df)
        results_no_schedule = simulator_no.run_no_schedule()
        print(f"  [OK] 无调度模拟完成")
        print(f"    - 日总费用: {results_no_schedule['total_cost']:,.2f} 元")
        print(f"    - 惩罚成本: {results_no_schedule['total_penalty']:,.2f} 元")
        print(f"    - 借车失败: {results_no_schedule['total_fail_borrow']:.0f} 次")
        print(f"    - 还车失败: {results_no_schedule['total_fail_return']:.0f} 次")
        print(f"    - 用户需求满足率: {results_no_schedule['success_rate']:.2f}%")
    except Exception as e:
        print(f"  [ERROR] 无调度模拟失败: {e}")
        return

    # 4. 有调度模拟（增强贪心算法）
    print("\n[第4步] 模式2: 有调度模拟（增强贪心策略）...")
    print("  运行中... (包含多轮调度、未来需求、距离加权、离散时延)")
    try:
        simulator_with = StationSimulator(config, demand_df)
        results_with_schedule = simulator_with.run_with_schedule()
        print(f"  [OK] 有调度模拟完成")
        print(f"    - 日总费用: {results_with_schedule['total_cost']:,.2f} 元")
        print(f"    - 惩罚成本: {results_with_schedule['total_penalty']:,.2f} 元")
        print(
            f"    - 运输成本: {results_with_schedule['total_transport_cost']:,.2f} 元")
        print(
            f"    - 借车失败: {results_with_schedule['total_fail_borrow']:.0f} 次")
        print(
            f"    - 还车失败: {results_with_schedule['total_fail_return']:.0f} 次")
        print(f"    - 用户需求满足率: {results_with_schedule['success_rate']:.2f}%")
    except Exception as e:
        print(f"  [ERROR] 有调度模拟失败: {e}")
        return

    # 5. 结果对比分析
    print("\n[第5步] 对比分析与优化效果评估...")
    cost_saving = results_no_schedule['total_cost'] - \
        results_with_schedule['total_cost']
    rate_improvement = results_with_schedule['success_rate'] - \
        results_no_schedule['success_rate']
    penalty_reduction = results_no_schedule['total_penalty'] - \
        results_with_schedule['total_penalty']

    print("\n" + "-"*80)
    print("【效果对比】")
    print("-"*80)
    print(f"{'指标':<30} {'无调度':>15} {'有调度':>15} {'优化幅度':>15}")
    print("-"*80)

    total_cost_no = results_no_schedule['total_cost']
    total_cost_with = results_with_schedule['total_cost']
    print(f"{'日总费用(元)':<30} {total_cost_no:>15,.0f} {total_cost_with:>15,.0f} {cost_saving:>14,.0f}")

    penalty_no = results_no_schedule['total_penalty']
    penalty_with = results_with_schedule['total_penalty']
    print(f"{'惩罚成本(元)':<30} {penalty_no:>15,.0f} {penalty_with:>15,.0f} {penalty_reduction:>14,.0f}")

    transport_cost = results_with_schedule['total_transport_cost']
    print(f"{'运输成本(元)':<30} {'—':>15} {transport_cost:>15,.0f} {'—':>15}")

    fail_borrow_no = results_no_schedule['total_fail_borrow']
    fail_borrow_with = results_with_schedule['total_fail_borrow']
    fail_reduction = fail_borrow_no - fail_borrow_with
    print(f"{'借车失败次数':<30} {fail_borrow_no:>15.0f} {fail_borrow_with:>15.0f} {fail_reduction:>14.0f}")

    fail_return_no = results_no_schedule['total_fail_return']
    fail_return_with = results_with_schedule['total_fail_return']
    print(f"{'还车失败次数':<30} {fail_return_no:>15.0f} {fail_return_with:>15.0f} {fail_return_no-fail_return_with:>14.0f}")

    success_rate_no = results_no_schedule['success_rate']
    success_rate_with = results_with_schedule['success_rate']
    print(f"{'用户需求满足率(%)':<30} {success_rate_no:>14.2f}% {success_rate_with:>14.2f}% {rate_improvement:>13.2f}%")

    print("-"*80)

    # 6. 调度效果总结
    print("\n【调度方案统计】")
    print("-"*80)
    all_schedules = results_with_schedule['records']['schedules']
    total_trips = sum(len(h_sch) for h_sch in all_schedules)
    total_vehicles = sum(sum(s['amount'] for s in h_sch)
                         for h_sch in all_schedules)

    print(f"全天调度轮次: {total_trips} 次行程")
    print(f"全天搬运单位: {total_vehicles:.0f} 辆·次")

    # 按小时统计调度
    schedule_by_hour = {}
    for hour, hour_schedules in enumerate(all_schedules):
        if hour_schedules:
            schedule_by_hour[hour] = (len(hour_schedules), sum(
                s['amount'] for s in hour_schedules))

    if schedule_by_hour:
        print("\n按小时调度分布:")
        for hour in sorted(schedule_by_hour.keys()):
            trips, vehicles = schedule_by_hour[hour]
            print(
                f"  {hour:02d}:00 - {hour+1:02d}:00  |  {trips} 次行程, {vehicles:.0f} 辆车")

    # 7. 保存结果文件
    print("\n[第6步] 保存结果到文件...")
    try:
        save_results(results_no_schedule, results_with_schedule, config)
        print(f"  [OK] 结果已保存")
    except Exception as e:
        print(f"  [ERROR] 保存结果失败: {e}")

    # 8. 最终总结
    print("\n" + "="*80)
    print("【优化方案总结】")
    print("="*80)
    print(f"""
调度优化方案采用【增强贪心算法】，具体策略：

1. 多轮调度机制
   - 每个时间段内根据货车容量进行多轮调度
   - 循环匹配供给站和需求站，直到容量用完或无可行配对

2. 未来需求预测
   - 考虑当前及未来2-3小时的库存缺口
   - 公式: deficit = max(0, R_i - S_i,t) + λ·Σ(k=1~2)max(0, R_i - S_i,t+k)
   - λ = 0.5（未来权重）

3. 距离加权优先级
   - 优先级分数: priority_ij = deficit_j × surplus_i / distance_ij
   - 贪心选择最高优先级的站点对进行调度

4. 离散时延到站
   - 调度车辆在出发时刻立即从源站扣减
   - 目标站在 arrival_hour = depart_hour + round(distance_ij / v) 时入库
   - 不再采用瞬时调度近似

【最终效果】
  成本节省: {cost_saving:,.0f} 元 ({cost_saving/total_cost_no*100:.1f}%)
  需求满足率提升: {rate_improvement:.2f} 百分点
  运输成本: {transport_cost:,.0f} 元/天
  """)

    print("="*80)
    print("问题3 第(2)小题求解完成。")
    print("="*80)


def save_results(results_no, results_with, config):
    """
    保存模拟结果到CSV文件
    题目要求的5个对比指标：
    1. 总费用
    2. 满桩发生次数
    3. 空桩发生次数
    4. 用户无法借车的次数
    5. 用户无法还车的次数
    """
    # 1. 对比汇总表（仅包括题目要求的指标）
    # 直接写入 CSV 文件，确保正确的数据类型格式
    output_dir = config['output_dir']
    summary_path = output_dir / 'problem3_comparison_summary.csv'
    schedules_path = output_dir / 'problem3_schedules_detail.csv'
    inventory_path = output_dir / 'problem3_inventory_detail.csv'

    with open(summary_path, 'w', encoding='utf-8-sig') as f:
        f.write('指标,无调度,有调度\n')
        f.write(
            f'总费用(元),{results_no["total_cost"]:.2f},{results_with["total_cost"]:.2f}\n')
        f.write(
            f'满桩发生次数(次),{int(results_no["total_full_count"])},{int(results_with["total_full_count"])}\n')
        f.write(
            f'空桩发生次数(次),{int(results_no["total_empty_count"])},{int(results_with["total_empty_count"])}\n')
        f.write(
            f'用户无法借车(次),{int(results_no["total_fail_borrow"])},{int(results_with["total_fail_borrow"])}\n')
        f.write(
            f'用户无法还车(次),{int(results_no["total_fail_return"])},{int(results_with["total_fail_return"])}\n')

    print(f"    - 已保存: {summary_path}")

    # 2. 调度方案详情
    schedules_data = []
    for hour, hour_schedules in enumerate(results_with['records']['schedules']):
        for sch in hour_schedules:
            schedules_data.append({
                '时刻': f"{hour:02d}:00",
                '出发站': config['idx_to_station'][sch['from']],
                '目标站': config['idx_to_station'][sch['to']],
                '调度数量(辆)': sch['amount'],
                '运输时长(小时)': sch['travel_time'],
                '到达时刻': f"{sch['arrival_hour']:02d}:00"
            })

    if schedules_data:
        schedules_df = pd.DataFrame(schedules_data)
        schedules_df.to_csv(schedules_path,
                            index=False, encoding='utf-8-sig')
        print(f"    - 已保存: {schedules_path}")

    # 3. 每小时库存对比
    inventory_data = []
    for t in range(24):
        for i in range(config['n_stations']):
            inventory_data.append({
                '小时': f"{t:02d}:00",
                '站点编号': config['idx_to_station'][i],
                '无调度库存': int(results_no['records']['inventory'][t][i]),
                '有调度库存': int(results_with['records']['inventory'][t][i]),
                '库存差': int(results_with['records']['inventory'][t][i] - results_no['records']['inventory'][t][i])
            })

    inventory_df = pd.DataFrame(inventory_data)
    inventory_df.to_csv(inventory_path,
                        index=False, encoding='utf-8-sig')
    print(f"    - 已保存: {inventory_path}")


if __name__ == "__main__":
    main()
