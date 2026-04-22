"""
问题3主程序
运行无调度和有调度模拟，输出对比结果
"""

import numpy as np
import pandas as pd
from problem3_config import get_config, load_demand_predictions
from problem3_simulation import StationSimulator, format_schedule_output


def print_results(results_no, results_with, config):
    """
    打印对比结果
    """
    print("\n" + "="*80)
    print("问题3 第(2)小题：调度优化对比结果")
    print("="*80)
    
    print("\n【指标对比】")
    print("-" * 60)
    print(f"{'指标':<35} {'无调度':>12} {'有调度':>12}")
    print("-" * 60)
    
    # 总费用
    print(f"{'日总费用 (元)':<35} {results_no['total_cost']:>12.2f} {results_with['total_cost']:>12.2f}")
    
    # 惩罚成本
    print(f"{'惩罚成本 (元)':<35} {results_no['total_penalty']:>12.2f} {results_with['total_penalty']:>12.2f}")
    
    # 运输成本
    print(f"{'运输成本 (元)':<35} {0:>12.2f} {results_with['total_transport_cost']:>12.2f}")
    
    # 借车失败次数
    print(f"{'借车失败次数 (次)':<35} {results_no['total_fail_borrow']:>12.0f} {results_with['total_fail_borrow']:>12.0f}")
    
    # 还车失败次数
    print(f"{'还车失败次数 (次)':<35} {results_no['total_fail_return']:>12.0f} {results_with['total_fail_return']:>12.0f}")
    
    # 总需求
    total_demand = results_no['total_demand_out'] + results_no['total_demand_in']
    print(f"{'总需求次数 (次)':<35} {total_demand:>12.0f} {total_demand:>12.0f}")
    
    # 需求率
    print(f"{'用户需求率 (%)':<35} {results_no['success_rate']:>12.2f} {results_with['success_rate']:>12.2f}")
    
    # 费用节省
    cost_saving = results_no['total_cost'] - results_with['total_cost']
    rate_improvement = results_with['success_rate'] - results_no['success_rate']
    
    print("-" * 60)
    print(f"{'费用节省 (元)':<35} {cost_saving:>12.2f}")
    print(f"{'需求率提升 (%)':<35} {rate_improvement:>12.2f}")
    
    print("\n" + "="*80)


def print_schedule_summary(schedules, config):
    """
    打印调度方案汇总
    """
    print("\n【调度方案汇总】")
    print("-" * 50)
    
    if not schedules:
        print("当天无调度")
        return
    
    # 按时间分组
    schedule_by_hour = {}
    for hour, hour_schedules in enumerate(schedules):
        if hour_schedules:
            schedule_by_hour[hour] = hour_schedules
    
    total_trips = 0
    total_vehicles = 0
    
    for hour, sch_list in schedule_by_hour.items():
        trips = len(sch_list)
        vehicles = sum(s['amount'] for s in sch_list)
        total_trips += trips
        total_vehicles += vehicles
        
        print(f"\n小时 {hour:02d}:00 (共{trips}次调度, {vehicles}辆车)")
        for sch in sch_list:
            from_id = config['idx_to_station'][sch['from']]
            to_id = config['idx_to_station'][sch['to']]
            print(f"  {from_id} → {to_id}: {sch['amount']}辆")
    
    print(f"\n全天统计: {total_trips}次调度行程, {total_vehicles}辆次搬运")


def print_station_hourly_inventory(results, config, station_ids=None):
    """
    打印指定站点的每小时库存变化
    """
    if station_ids is None:
        # 默认选择几个代表性站点
        station_ids = ['S002', 'S012', 'S023']
    
    station_to_idx = config['station_to_idx']
    idx_to_station = config['idx_to_station']
    
    for sid in station_ids:
        if sid not in station_to_idx:
            print(f"站点 {sid} 不存在")
            continue
        
        idx = station_to_idx[sid]
        
        print(f"\n站点 {sid} 每小时库存变化:")
        print("  小时 | 无调度 | 有调度")
        print("  -----|--------|--------")
        
        for t in range(24):
            inv_no = results['no_schedule']['records']['inventory'][t][idx]
            inv_with = results['with_schedule']['records']['inventory'][t][idx]
            print(f"  {t:02d}:00 | {inv_no:6.0f} | {inv_with:6.0f}")


def save_results_to_csv(results_no, results_with, config):
    """
    保存结果到CSV文件
    """
    # 保存对比汇总
    summary = {
        '指标': ['日总费用(元)', '惩罚成本(元)', '运输成本(元)', '借车失败(次)', '还车失败(次)', '用户需求率(%)'],
        '无调度': [results_no['total_cost'], results_no['total_penalty'], 0, 
                  results_no['total_fail_borrow'], results_no['total_fail_return'], results_no['success_rate']],
        '有调度': [results_with['total_cost'], results_with['total_penalty'], results_with['total_transport_cost'],
                  results_with['total_fail_borrow'], results_with['total_fail_return'], results_with['success_rate']]
    }
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv('problem3_comparison.csv', index=False, encoding='utf-8-sig')
    print("\n对比结果已保存至: problem3_comparison.csv")
    
    # 保存调度方案
    schedules_data = []
    for hour, hour_schedules in enumerate(results_with['records']['schedules']):
        for sch in hour_schedules:
            schedules_data.append({
                '小时': hour,
                '出发站点': config['idx_to_station'][sch['from']],
                '到达站点': config['idx_to_station'][sch['to']],
                '调度数量': sch['amount']
            })
    if schedules_data:
        schedules_df = pd.DataFrame(schedules_data)
        schedules_df.to_csv('problem3_schedules.csv', index=False, encoding='utf-8-sig')
        print("调度方案已保存至: problem3_schedules.csv")
    
    # 保存每小时库存
    inventory_data = []
    for t in range(24):
        for i in range(config['n_stations']):
            inventory_data.append({
                '小时': t,
                '站点编号': config['idx_to_station'][i],
                '无调度库存': results_no['records']['inventory'][t][i],
                '有调度库存': results_with['records']['inventory'][t][i]
            })
    inventory_df = pd.DataFrame(inventory_data)
    inventory_df.to_csv('problem3_inventory.csv', index=False, encoding='utf-8-sig')
    print("每小时库存已保存至: problem3_inventory.csv")


def main():
    """
    主函数
    """
    print("="*80)
    print("问题3：共享单车站点智能调度优化")
    print("="*80)
    
    # 1. 加载配置
    print("\n[1] 加载配置和数据...")
    config = get_config()
    print(f"  - 站点数量: {config['n_stations']}")
    print(f"  - 货车容量: {config['truck_capacity']}辆")
    print(f"  - 调度时间窗: {config['time_window_start']}:00 - {config['time_window_end']}:00")
    print(f"  - 满桩惩罚系数: α={config['alpha']}")
    print(f"  - 空桩惩罚系数: β={config['beta']}")
    
    # 2. 加载需求数据
    print("\n[2] 加载需求数据...")
    demand_df = load_demand_predictions()
    print(f"  - 需求数据行数: {len(demand_df)}")
    
    # 3. 无调度模拟
    print("\n[3] 运行无调度模拟...")
    sim = StationSimulator(config, demand_df)
    results_no = sim.run_no_schedule()
    print(f"  - 日总费用: {results_no['total_cost']:.2f}元")
    print(f"  - 用户需求率: {results_no['success_rate']:.2f}%")
    
    # 4. 有调度模拟
    print("\n[4] 运行有调度模拟（增强贪心算法）...")
    sim = StationSimulator(config, demand_df)
    results_with = sim.run_with_schedule()
    print(f"  - 日总费用: {results_with['total_cost']:.2f}元")
    print(f"  - 用户需求率: {results_with['success_rate']:.2f}%")
    print(f"  - 运输成本: {results_with['total_transport_cost']:.2f}元")
    
    # 5. 打印对比结果
    results = {
        'no_schedule': results_no,
        'with_schedule': results_with
    }
    print_results(results_no, results_with, config)
    
    # 6. 打印调度方案详情
    print_schedule_summary(results_with['records']['schedules'], config)
    
    # 7. 打印代表性站点库存变化
    print_station_hourly_inventory(results, config, station_ids=['S002', 'S012', 'S023'])
    
    # 8. 保存结果
    print("\n[5] 保存结果...")
    save_results_to_csv(results_no, results_with, config)
    
    print("\n" + "="*80)
    print("问题3第(2)小题运行完成！")
    print("="*80)


if __name__ == "__main__":
    main()