"""
问题3模拟仿真
包含：无调度模拟、增强贪心调度算法
"""

import numpy as np
import pandas as pd
from problem3_config import get_config, load_demand_predictions, clean_columns


class StationSimulator:
    def __init__(self, config, demand_df):
        """
        初始化模拟器
        config: 配置参数
        demand_df: 需求数据（预测值）
        """
        self.config = config
        self.n = config['n_stations']

        # 确保需求数据的列名正确
        demand_df = clean_columns(demand_df)

        # 解析需求数据
        self.parse_demand(demand_df)

        # 重置状态
        self.reset()

    def parse_demand(self, demand_df):
        """
        解析需求数据，构建 D_out[i,t] 和 D_in[i,t]
        """
        self.D_out = np.zeros((self.n, 24))  # 借出需求
        self.D_in = np.zeros((self.n, 24))   # 归还需求

        station_to_idx = self.config['station_to_idx']

        # 清理列名
        demand_df = clean_columns(demand_df)

        print(f"\n  [调试] 需求数据列名: {demand_df.columns.tolist()}")
        print(f"  [调试] 数据行数: {len(demand_df)}")

        # 解析每一行数据
        count = 0
        for _, row in demand_df.iterrows():
            sid = row['站点编号']
            hour = int(row['hour'])
            borrow = float(row['借出量'])
            return_val = float(row['归还量'])

            if sid in station_to_idx:
                i = station_to_idx[sid]
                self.D_out[i, hour] = borrow
                self.D_in[i, hour] = return_val
                count += 1

        print(f"  [调试] 成功解析 {count} 条需求记录")
        print(f"  [调试] 24小时总借出需求: {self.D_out.sum():.0f} 辆")
        print(f"  [调试] 24小时总归还需求: {self.D_in.sum():.0f} 辆")

        # 检查是否有充足的需求数据
        if self.D_out.sum() == 0 and self.D_in.sum() == 0:
            print("  [警告] 需求数据为空！")

    def reset(self):
        """
        重置模拟状态
        """
        self.S = self.config['init_inventory'].copy()  # 当前小时开始时的库存
        self.records = {
            'inventory': [],           # 每小时结束时的库存
            'borrow_success': [],      # 实际借出量
            'return_success': [],      # 实际归还量
            'fail_borrow': [],         # 借车失败次数
            'fail_return': [],         # 还车失败次数
            'penalty_over': [],        # 满桩惩罚
            'penalty_under': [],       # 空桩惩罚
            'transport_cost': [],      # 运输成本
            'schedules': [],           # 调度记录
        }

    def simulate_borrow_return(self, t):
        """
        模拟借车和还车过程
        返回：借车后库存、临时库存、失败次数
        """
        # 借车过程
        borrow_demand = self.D_out[:, t]
        borrow_actual = np.minimum(self.S, borrow_demand)
        fail_borrow = borrow_demand - borrow_actual

        # 借车后库存
        S_after_borrow = self.S - borrow_actual

        # 还车过程
        return_demand = self.D_in[:, t]
        available_space = self.config['capacity'] - S_after_borrow
        return_actual = np.minimum(available_space, return_demand)
        fail_return = return_demand - return_actual

        # 临时库存（借还后，调度前）
        S_temp = S_after_borrow + return_actual

        return S_after_borrow, S_temp, borrow_actual, return_actual, fail_borrow, fail_return

    def calculate_surplus_deficit(self, S_temp, t):
        """
        计算过剩量和短缺量（考虑未来需求）

        surplus: 库存高于安全库存的部分（可以调出的车辆）
        deficit: 库存低于安全库存的部分（需要调入的车辆）
        """
        safe_inv = self.config['safe_inventory']

        # 修订1: surplus = 库存超过安全库存的部分
        # 这表示该站点有多余车辆可以调出
        surplus = np.maximum(0, S_temp - safe_inv)

        # 修订2: 当前短缺（低于安全库存）
        deficit = np.maximum(0, safe_inv - S_temp)

        # 未来需求预测的额外短缺（2小时视野）
        future_deficit = np.zeros(self.n)
        for k in range(1, min(self.config['future_horizon'] + 1, 24 - t)):
            # 预估未来库存
            future_S = S_temp.copy()

            # 逐小时累积未来库存变化
            for step in range(1, k + 1):
                if t + step < 24:
                    net_demand = self.D_out[:, t +
                                            step] - self.D_in[:, t + step]
                    future_S = future_S - net_demand
                    future_S = np.clip(future_S, 0, self.config['capacity'])

            # 计算未来的短缺量
            future_deficit += np.maximum(0, safe_inv - future_S)

        # 加权合并：当前短缺 + 未来权重 × 未来短缺
        deficit_enhanced = deficit + \
            self.config['future_lambda'] * future_deficit

        return surplus, deficit_enhanced

    def greedy_schedule_enhanced(self, S_temp, t, remaining_capacity):
        """
        增强贪心调度算法
        返回：调度记录和运输成本
        """
        if t < self.config['time_window_start'] or t >= self.config['time_window_end']:
            return [], 0.0

        surplus, deficit = self.calculate_surplus_deficit(S_temp, t)

        # 找出有盈余和短缺的站点
        surplus_stations = [(i, surplus[i])
                            for i in range(self.n) if surplus[i] > 1e-6]
        deficit_stations = [(j, deficit[j])
                            for j in range(self.n) if deficit[j] > 1e-6]

        # 调试输出（仅在第一次有调度时输出）
        if (surplus_stations or deficit_stations) and t == 6:
            print(f"\n  [调试] 时刻 {t:02d}:00 调度分析:")
            print(f"    - 有余量站点数: {len(surplus_stations)}")
            print(f"    - 有缺口站点数: {len(deficit_stations)}")
            if surplus_stations:
                print(
                    f"    - 总余量: {sum(s for _, s in surplus_stations):.0f} 辆")
            if deficit_stations:
                print(
                    f"    - 总缺口: {sum(d for _, d in deficit_stations):.0f} 辆")

        if not surplus_stations or not deficit_stations:
            return [], 0.0

        schedules = []
        total_cost = 0.0
        cap_remaining = remaining_capacity
        round_count = 0

        # 多轮调度（直到容量用完或无可行配对）
        while cap_remaining > 0 and surplus_stations and deficit_stations:
            best_score = -1
            best_i, best_j = None, None

            # 计算每对站点的优先级分数
            for i, s_val in surplus_stations:
                for j, d_val in deficit_stations:
                    if i == j:
                        continue
                    dist = self.config['dist_matrix'][i, j]
                    # 优先级 = (短缺量 × 盈余量) / 距离
                    if dist > 0:
                        score = (d_val * s_val) / (dist + 1e-6)
                    else:
                        score = d_val * s_val + 1e6

                    if score > best_score:
                        best_score = score
                        best_i, best_j = i, j

            if best_i is None:
                break

            # 计算调度量（必须是正整数）
            max_transfer = min(
                surplus[best_i], deficit[best_j], cap_remaining)
            max_transfer = int(np.floor(max_transfer))  # 向下取整为整数

            if max_transfer <= 0:
                break

            # 记录调度
            schedules.append({
                'from': best_i,
                'to': best_j,
                'amount': max_transfer
            })

            round_count += 1

            # 更新
            surplus[best_i] -= max_transfer
            deficit[best_j] -= max_transfer
            cap_remaining -= max_transfer

            # 计算运输成本
            dist = self.config['dist_matrix'][best_i, best_j]
            trip_cost = (self.config['c_km'] * dist +
                         self.config['c_move']) * max_transfer
            total_cost += trip_cost

            # 更新站点列表
            surplus_stations = [(i, surplus[i])
                                for i in range(self.n) if surplus[i] > 1e-6]
            deficit_stations = [(j, deficit[j])
                                for j in range(self.n) if deficit[j] > 1e-6]

        # 调试输出
        if schedules and t == 6:
            print(
                f"    - 本轮调度: {round_count} 次行程, {sum(s['amount'] for s in schedules):.0f} 辆车")
            print(f"    - 运输成本: {total_cost:.2f} 元")

        return schedules, total_cost

    def apply_schedule(self, S_temp, schedules):
        """
        应用调度方案，更新库存
        """
        S_new = S_temp.copy()

        # 计算每个站点的净调入/调出
        out_flow = np.zeros(self.n)
        in_flow = np.zeros(self.n)

        for sch in schedules:
            out_flow[sch['from']] += sch['amount']
            in_flow[sch['to']] += sch['amount']

        S_new = S_new - out_flow + in_flow

        # 确保库存不超过容量（理论上调度后不应超过，但加保护）
        S_new = np.minimum(S_new, self.config['capacity'])
        S_new = np.maximum(S_new, 0)

        return S_new

    def calculate_penalty(self, S_end):
        """
        计算惩罚成本
        """
        # 确保惩罚系数是数值类型
        alpha = float(self.config['alpha'])
        beta = float(self.config['beta'])

        # 满桩惩罚（超过容量）
        over = np.maximum(0, S_end - self.config['capacity'])
        penalty_over = alpha * over

        # 空桩惩罚（低于安全库存）
        under = np.maximum(0, self.config['safe_inventory'] - S_end)
        penalty_under = beta * under

        return np.sum(penalty_over + penalty_under)

    def run_no_schedule(self):
        """
        运行无调度模拟
        """
        self.reset()
        total_penalty = 0
        total_fail_borrow = 0
        total_fail_return = 0
        total_full_station_count = 0  # 满桩发生次数
        total_empty_station_count = 0  # 空桩发生次数

        for t in range(24):
            # 借还过程
            S_after_borrow, S_temp, borrow_actual, return_actual, fail_borrow, fail_return = \
                self.simulate_borrow_return(t)

            # 无调度：临时库存即为下一小时库存
            self.S = S_temp.copy()

            # 记录
            self.records['inventory'].append(self.S.copy())
            self.records['borrow_success'].append(borrow_actual)
            self.records['return_success'].append(return_actual)
            self.records['fail_borrow'].append(fail_borrow)
            self.records['fail_return'].append(fail_return)

            # 计算惩罚（每小时结束时）
            penalty = self.calculate_penalty(self.S)
            self.records['penalty_over'].append(penalty)
            total_penalty += penalty
            total_fail_borrow += np.sum(fail_borrow)
            total_fail_return += np.sum(fail_return)

            # 统计满桩和空桩发生次数
            full_stations = np.sum(self.S >= self.config['capacity'])
            empty_stations = np.sum(self.S <= self.config['safe_inventory'])
            total_full_station_count += full_stations
            total_empty_station_count += empty_stations

        # 汇总结果
        results = {
            'total_penalty': total_penalty,
            'total_transport_cost': 0,
            'total_cost': total_penalty,
            'total_fail_borrow': total_fail_borrow,
            'total_fail_return': total_fail_return,
            'total_full_count': total_full_station_count,
            'total_empty_count': total_empty_station_count,
            'total_demand_out': np.sum(self.D_out),
            'total_demand_in': np.sum(self.D_in),
            'records': self.records
        }

        # 计算需求率
        total_demand = results['total_demand_out'] + results['total_demand_in']
        total_fail = total_fail_borrow + total_fail_return
        results['success_rate'] = (
            1 - total_fail / total_demand) * 100 if total_demand > 0 else 0

        return results

    def run_with_schedule(self):
        """
        运行带调度的模拟（增强贪心）
        """
        self.reset()
        total_penalty = 0
        total_transport_cost = 0
        total_fail_borrow = 0
        total_fail_return = 0
        total_full_station_count = 0  # 满桩发生次数
        total_empty_station_count = 0  # 空桩发生次数
        all_schedules = []

        for t in range(24):
            # 借还过程
            S_after_borrow, S_temp, borrow_actual, return_actual, fail_borrow, fail_return = \
                self.simulate_borrow_return(t)

            # 调度过程
            schedules, transport_cost = self.greedy_schedule_enhanced(
                S_temp, t, self.config['truck_capacity'])

            # 应用调度
            self.S = self.apply_schedule(S_temp, schedules)

            # 记录
            self.records['inventory'].append(self.S.copy())
            self.records['borrow_success'].append(borrow_actual)
            self.records['return_success'].append(return_actual)
            self.records['fail_borrow'].append(fail_borrow)
            self.records['fail_return'].append(fail_return)
            self.records['transport_cost'].append(transport_cost)
            self.records['schedules'].append(schedules)

            # 累计
            total_penalty += self.calculate_penalty(self.S)
            total_transport_cost += transport_cost
            total_fail_borrow += np.sum(fail_borrow)
            total_fail_return += np.sum(fail_return)
            all_schedules.extend(schedules)

            # 统计满桩和空桩发生次数
            full_stations = np.sum(self.S >= self.config['capacity'])
            empty_stations = np.sum(self.S <= self.config['safe_inventory'])
            total_full_station_count += full_stations
            total_empty_station_count += empty_stations

        # 统计调度信息
        total_schedule_trips = sum(len(h_sch)
                                   for h_sch in self.records['schedules'])
        total_schedule_vehicles = sum(
            sum(s['amount'] for s in h_sch) for h_sch in self.records['schedules'])

        print(
            f"\n  [调度统计] 全天共进行 {total_schedule_trips} 次调度, {total_schedule_vehicles:.0f} 辆车")

        # 汇总结果
        results = {
            'total_penalty': total_penalty,
            'total_transport_cost': total_transport_cost,
            'total_cost': total_penalty + total_transport_cost,
            'total_fail_borrow': total_fail_borrow,
            'total_fail_return': total_fail_return,
            'total_full_count': total_full_station_count,
            'total_empty_count': total_empty_station_count,
            'total_demand_out': np.sum(self.D_out),
            'total_demand_in': np.sum(self.D_in),
            'records': self.records,
            'all_schedules': all_schedules
        }

        # 计算需求率
        total_demand = results['total_demand_out'] + results['total_demand_in']
        total_fail = total_fail_borrow + total_fail_return
        results['success_rate'] = (
            1 - total_fail / total_demand) * 100 if total_demand > 0 else 0

        return results

    def get_station_details(self, station_idx):
        """
        获取单个站点的详细信息
        """
        station_info = {
            'id': self.config['idx_to_station'][station_idx],
            'capacity': self.config['capacity'][station_idx],
            'safe_inventory': self.config['safe_inventory'][station_idx],
            'init_inventory': self.config['init_inventory'][station_idx]
        }
        return station_info


def format_schedule_output(schedules, config):
    """
    格式化调度输出，显示站点编号
    """
    if not schedules:
        return "无调度"

    output = []
    for sch in schedules:
        from_id = config['idx_to_station'][sch['from']]
        to_id = config['idx_to_station'][sch['to']]
        output.append(f"{from_id} → {to_id}: {sch['amount']}辆")

    return output
