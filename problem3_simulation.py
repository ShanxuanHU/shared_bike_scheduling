"""
问题3模拟仿真
包含：无调度模拟、带离散时延的增强贪心调度算法
"""

import numpy as np

from problem3_config import clean_columns


class StationSimulator:
    def __init__(self, config, demand_df):
        """
        初始化模拟器
        config: 配置参数
        demand_df: 需求数据（预测值）
        """
        self.config = config
        self.n = config["n_stations"]
        self.max_delay = int(np.max(config["time_matrix"]))

        demand_df = clean_columns(demand_df)
        self.parse_demand(demand_df)
        self.reset()

    def parse_demand(self, demand_df):
        """
        解析需求数据，构建 D_out[i,t] 和 D_in[i,t]。
        """
        self.D_out = np.zeros((self.n, 24))
        self.D_in = np.zeros((self.n, 24))

        station_to_idx = self.config["station_to_idx"]
        demand_df = clean_columns(demand_df)

        for _, row in demand_df.iterrows():
            sid = row["站点编号"]
            hour = int(row["hour"])
            borrow = float(row["借出量"])
            return_val = float(row["归还量"])

            if sid in station_to_idx:
                i = station_to_idx[sid]
                self.D_out[i, hour] = borrow
                self.D_in[i, hour] = return_val

        if self.D_out.sum() == 0 and self.D_in.sum() == 0:
            print("  [警告] 需求数据为空！")

    def reset(self):
        """
        重置模拟状态。
        """
        self.S = self.config["init_inventory"].copy()
        horizon = 24 + self.max_delay + 1
        self.pending_arrivals = np.zeros((horizon, self.n))
        self.records = {
            "inventory": [],
            "borrow_success": [],
            "return_success": [],
            "fail_borrow": [],
            "fail_return": [],
            "penalty_over": [],
            "penalty_under": [],
            "transport_cost": [],
            "schedules": [],
            "arrivals": [],
        }

    def apply_due_arrivals(self, t):
        """
        将在当前时刻到站的调度车辆加入库存。
        """
        arrivals = self.pending_arrivals[t].copy()
        self.S = self.S + arrivals
        self.S = np.maximum(self.S, 0)
        return arrivals

    def simulate_borrow_return(self, t):
        """
        模拟借车和还车过程。
        """
        borrow_demand = self.D_out[:, t]
        borrow_actual = np.minimum(self.S, borrow_demand)
        fail_borrow = borrow_demand - borrow_actual

        s_after_borrow = self.S - borrow_actual

        return_demand = self.D_in[:, t]
        available_space = np.maximum(0, self.config["capacity"] - s_after_borrow)
        return_actual = np.minimum(available_space, return_demand)
        fail_return = return_demand - return_actual

        s_temp = s_after_borrow + return_actual
        return s_after_borrow, s_temp, borrow_actual, return_actual, fail_borrow, fail_return

    def calculate_surplus_deficit(self, s_temp, t):
        """
        计算过剩量和短缺量，并考虑已知在途车辆与未来需求。
        """
        safe_inv = self.config["safe_inventory"]
        surplus = np.maximum(0, s_temp - safe_inv)
        deficit = np.maximum(0, safe_inv - s_temp)

        future_deficit = np.zeros(self.n)
        for k in range(1, min(self.config["future_horizon"] + 1, 24 - t)):
            future_s = s_temp.copy()
            for step in range(1, k + 1):
                future_hour = t + step
                if future_hour >= 24:
                    break
                future_s = future_s + self.pending_arrivals[future_hour]
                net_demand = self.D_out[:, future_hour] - self.D_in[:, future_hour]
                future_s = future_s - net_demand
                future_s = np.maximum(future_s, 0)
            future_deficit += np.maximum(0, safe_inv - future_s)

        deficit_enhanced = deficit + self.config["future_lambda"] * future_deficit
        return surplus, deficit_enhanced

    def project_local_penalty(self, s_temp, t, from_idx, to_idx, transfer_amount, arrival_hour):
        """
        在两个相关站点上做局部滚动仿真，估计未来惩罚成本。
        """
        tracked_indices = [from_idx, to_idx]
        inventory = {
            from_idx: float(s_temp[from_idx] - transfer_amount),
            to_idx: float(s_temp[to_idx]),
        }

        if arrival_hour == t:
            inventory[to_idx] += transfer_amount

        total_penalty = 0.0
        for hour in range(t, 24):
            if hour > t:
                for idx in tracked_indices:
                    inventory[idx] += float(self.pending_arrivals[hour, idx])
                if hour == arrival_hour:
                    inventory[to_idx] += transfer_amount

                for idx in tracked_indices:
                    borrow_demand = float(self.D_out[idx, hour])
                    borrow_actual = min(inventory[idx], borrow_demand)
                    inventory[idx] -= borrow_actual

                    return_demand = float(self.D_in[idx, hour])
                    available_space = max(0.0, float(self.config["capacity"][idx]) - inventory[idx])
                    return_actual = min(available_space, return_demand)
                    inventory[idx] += return_actual

            for idx in tracked_indices:
                over = max(0.0, inventory[idx] - float(self.config["capacity"][idx]))
                under = max(0.0, float(self.config["safe_inventory"][idx]) - inventory[idx])
                total_penalty += float(self.config["alpha"]) * over
                total_penalty += float(self.config["beta"]) * under

        return total_penalty

    def estimate_transfer_net_gain(self, s_temp, t, from_idx, to_idx, amount):
        """
        估计一次调度在剩余时段内的净收益：
        净收益 = 惩罚下降 - 运输成本
        """
        delay = int(self.config["time_matrix"][from_idx, to_idx])
        arrival_hour = t + delay
        if arrival_hour >= 24:
            return float("-inf"), delay, arrival_hour

        no_transfer_penalty = self.project_local_penalty(
            s_temp, t, from_idx, to_idx, transfer_amount=0, arrival_hour=arrival_hour
        )
        with_transfer_penalty = self.project_local_penalty(
            s_temp, t, from_idx, to_idx, transfer_amount=amount, arrival_hour=arrival_hour
        )

        dist = self.config["dist_matrix"][from_idx, to_idx]
        trip_cost = self.config["c_km"] * dist + self.config["c_move"] * amount
        penalty_reduction = no_transfer_penalty - with_transfer_penalty
        net_gain = penalty_reduction - trip_cost
        return net_gain, delay, arrival_hour

    def greedy_schedule_enhanced(self, s_temp, t, truck_capacity):
        """
        增强贪心调度算法。
        返回：调度记录和运输成本。
        """
        if t < self.config["time_window_start"] or t >= self.config["time_window_end"]:
            return [], 0.0

        surplus, deficit = self.calculate_surplus_deficit(s_temp, t)
        surplus_stations = [(i, surplus[i]) for i in range(self.n) if surplus[i] > 1e-6]
        deficit_stations = [(j, deficit[j]) for j in range(self.n) if deficit[j] > 1e-6]

        if not surplus_stations or not deficit_stations:
            return [], 0.0

        schedules = []
        total_cost = 0.0

        while surplus_stations and deficit_stations:
            best_gain = float("-inf")
            best_plan = None

            for i, s_val in surplus_stations:
                for j, d_val in deficit_stations:
                    if i == j:
                        continue
                    max_transfer = int(np.floor(min(s_val, d_val, truck_capacity)))
                    if max_transfer <= 0:
                        continue

                    for amount in range(1, max_transfer + 1):
                        net_gain, delay, arrival_hour = self.estimate_transfer_net_gain(
                            s_temp, t, i, j, amount
                        )
                        if net_gain > best_gain:
                            best_gain = net_gain
                            best_plan = {
                                "from": i,
                                "to": j,
                                "amount": amount,
                                "travel_time": delay,
                                "arrival_hour": arrival_hour,
                                "net_gain": net_gain,
                            }

            if best_plan is None or best_gain <= 0:
                break

            schedules.append(
                {
                    "from": best_plan["from"],
                    "to": best_plan["to"],
                    "amount": best_plan["amount"],
                    "depart_hour": t,
                    "travel_time": best_plan["travel_time"],
                    "arrival_hour": best_plan["arrival_hour"],
                    "net_gain": round(best_plan["net_gain"], 2),
                }
            )

            surplus[best_plan["from"]] -= best_plan["amount"]
            deficit[best_plan["to"]] -= best_plan["amount"]
            dist = self.config["dist_matrix"][best_plan["from"], best_plan["to"]]
            trip_cost = self.config["c_km"] * dist + self.config["c_move"] * best_plan["amount"]
            total_cost += trip_cost

            surplus_stations = [(i, surplus[i]) for i in range(self.n) if surplus[i] > 1e-6]
            deficit_stations = [(j, deficit[j]) for j in range(self.n) if deficit[j] > 1e-6]

        return schedules, total_cost

    def apply_schedule(self, s_temp, schedules):
        """
        应用调度方案。
        出发站当小时立即扣减，目标站在 arrival_hour 到达入库。
        """
        s_new = s_temp.copy()
        out_flow = np.zeros(self.n)
        immediate_in_flow = np.zeros(self.n)

        for sch in schedules:
            out_flow[sch["from"]] += sch["amount"]
            arrival_hour = sch["arrival_hour"]
            if arrival_hour < len(self.pending_arrivals):
                if arrival_hour == sch["depart_hour"]:
                    immediate_in_flow[sch["to"]] += sch["amount"]
                else:
                    self.pending_arrivals[arrival_hour, sch["to"]] += sch["amount"]

        s_new = s_new - out_flow + immediate_in_flow
        s_new = np.maximum(s_new, 0)
        return s_new

    def calculate_penalty(self, s_end):
        """
        计算惩罚成本。
        """
        alpha = float(self.config["alpha"])
        beta = float(self.config["beta"])

        over = np.maximum(0, s_end - self.config["capacity"])
        under = np.maximum(0, self.config["safe_inventory"] - s_end)

        penalty_over = alpha * over
        penalty_under = beta * under
        return np.sum(penalty_over), np.sum(penalty_under)

    def run_no_schedule(self):
        """
        运行无调度模拟。
        """
        self.reset()
        total_penalty = 0
        total_fail_borrow = 0
        total_fail_return = 0
        total_full_station_count = 0
        total_empty_station_count = 0

        for t in range(24):
            arrivals = self.apply_due_arrivals(t)
            _, s_temp, borrow_actual, return_actual, fail_borrow, fail_return = self.simulate_borrow_return(t)
            self.S = s_temp.copy()

            self.records["arrivals"].append(arrivals)
            self.records["inventory"].append(self.S.copy())
            self.records["borrow_success"].append(borrow_actual)
            self.records["return_success"].append(return_actual)
            self.records["fail_borrow"].append(fail_borrow)
            self.records["fail_return"].append(fail_return)
            self.records["transport_cost"].append(0.0)
            self.records["schedules"].append([])

            penalty_over, penalty_under = self.calculate_penalty(self.S)
            self.records["penalty_over"].append(penalty_over)
            self.records["penalty_under"].append(penalty_under)
            total_penalty += penalty_over + penalty_under
            total_fail_borrow += np.sum(fail_borrow)
            total_fail_return += np.sum(fail_return)

            full_stations = np.sum(self.S >= self.config["capacity"])
            empty_stations = np.sum(self.S <= self.config["safe_inventory"])
            total_full_station_count += full_stations
            total_empty_station_count += empty_stations

        results = {
            "total_penalty": total_penalty,
            "total_transport_cost": 0,
            "total_cost": total_penalty,
            "total_fail_borrow": total_fail_borrow,
            "total_fail_return": total_fail_return,
            "total_full_count": total_full_station_count,
            "total_empty_count": total_empty_station_count,
            "total_demand_out": np.sum(self.D_out),
            "total_demand_in": np.sum(self.D_in),
            "records": self.records,
        }

        total_demand = results["total_demand_out"] + results["total_demand_in"]
        total_fail = total_fail_borrow + total_fail_return
        results["success_rate"] = (1 - total_fail / total_demand) * 100 if total_demand > 0 else 0
        return results

    def run_with_schedule(self):
        """
        运行带调度的模拟（增强贪心 + 离散时延）。
        """
        self.reset()
        total_penalty = 0
        total_transport_cost = 0
        total_fail_borrow = 0
        total_fail_return = 0
        total_full_station_count = 0
        total_empty_station_count = 0
        all_schedules = []

        for t in range(24):
            arrivals = self.apply_due_arrivals(t)
            _, s_temp, borrow_actual, return_actual, fail_borrow, fail_return = self.simulate_borrow_return(t)
            schedules, transport_cost = self.greedy_schedule_enhanced(
                s_temp, t, self.config["truck_capacity"]
            )
            self.S = self.apply_schedule(s_temp, schedules)

            self.records["arrivals"].append(arrivals)
            self.records["inventory"].append(self.S.copy())
            self.records["borrow_success"].append(borrow_actual)
            self.records["return_success"].append(return_actual)
            self.records["fail_borrow"].append(fail_borrow)
            self.records["fail_return"].append(fail_return)
            self.records["transport_cost"].append(transport_cost)
            self.records["schedules"].append(schedules)

            penalty_over, penalty_under = self.calculate_penalty(self.S)
            self.records["penalty_over"].append(penalty_over)
            self.records["penalty_under"].append(penalty_under)
            total_penalty += penalty_over + penalty_under
            total_transport_cost += transport_cost
            total_fail_borrow += np.sum(fail_borrow)
            total_fail_return += np.sum(fail_return)
            all_schedules.extend(schedules)

            full_stations = np.sum(self.S >= self.config["capacity"])
            empty_stations = np.sum(self.S <= self.config["safe_inventory"])
            total_full_station_count += full_stations
            total_empty_station_count += empty_stations

        total_schedule_trips = sum(len(h_sch) for h_sch in self.records["schedules"])
        total_schedule_vehicles = sum(
            sum(s["amount"] for s in h_sch) for h_sch in self.records["schedules"]
        )
        print(f"\n  [调度统计] 全天共进行 {total_schedule_trips} 次调度, {total_schedule_vehicles:.0f} 辆车")

        results = {
            "total_penalty": total_penalty,
            "total_transport_cost": total_transport_cost,
            "total_cost": total_penalty + total_transport_cost,
            "total_fail_borrow": total_fail_borrow,
            "total_fail_return": total_fail_return,
            "total_full_count": total_full_station_count,
            "total_empty_count": total_empty_station_count,
            "total_demand_out": np.sum(self.D_out),
            "total_demand_in": np.sum(self.D_in),
            "records": self.records,
            "all_schedules": all_schedules,
        }

        total_demand = results["total_demand_out"] + results["total_demand_in"]
        total_fail = total_fail_borrow + total_fail_return
        results["success_rate"] = (1 - total_fail / total_demand) * 100 if total_demand > 0 else 0
        return results

    def get_station_details(self, station_idx):
        """
        获取单个站点的详细信息。
        """
        return {
            "id": self.config["idx_to_station"][station_idx],
            "capacity": self.config["capacity"][station_idx],
            "safe_inventory": self.config["safe_inventory"][station_idx],
            "init_inventory": self.config["init_inventory"][station_idx],
        }


def format_schedule_output(schedules, config):
    """
    格式化调度输出，显示站点编号与到达时刻。
    """
    if not schedules:
        return "无调度"

    output = []
    for sch in schedules:
        from_id = config["idx_to_station"][sch["from"]]
        to_id = config["idx_to_station"][sch["to"]]
        output.append(
            f"{from_id} -> {to_id}: {sch['amount']}辆, "
            f"{sch['depart_hour']:02d}:00 发车, {sch['arrival_hour']:02d}:00 到达"
        )

    return output
