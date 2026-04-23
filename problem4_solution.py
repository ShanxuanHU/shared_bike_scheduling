from pathlib import Path

import pandas as pd

from problem3_config import (
    BASE_DIR,
    build_config,
    haversine_distance,
    load_data,
    load_demand_predictions,
)
from problem3_simulation import StationSimulator


OUTPUT_DIR = BASE_DIR / "outputs" / "problem4"


NEW_STATION_GROUPS = [
    {
        "station_id": "N101",
        "station_name": "宁大教学生活扩展站",
        "capacity": 30,
        "target_share": 0.22,
        "donors": ["S002", "S003", "S004"],
    },
    {
        "station_id": "N102",
        "station_name": "诺大教学宿舍扩展站",
        "capacity": 30,
        "target_share": 0.22,
        "donors": ["S005", "S006", "S007", "S008"],
    },
    {
        "station_id": "N103",
        "station_name": "天一广场商务接驳站",
        "capacity": 35,
        "target_share": 0.18,
        "donors": ["S012", "S017", "S018", "S025"],
    },
    {
        "station_id": "N104",
        "station_name": "高新区研发走廊站",
        "capacity": 30,
        "target_share": 0.18,
        "donors": ["S013", "S014", "S015", "S016"],
    },
    {
        "station_id": "N105",
        "station_name": "镇海象山路接驳站",
        "capacity": 25,
        "target_share": 0.15,
        "donors": ["S020", "S030"],
    },
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stations, params = load_data()
    demand_df = load_demand_predictions()

    print("=" * 80)
    print("问题4：模型推广与灵敏度分析")
    print("=" * 80)

    demand_summary = build_station_demand_summary(demand_df)
    new_station_plan = propose_new_stations(stations, demand_summary)
    expanded_stations = append_new_stations(stations, new_station_plan)
    expanded_demand = redistribute_demand(demand_df, stations, new_station_plan, demand_summary)

    print("\n[第1步] 完成新增5站点方案构造")
    print(f"  - 原站点数: {len(stations)}")
    print(f"  - 扩展后站点数: {len(expanded_stations)}")

    base_config = build_config(stations, params, output_dir=OUTPUT_DIR)
    expanded_config = build_config(expanded_stations, params, output_dir=OUTPUT_DIR)

    print("\n[第2步] 运行原始网络与扩展网络仿真")
    base_results = run_both_modes(base_config, demand_df)
    expanded_results = run_both_modes(expanded_config, expanded_demand)

    scenario_comparison = build_scenario_comparison(base_results, expanded_results)

    print("\n[第3步] 运行敏感性分析")
    sensitivity_df = run_sensitivity_analysis(expanded_stations, params, expanded_demand)

    save_outputs(new_station_plan, expanded_demand, scenario_comparison, sensitivity_df)
    write_summary_markdown(new_station_plan, scenario_comparison, sensitivity_df)

    print("\n[结果摘要]")
    print(scenario_comparison.to_string(index=False))
    print("\n已生成问题4结果到 outputs/problem4/")


def build_station_demand_summary(demand_df):
    summary = (
        demand_df.groupby("站点编号")[["借出量", "归还量"]]
        .sum()
        .rename(columns={"借出量": "total_borrow", "归还量": "total_return"})
    )
    summary["total_demand"] = summary["total_borrow"] + summary["total_return"]
    return summary


def propose_new_stations(stations, demand_summary):
    plan_rows = []
    station_index = stations.set_index("站点编号")

    for group in NEW_STATION_GROUPS:
        donors = group["donors"]
        donor_rows = station_index.loc[donors].reset_index()
        donor_rows = donor_rows.merge(
            demand_summary.reset_index(), on="站点编号", how="left"
        )

        weights = donor_rows["total_demand"].fillna(1).clip(lower=1)
        lon = (donor_rows["经度"] * weights).sum() / weights.sum()
        lat = (donor_rows["纬度"] * weights).sum() / weights.sum()
        current_inventory = round(group["capacity"] * 0.5)

        donor_descriptions = []
        for donor in donors:
            dist = haversine_distance(
                float(station_index.loc[donor, "经度"]),
                float(station_index.loc[donor, "纬度"]),
                lon,
                lat,
            )
            donor_descriptions.append(f"{donor}({dist:.2f}km)")

        plan_rows.append(
            {
                "站点编号": group["station_id"],
                "站点名称": group["station_name"],
                "经度": round(lon, 6),
                "纬度": round(lat, 6),
                "总桩位数": group["capacity"],
                "当前库存量": current_inventory,
                "需求承接比例": group["target_share"],
                "承接来源站点": ",".join(donors),
                "最近来源站距离说明": "; ".join(donor_descriptions),
            }
        )

    return pd.DataFrame(plan_rows)


def append_new_stations(stations, new_station_plan):
    appended = pd.concat(
        [
            stations.copy(),
            new_station_plan[["站点编号", "站点名称", "经度", "纬度", "总桩位数", "当前库存量"]],
        ],
        ignore_index=True,
    )
    return appended


def redistribute_demand(demand_df, stations, new_station_plan, demand_summary):
    expanded = demand_df.copy()
    expanded["借出量"] = expanded["借出量"].astype(float)
    expanded["归还量"] = expanded["归还量"].astype(float)
    per_new_station_frames = []
    station_index = stations.set_index("站点编号")

    for _, row in new_station_plan.iterrows():
        new_station_id = row["站点编号"]
        donors = row["承接来源站点"].split(",")
        target_share = float(row["需求承接比例"])

        donor_weights = []
        donor_info = []
        for donor in donors:
            donor_demand = float(demand_summary.loc[donor, "total_demand"])
            dist = haversine_distance(
                float(station_index.loc[donor, "经度"]),
                float(station_index.loc[donor, "纬度"]),
                float(row["经度"]),
                float(row["纬度"]),
            )
            weight = donor_demand / max(dist, 0.3)
            donor_weights.append(weight)
            donor_info.append((donor, weight))

        total_weight = sum(donor_weights)
        donor_ratios = {
            donor: target_share * weight / total_weight for donor, weight in donor_info
        }

        new_rows = []
        for donor, donor_ratio in donor_ratios.items():
            mask = expanded["站点编号"] == donor
            donor_slice = expanded.loc[mask].copy()

            moved_borrow = donor_slice["借出量"] * donor_ratio
            moved_return = donor_slice["归还量"] * donor_ratio

            expanded.loc[mask, "借出量"] = donor_slice["借出量"] - moved_borrow
            expanded.loc[mask, "归还量"] = donor_slice["归还量"] - moved_return

            donor_slice["站点编号"] = new_station_id
            donor_slice["借出量"] = moved_borrow
            donor_slice["归还量"] = moved_return
            new_rows.append(donor_slice[["date", "hour", "站点编号", "借出量", "归还量"]])

        merged_new_rows = pd.concat(new_rows, ignore_index=True)
        merged_new_rows = (
            merged_new_rows.groupby(["date", "hour", "站点编号"], as_index=False)[["借出量", "归还量"]]
            .sum()
        )
        per_new_station_frames.append(merged_new_rows)

    expanded = expanded[["date", "hour", "站点编号", "借出量", "归还量"]].copy()
    if per_new_station_frames:
        expanded = pd.concat([expanded] + per_new_station_frames, ignore_index=True)

    expanded[["借出量", "归还量"]] = expanded[["借出量", "归还量"]].clip(lower=0)
    expanded = expanded.sort_values(["站点编号", "date", "hour"]).reset_index(drop=True)
    return expanded


def run_both_modes(config, demand_df):
    no_schedule = StationSimulator(config, demand_df).run_no_schedule()
    with_schedule = StationSimulator(config, demand_df).run_with_schedule()
    return {"no_schedule": no_schedule, "with_schedule": with_schedule}


def build_scenario_comparison(base_results, expanded_results):
    rows = []
    scenarios = [
        ("原网络-无调度", base_results["no_schedule"]),
        ("原网络-有调度", base_results["with_schedule"]),
        ("扩展网络-无调度", expanded_results["no_schedule"]),
        ("扩展网络-有调度", expanded_results["with_schedule"]),
    ]

    for name, result in scenarios:
        rows.append(
            {
                "场景": name,
                "总费用(元)": round(result["total_cost"], 2),
                "惩罚成本(元)": round(result["total_penalty"], 2),
                "运输成本(元)": round(result["total_transport_cost"], 2),
                "满桩发生次数": int(result["total_full_count"]),
                "空桩发生次数": int(result["total_empty_count"]),
                "无法借车次数": int(result["total_fail_borrow"]),
                "无法还车次数": int(result["total_fail_return"]),
                "需求满足率(%)": round(result["success_rate"], 2),
            }
        )

    return pd.DataFrame(rows)


def run_sensitivity_analysis(expanded_stations, params, expanded_demand):
    experiments = []
    parameter_mapping = {
        "每公里运输成本": "c_km",
        "每辆单次搬运成本": "c_move",
        "满桩惩罚系数": "alpha",
        "空桩惩罚系数": "beta",
    }
    base_values = {
        "每公里运输成本": float(params["每公里运输成本"]),
        "每辆单次搬运成本": float(params["每辆单次搬运成本"]),
        "满桩惩罚系数": float(params["满桩惩罚系数"]),
        "空桩惩罚系数": float(params["空桩惩罚系数"]),
    }

    for param_name in parameter_mapping:
        for multiplier in [0.6, 0.8, 1.0, 1.2, 1.4]:
            adjusted_params = params.copy()
            adjusted_params[param_name] = float(params[param_name]) * multiplier
            config = build_config(expanded_stations, adjusted_params, output_dir=OUTPUT_DIR)
            result = StationSimulator(config, expanded_demand).run_with_schedule()
            schedules = result["records"]["schedules"]
            schedule_trips = sum(len(hour_schedules) for hour_schedules in schedules)
            moved_vehicles = sum(
                sum(schedule["amount"] for schedule in hour_schedules)
                for hour_schedules in schedules
            )
            experiments.append(
                {
                    "参数": param_name,
                    "基准值": base_values[param_name],
                    "倍率": multiplier,
                    "参数取值": round(float(adjusted_params[param_name]), 4),
                    "总费用(元)": round(result["total_cost"], 2),
                    "惩罚成本(元)": round(result["total_penalty"], 2),
                    "运输成本(元)": round(result["total_transport_cost"], 2),
                    "调度次数": int(schedule_trips),
                    "调度车辆数": int(moved_vehicles),
                    "无法借车次数": int(result["total_fail_borrow"]),
                    "无法还车次数": int(result["total_fail_return"]),
                    "需求满足率(%)": round(result["success_rate"], 2),
                }
            )

    return pd.DataFrame(experiments)


def save_outputs(new_station_plan, expanded_demand, scenario_comparison, sensitivity_df):
    new_station_plan.to_csv(OUTPUT_DIR / "problem4_new_station_plan.csv", index=False, encoding="utf-8-sig")
    expanded_demand.to_csv(OUTPUT_DIR / "problem4_expanded_demand.csv", index=False, encoding="utf-8-sig")
    scenario_comparison.to_csv(OUTPUT_DIR / "problem4_scenario_comparison.csv", index=False, encoding="utf-8-sig")
    sensitivity_df.to_csv(OUTPUT_DIR / "problem4_sensitivity_results.csv", index=False, encoding="utf-8-sig")


def write_summary_markdown(new_station_plan, scenario_comparison, sensitivity_df):
    expanded_with = scenario_comparison[scenario_comparison["场景"] == "扩展网络-有调度"].iloc[0]
    base_with = scenario_comparison[scenario_comparison["场景"] == "原网络-有调度"].iloc[0]
    cost_change = base_with["总费用(元)"] - expanded_with["总费用(元)"]
    rate_change = expanded_with["需求满足率(%)"] - base_with["需求满足率(%)"]
    if cost_change >= 0:
        cost_line = f"- 相比原网络有调度方案，总费用下降：{cost_change:.2f} 元。"
    else:
        cost_line = f"- 相比原网络有调度方案，总费用上升：{abs(cost_change):.2f} 元。"

    best_cost = sensitivity_df.sort_values("总费用(元)").iloc[0]
    worst_cost = sensitivity_df.sort_values("总费用(元)", ascending=False).iloc[0]
    response_rows = []
    for param_name, group in sensitivity_df.groupby("参数"):
        ordered = group.sort_values("倍率")
        low = ordered.iloc[0]
        high = ordered.iloc[-1]
        response_rows.append(
            {
                "参数": param_name,
                "总费用变化": high["总费用(元)"] - low["总费用(元)"],
                "调度次数变化": high["调度次数"] - low["调度次数"],
                "调度车辆变化": high["调度车辆数"] - low["调度车辆数"],
            }
        )
    response_df = pd.DataFrame(response_rows).sort_values("总费用变化", key=lambda s: s.abs(), ascending=False)
    most_sensitive = response_df.iloc[0]

    lines = [
        "# 问题4结果摘要",
        "",
        "## 新增站点方案",
        "",
        f"新增站点数：{len(new_station_plan)} 个。",
        "新增站点采用“局部高需求片区增设接驳站”的思路，并通过分流相邻既有站点的预测需求，使原问题3模型可直接推广到35站网络。",
        "",
        "## 扩展网络效果",
        "",
        cost_line,
        f"- 相比原网络有调度方案，需求满足率变化：{rate_change:.2f} 个百分点。",
        f"- 扩展网络有调度总费用：{expanded_with['总费用(元)']:.2f} 元。",
        f"- 扩展网络有调度需求满足率：{expanded_with['需求满足率(%)']:.2f}%。",
        "",
        "## 敏感性分析结论",
        "",
        f"- 最低总费用情景：{best_cost['参数']} × {best_cost['倍率']:.1f}，总费用 {best_cost['总费用(元)']:.2f} 元。",
        f"- 最高总费用情景：{worst_cost['参数']} × {worst_cost['倍率']:.1f}，总费用 {worst_cost['总费用(元)']:.2f} 元。",
        f"- 从 0.6 倍到 1.4 倍的区间看，最敏感参数是{most_sensitive['参数']}，总费用变化 {most_sensitive['总费用变化']:.2f} 元。",
        "- 提高运输成本参数会压缩调度积极性，通常表现为调度次数和调度车辆数下降。",
        "- 提高惩罚系数会提升调度激励，尤其空桩惩罚系数上升时，模型更倾向于提前补车。",
        "- 若满桩惩罚系数变化引起的方案变化很小，可解释为当前测试日的主要矛盾是空桩而不是满桩。",
    ]

    (OUTPUT_DIR / "problem4_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
