"""
为问题3生成工作日预测数据

问题2（main.py）：预测 2025-04-13（周日）的需求
问题3需要：选择一个工作日（非周末），预测其借还需求

此脚本选择 2025-04-11（周五）作为目标工作日，用第二题的模型进行预测
"""

import pandas as pd
import numpy as np
from data_preprocess import preprocess_all
from model_train_predict import train_model, predict, evaluate


def generate_workday_predictions():
    """
    为工作日（2025-04-11 周五）生成预测数据，供问题3使用
    """
    # 加载并预处理数据
    print("[第一步] 加载和预处理数据...")
    df = preprocess_all()

    # 划分训练集：2025-04-13之前的所有数据（包括工作日和周末）
    # 这样可以让模型学习到工作日的特征（is_weekday=1）
    train_df = df[df['date'] < '2025-04-13']

    # 选择工作日 2025-04-11（周五）作为预测目标
    workday_date = '2025-04-11'
    test_df = df[df['date'] == workday_date].copy()

    if len(test_df) == 0:
        print(f"  警告: 数据中不存在 {workday_date}，使用 2025-04-09（周三）")
        workday_date = '2025-04-09'
        test_df = df[df['date'] == workday_date].copy()

    print(f"  ✓ 选择 {workday_date} 作为目标工作日进行预测")
    print(f"    - 训练集行数: {len(train_df)}")
    print(f"    - 测试集行数: {len(test_df)}")

    # 训练模型
    print("\n[第二步] 训练XGBoost模型...")
    model_borrow, features = train_model(train_df, 'borrow_res')
    model_return, _ = train_model(train_df, 'return_res')
    print(f"  ✓ 模型训练完成")
    print(f"    - 使用特征数: {len(features)}")
    print(f"    - 特征列表: {features}")

    # 预测
    print("\n[第三步] 进行需求预测...")
    test_df['borrow_pred'] = predict(
        test_df, model_borrow, features, 'borrow_base')
    test_df['return_pred'] = predict(
        test_df, model_return, features, 'return_base')

    # 确保预测值非负
    test_df['borrow_pred'] = test_df['borrow_pred'].clip(lower=0)
    test_df['return_pred'] = test_df['return_pred'].clip(lower=0)

    print(f"  ✓ 预测完成")

    # 评估准确性
    print("\n[第四步] 评估预测精度...")
    mae_b, rmse_b, mape_b = evaluate(test_df['borrow'], test_df['borrow_pred'])
    mae_r, rmse_r, mape_r = evaluate(test_df['return'], test_df['return_pred'])
    print(f"  借出量 - MAE: {mae_b:.2f}, RMSE: {rmse_b:.2f}, MAPE: {mape_b:.2%}")
    print(f"  归还量 - MAE: {mae_r:.2f}, RMSE: {rmse_r:.2f}, MAPE: {mape_r:.2%}")

    # 生成问题3需要的预测文件
    print("\n[第五步] 生成问题3的预测数据文件...")

    # 创建问题3所需的格式：日期、小时、站点编号、借出量、归还量
    predictions_for_problem3 = test_df[[
        'date',
        'hour',
        'station_id',
        'borrow_pred',
        'return_pred'
    ]].copy()

    # 重命名为中文列名，保持与附件2一致
    predictions_for_problem3 = predictions_for_problem3.rename(columns={
        'date': '日期',
        'hour': '小时(0-23)',
        'station_id': '站点编号',
        'borrow_pred': '借出量',
        'return_pred': '归还量'
    })

    # 四舍五入为整数（实际车辆数）
    predictions_for_problem3['借出量'] = predictions_for_problem3['借出量'].round(
        0).astype(int)
    predictions_for_problem3['归还量'] = predictions_for_problem3['归还量'].round(
        0).astype(int)

    # 保存文件：使用工作日日期命名
    output_filename = f'predictions_{workday_date}.csv'
    predictions_for_problem3.to_csv(
        output_filename, index=False, encoding='utf-8-sig')

    print(f"  ✓ 已生成 {output_filename}")
    print(f"    - 数据行数: {len(predictions_for_problem3)}")
    print(f"    - 覆盖站点: {predictions_for_problem3['站点编号'].nunique()} 个")
    print(f"    - 时间范围: 0-23 小时")

    # 打印统计信息
    print("\n[第六步] 预测数据统计...")
    print(f"  24小时总预测借出量: {predictions_for_problem3['借出量'].sum():.0f} 辆")
    print(f"  24小时总预测归还量: {predictions_for_problem3['归还量'].sum():.0f} 辆")

    # 打印每小时的数据摘要
    hourly_stats = predictions_for_problem3.groupby(
        '小时(0-23)')[['借出量', '归还量']].sum()
    print("\n  每小时预测统计（前10小时）:")
    print(hourly_stats.head(10))

    print(f"\n✅ 问题3的预测数据已生成完毕！")
    print(f"   problem3_solution.py 将自动使用 {output_filename}")

    return predictions_for_problem3, workday_date


if __name__ == "__main__":
    print("="*80)
    print("为问题3生成工作日预测数据")
    print("="*80 + "\n")

    predictions, workday = generate_workday_predictions()

    print("\n" + "="*80)
    print(f"问题3 配置说明")
    print("="*80)
    print(f"""
问题2（main.py）：
  ✓ 目标日期: 2025-04-13（周日）
  ✓ 输出文件: 无需保存预测文件
  ✓ 用途: 需求预测模型评估

问题3（problem3_solution.py）：
  ✓ 目标日期: {workday}（周五，工作日）
  ✓ 预测文件: predictions_{workday}.csv
  ✓ 用途: 调度优化方案对比
  ✓ 对比指标:
    - 总费用（惩罚成本 + 运输成本）
    - 满桩/空桩发生次数
    - 用户无法借车/还车的次数
    - 用户需求满足率

执行步骤：
  1. 运行此脚本生成预测数据: python problem2_workday_predict.py
  2. 运行问题3求解: python problem3_solution.py
""")
