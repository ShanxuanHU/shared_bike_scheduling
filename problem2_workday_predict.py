from pathlib import Path

from data_preprocess import BASE_DIR, build_model_datasets
from model_train_predict import evaluate, predict, train_model


DEFAULT_WORKDAY = "2025-04-11"
OUTPUT_DIR = BASE_DIR / "outputs" / "problem2"


def generate_workday_predictions(target_date=DEFAULT_WORKDAY):
    """
    为问题3生成工作日预测数据。
    """
    print("[第一步] 加载并构造训练/测试数据...")
    train_df, test_df, features = build_model_datasets(target_date)
    print(f"  [OK] 目标日期: {target_date}")
    print(f"  [OK] 训练样本数: {len(train_df)}")
    print(f"  [OK] 测试样本数: {len(test_df)}")

    print("\n[第二步] 训练 XGBoost 模型...")
    model_borrow, _ = train_model(train_df, "borrow_res", features)
    model_return, _ = train_model(train_df, "return_res", features)
    print(f"  [OK] 模型训练完成，特征数: {len(features)}")

    print("\n[第三步] 生成 24 小时需求预测...")
    test_df = test_df.copy()
    test_df["borrow_pred"] = predict(test_df, model_borrow, features, "borrow_base").clip(lower=0)
    test_df["return_pred"] = predict(test_df, model_return, features, "return_base").clip(lower=0)
    print("  [OK] 预测完成")

    print("\n[第四步] 评估预测精度...")
    mae_b, rmse_b, mape_b = evaluate(test_df["borrow"], test_df["borrow_pred"])
    mae_r, rmse_r, mape_r = evaluate(test_df["return"], test_df["return_pred"])
    print(f"  借出量 - MAE: {mae_b:.2f}, RMSE: {rmse_b:.2f}, MAPE: {mape_b:.2%}")
    print(f"  归还量 - MAE: {mae_r:.2f}, RMSE: {rmse_r:.2f}, MAPE: {mape_r:.2%}")

    print("\n[第五步] 生成问题3输入文件...")
    predictions_for_problem3 = test_df[
        ["date", "hour", "station_id", "borrow_pred", "return_pred"]
    ].copy()
    predictions_for_problem3 = predictions_for_problem3.rename(
        columns={
            "date": "日期",
            "hour": "小时(0-23)",
            "station_id": "站点编号",
            "borrow_pred": "借出量",
            "return_pred": "归还量",
        }
    )
    predictions_for_problem3["借出量"] = predictions_for_problem3["借出量"].round(0).astype(int)
    predictions_for_problem3["归还量"] = predictions_for_problem3["归还量"].round(0).astype(int)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"predictions_{target_date}.csv"
    predictions_for_problem3.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"  [OK] 已保存: {output_path}")
    print(f"  [OK] 覆盖站点: {predictions_for_problem3['站点编号'].nunique()} 个")
    print(f"  [OK] 总借出预测: {predictions_for_problem3['借出量'].sum():.0f} 辆")
    print(f"  [OK] 总归还预测: {predictions_for_problem3['归还量'].sum():.0f} 辆")

    return predictions_for_problem3, target_date


if __name__ == "__main__":
    print("=" * 80)
    print("为问题3生成工作日预测数据")
    print("=" * 80)
    generate_workday_predictions()
