from pathlib import Path

import matplotlib.pyplot as plt

from problem2_data_preprocess import BASE_DIR, FORECAST_FEATURES, build_model_datasets
from problem2_model_train_predict import evaluate, predict, train_model


TARGET_DATE = "2025-04-13"
OUTPUT_DIR = BASE_DIR / "outputs" / "problem2"
FIGURE_DIR = OUTPUT_DIR / "figures"


def main():
    train_df, test_df, features = build_model_datasets(TARGET_DATE)

    model_borrow, _ = train_model(train_df, "borrow_res", features)
    model_return, _ = train_model(train_df, "return_res", features)

    test_df = test_df.copy()
    test_df["borrow_pred"] = predict(test_df, model_borrow, features, "borrow_base").clip(lower=0)
    test_df["return_pred"] = predict(test_df, model_return, features, "return_base").clip(lower=0)

    mae_b, rmse_b, mape_b = evaluate(test_df["borrow"], test_df["borrow_pred"])
    mae_r, rmse_r, mape_r = evaluate(test_df["return"], test_df["return_pred"])

    print(f"Target date: {TARGET_DATE}")
    print(f"Borrow: MAE={mae_b:.3f}, RMSE={rmse_b:.3f}, MAPE={mape_b:.3%}")
    print(f"Return: MAE={mae_r:.3f}, RMSE={rmse_r:.3f}, MAPE={mape_r:.3%}")
    print(f"Features: {FORECAST_FEATURES}")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    for sid in ["S002", "S012", "S023"]:
        tmp = test_df[test_df["station_id"] == sid].copy()
        if tmp.empty:
            continue

        save_series_plot(
            tmp["hour"],
            tmp["borrow"],
            tmp["borrow_pred"],
            f"Station {sid} - Borrow Demand",
            "Borrow Volume",
            FIGURE_DIR / f"{sid}_borrow.png",
        )
        save_series_plot(
            tmp["hour"],
            tmp["return"],
            tmp["return_pred"],
            f"Station {sid} - Return Demand",
            "Return Volume",
            FIGURE_DIR / f"{sid}_return.png",
        )


def save_series_plot(hours, actual, predicted, title, y_label, output_path):
    plt.figure(figsize=(10, 5))

    for y in range(int(actual.min()), int(actual.max()) + 5, 5):
        plt.axhline(y=y, linestyle="--", linewidth=0.5, color="gray", alpha=0.3)

    plt.plot(hours, actual, marker="o", markersize=5, linewidth=2, label="Actual")
    plt.plot(
        hours,
        predicted,
        marker="s",
        markersize=5,
        linewidth=2,
        linestyle="--",
        label="Predicted",
    )

    plt.title(title, fontsize=14)
    plt.xlabel("Hour of Day")
    plt.ylabel(y_label)
    plt.xticks(range(0, 24, 2))
    plt.grid(False)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
