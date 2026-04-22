import pandas as pd
import matplotlib.pyplot as plt
from data_preprocess import preprocess_all
from model_train_predict import train_model, predict, evaluate

# ===== 数据 =====
df = preprocess_all()

# ===== 划分训练/测试 =====
train_df = df[df['date'] < '2025-04-13']
test_df = df[df['date'] == '2025-04-13']

# ===== 模型训练 =====
model_borrow, features = train_model(train_df, 'borrow_res')
model_return, _ = train_model(train_df, 'return_res')

# ===== 预测 =====
test_df['borrow_pred'] = predict(test_df, model_borrow, features, 'borrow_base')
test_df['return_pred'] = predict(test_df, model_return, features, 'return_base')

# ===== 评估 =====
mae_b, rmse_b, mape_b = evaluate(test_df['borrow'], test_df['borrow_pred'])
mae_r, rmse_r, mape_r = evaluate(test_df['return'], test_df['return_pred'])

print("Borrow:", mae_b, rmse_b, mape_b)
print("Return:", mae_r, rmse_r, mape_r)

# ===== 绘图（改进版） =====
import os

# 创建保存目录
os.makedirs("figures", exist_ok=True)

stations = ['S002','S012','S023']

for sid in stations:
    tmp = test_df[test_df['station_id']==sid]

    # ===== Borrow 图 =====
    plt.figure(figsize=(10,5))

    # 灰色背景参考线（横线）
    for y in range(int(tmp['borrow'].min()), int(tmp['borrow'].max())+5, 5):
        plt.axhline(y=y, linestyle='--', linewidth=0.5, color='gray', alpha=0.3)

    # 真实值
    plt.plot(
        tmp['hour'],
        tmp['borrow'],
        marker='o',
        markersize=5,
        linewidth=2,
        label='Actual'
    )

    # 预测值
    plt.plot(
        tmp['hour'],
        tmp['borrow_pred'],
        marker='s',
        markersize=5,
        linewidth=2,
        linestyle='--',
        label='Predicted'
    )

    plt.title(f'Station {sid} - Borrow Demand', fontsize=14)
    plt.xlabel('Hour of Day')
    plt.ylabel('Borrow Volume')

    plt.xticks(range(0,24,2))
    plt.grid(False)
    plt.legend()

    plt.tight_layout()

    # 保存图片
    plt.savefig(f'figures/{sid}_borrow.png', dpi=300)
    plt.close()

    # ===== Return 图 =====
    plt.figure(figsize=(10,5))

    for y in range(int(tmp['return'].min()), int(tmp['return'].max())+5, 5):
        plt.axhline(y=y, linestyle='--', linewidth=0.5, color='gray', alpha=0.3)

    plt.plot(
        tmp['hour'],
        tmp['return'],
        marker='o',
        markersize=5,
        linewidth=2,
        label='Actual'
    )

    plt.plot(
        tmp['hour'],
        tmp['return_pred'],
        marker='s',
        markersize=5,
        linewidth=2,
        linestyle='--',
        label='Predicted'
    )

    plt.title(f'Station {sid} - Return Demand', fontsize=14)
    plt.xlabel('Hour of Day')
    plt.ylabel('Return Volume')

    plt.xticks(range(0,24,2))
    plt.grid(False)
    plt.legend()

    plt.tight_layout()

    plt.savefig(f'figures/{sid}_return.png', dpi=300)
    plt.close()