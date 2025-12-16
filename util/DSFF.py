import numpy as np
import pandas as pd

def read_csv(path):
    df = pd.read_csv(path)
    return df

def calculate_weight(df, features):
    weights = {}
    for feat in features:
        # pcc static
        r1 = abs(np.corrcoef(df[feat], df['label'])[0, 1])

        # dynamic
        diff_feat = df[feat].diff().fillna(0)
        r2 = abs(np.corrcoef(diff_feat, df['label'])[0, 1])

        weights[feat] = r1 * r2  # 融合动态信息

    return weights

if __name__ == '__main__':
    path = "new_timeseries_feat.csv"
    df = read_csv(path)
    features = ['opposite_count', 'opposite_unique', 'call_dur_sum', 'sms_count', 'flow_sum', 'sms_nunique']
    df[features] = df[features].fillna(0)

    weights = calculate_weight(df, features)
    # 归一化
    max_w = max(weights.values())
    weights = {k: v / max_w for k, v in weights.items()}
    print("权重（融合动态变化率相关性）：")
    print(weights)


