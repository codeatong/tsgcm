import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import DSFF

'''
DWED method implement
'''
def weighted_distance_grouping(data, weight_dict=None, save_prefix="grouped"):

    features = [col for col in data.columns if col not in ['label', 'phone_no_m']]
    X = data[features].copy()
    y = data['label']
    print(features)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=features)


    if weight_dict is None:
        weight_dict = {f: 1.0 for f in features}
    weights = np.array([weight_dict.get(f, 1.0) for f in features])

    # 4. 计算两个中心（加权平均）
    center_0 = np.average(X_scaled[y == 0], axis=0, weights=None)
    center_1 = np.average(X_scaled[y == 1], axis=0, weights=None)

    print("Center_0:", center_0)
    print("Center_1:", center_1)

    # 5. 计算加权欧氏距离
    def weighted_euclidean(a, b, w):
        return np.sqrt(np.sum(w * (a - b) ** 2, axis=1))

    dist_to_0 = weighted_euclidean(X_scaled.values, center_0, weights)
    dist_to_1 = weighted_euclidean(X_scaled.values, center_1, weights)

    # 6. 分组
    group_labels = np.where(dist_to_0 < dist_to_1, 0, 1)

    # 7. 保存结果
    data['group'] = group_labels
    data[data['group'] == 0].to_csv(f"{save_prefix}_near_label0.csv", index=False)
    data[data['group'] == 1].to_csv(f"{save_prefix}_near_label1.csv", index=False)

    print(f"save: {save_prefix}_near_label0.csv and {save_prefix}_near_label1.csv")
    return data

# ===== 示例用法 =====
df = pd.read_csv("new_timeseries_feat.csv")  # yours
FEATURE_COLS = ['opposite_count', 'opposite_unique', 'sms_count', 'call_dur_sum', 'flow_sum','sms_nunique']
df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)
weights = DSFF.calculate_weight(df, FEATURE_COLS)
# weight_dict = {'opposite_count': weights["opposite_count"], 'opposite_unique': weights["opposite_unique"],
#                'sms_count': weights["sms_count"], 'call_dur_sum': weights["call_dur_sum"], 'flow_sum': weights["flow_sum"]
#                 ,'sms_nunique': weights["sms_nunique"] }

result_df = weighted_distance_grouping(df, weight_dict=weights, save_prefix="user_groups")

