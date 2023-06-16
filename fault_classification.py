import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

def get_fault_dict(df_fault_result):
    fault_dict = dict()
    list_fault_class_code = list(df_fault_result["fault_class_code"])
    list_fault_class = list(df_fault_result["fault_class"])
    for i in range(len(list_fault_class_code)):
        fault_dict[list_fault_class_code[i]] = list_fault_class[i]
    fault_dict[0] = "正常"
    return fault_dict


df_total_result = pd.read_csv("normal_wd_sd_gps.csv")


feature_columns = ['lcsd', 'xfwd', 'E', 'N']
label_columns = ['fault_class_code']
df_feature_result = df_total_result[feature_columns]
df_feature_result['fault_class_code'] = 0
# print(df_feature_result)
# print(df_feature_result.values)
# print(type(df_feature_result.values))

df_fault_result = pd.read_csv("fault_wd_sd_gps.csv")
fault_dict = get_fault_dict(df_fault_result)
df_fault_result = df_fault_result[feature_columns + label_columns]

df_fault_result = df_fault_result.sample(frac=1)


df_feature_result = df_feature_result.sample(frac=0.01)
print(len(df_feature_result))
df_total_result = pd.concat([df_feature_result, df_fault_result])

df_total_result = df_total_result.sample(frac=1)

print(df_total_result)

tdata = df_total_result[feature_columns].values
target = df_total_result[label_columns].values
X_train, X_test, y_train, y_test = train_test_split(tdata, target, test_size=0.2)
# scaler = StandardScaler()
# scaler.fit(X_train)
#
# x_train, x_test = scaler.transform(X_train), scaler.transform(X_test)

model = RandomForestClassifier(max_depth=10)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_pred, y_test)
print(acc)
print(len(y_pred))

