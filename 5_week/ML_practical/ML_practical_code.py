import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy.linalg as lin
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV



# 0 - low cost
# 1 - medium cost
# 2 - high cost
# 3 - very high cost

# Mobile Price Classification, 타겟 변수는 price_range

# test = pd.read_csv("test.csv")
data = pd.read_csv("train.csv")


# fig, ax = plt.subplots()
# ax.scatter(x = data['battery_power'], y = data['price_range'])
# plt.ylabel('price_range', fontsize=13)
# plt.xlabel('battery_power', fontsize=13)
# plt.show()

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

all_data = pd.concat((X_train, X_test))

all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

# missing data가 없는 걸 확인 할 수 있었다.

# print(all_data.dtypes)

# 데이터의 자료형을 확인했더니 전부 숫자형임을 확인했습니다.

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

features = X_test_s.T
cov_matrix = np.cov(features)
eigenvalues = lin.eig(cov_matrix)[0]
eigenvalues = np.sort(eigenvalues)[::-1]
dimension_num = [i for i in range(1, len(eigenvalues)+1)]

graph_dimension = pd.DataFrame()
graph_dimension['eigenvalues'] = eigenvalues
graph_dimension['dimension_num'] = dimension_num

fig, ax = plt.subplots()
ax.scatter(x = graph_dimension['dimension_num'], y = graph_dimension['eigenvalues'])
plt.ylabel('eigenvalues', fontsize=13)
plt.xlabel('dimension_num', fontsize=13)
# plt.show()

# 차원축소를 해보려했는데 축소를 하지 않는게 나을 거라는 판단이 들어서 그냥 진행하겠습니다.

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

tuned_parameters_for_GB = {
    'max_depth': [2, 3],
    'n_estimators': (np.arange(1, 200, 10)),
}

GB_model = GradientBoostingRegressor()
model_GB = GridSearchCV(GB_model, tuned_parameters_for_GB, cv=5, verbose=3)
model_GB.fit(X_train_s, y_train)
print(model_GB.best_params_)


tuned_parameters_xgb = {
    'max_depth': [2, 3],
    'n_estimators': (np.arange(1,200,10)),
    'gamma': (np.arange(0.1, 1, 0.1))
}

XGB_model = xgb.XGBRegressor()
XGB_model = GridSearchCV(XGB_model, tuned_parameters_xgb, cv=5, verbose = 3)
XGB_model.fit(X_train_s, y_train)
print(XGB_model.best_params_)

tuned_parameters_lgb = {
    'boosting_type': ['gbdt', 'dart', 'goss', 'rf'],
    'max_depth': [2, 3],
    'n_estimators': (np.arange(1,200,10)),
    'num_leaves': [30, 50, 70, 100]
}
LGB_model= lgb.LGBMRegressor()
LGB_model = GridSearchCV(LGB_model, tuned_parameters_lgb, cv=5, verbose = 3)
LGB_model.fit(X_train_s, y_train)
print(XGB_model.best_params_)