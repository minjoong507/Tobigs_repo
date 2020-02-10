import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
color = sns.color_palette()
sns.set_style('darkgrid')

from scipy import stats
from scipy.stats import norm, skew

test = pd.read_csv('test_new.csv')
train = pd.read_csv('train_new.csv')

#check the numbers of samples and features
print("The train data size before dropping Id feature is : {} ".format(train.shape))
print("The test data size before dropping Id feature is : {} ".format(test.shape))

#Save the 'Id' column
train_ID = train['Id']
test_ID = test['Id']

#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

#check again the data size after dropping the 'Id' variable
print("\nThe train data size after dropping Id feature is : {} ".format(train.shape))
print("The test data size after dropping Id feature is : {} ".format(test.shape))
ntrain = train.shape[0]
print(ntrain)
print("))))")
fig, ax = plt.subplots()
ax.scatter(x = train['SalePrice'], y = train['GrLivArea'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
# plt.show()
train['Z_SalePrice'] = stats.zscore(train['SalePrice'])
train['Z_GrLivArea'] = stats.zscore(train['GrLivArea'])
print(train.shape)

train = train[train['Z_SalePrice'].between(-2, 4)]
train = train[train['Z_GrLivArea'].between(-3, 4)]

print(train.shape)
train.drop("Z_SalePrice", axis=1)
train.drop("Z_GrLivArea", axis=1)


fig, ax = plt.subplots()
ax.scatter(x = train['Z_SalePrice'], y = train['Z_GrLivArea'])
plt.ylabel('Z_SalePrice', fontsize=13)
plt.xlabel('Z_GrLivArea', fontsize=13)
# plt.show()

ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
# print("all_data size is : {}".format(all_data.shape))

all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})


f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index, y=all_data_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)

# print(all_data["PoolQC"])
all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
# print(all_data["PoolQC"])
# print(all_data)
#
#
# print(all_data["LotFrontage"])
# print(all_data["Neighborhood"])
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.mean()))

# print(all_data["BsmtExposure"])
# print(all_data["BsmtFinType1"])
#
# for i in all_data["BsmtQual"].isnull():
#     if i is True:
#         print("fuck")
# all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)
all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)

print(all_data['MSSubClass'])
#MSSubClass=The building class
all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)


#Changing OverallCond into a categorical variable
all_data['OverallCond'] = all_data['OverallCond'].astype(str)


#Year and month sold are transformed into categorical features.
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)
all_data['FireplaceQu'] = all_data['FireplaceQu'].astype(str)
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
        'YrSold', 'MoSold')
for col in cols:
    LE = LabelEncoder()
    print(all_data[col])
    print(col)
    all_data[col] = LE.fit_transform(all_data[col])
    print(all_data[col])

# shape
print(all_data['FireplaceQu'])
print('Shape all_data: {}'.format(all_data.shape))



X_train, X_val, y_train, y_val = train_test_split(train, test, random_state=49)

gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120, random_state=42)
gbrt.fit(X_train, y_train)

# 최적의 트리 개수 찾기
errors = [mean_squared_error(y_val, y_pred)
          for y_pred in gbrt.staged_predict(X_val)]
bst_n_estimators = np.argmin(errors)

# 최적의 트리개수로 그래디언트 부스팅 학습
gbrt_best = GradientBoostingRegressor(max_depth=2, n_estimators=bst_n_estimators, random_state=42)
gbrt_best.fit(X_train, y_train)