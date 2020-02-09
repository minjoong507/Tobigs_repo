import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
from sklearn.preprocessing import StandardScaler

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
print(all_data)


print(all_data["LotFrontage"])
print(all_data["Neighborhood"])
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.mean()))

print(all_data["BsmtExposure"])
print(all_data["BsmtFinType1"])

for i in all_data["BsmtQual"].isnull():
    if i is True:
        print("fuck")

print(all_data['MSZoning'].mode()[0])
