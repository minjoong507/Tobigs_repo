import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

#데이터 입출력
data = pd.read_csv("data/Auction_master_train.csv", sep=",")
print(data)


#결측값이 있는 모든 열 제거
data_drop_column = data.dropna(axis=1)
print(data_drop_column)

print(data_drop_column.columns)

#Heatmap 출력
sns.heatmap(data=data_drop_column.corr())
plt.show()

#Histogram 출력
data_drop_column.hist(bins=30)
plt.show()

print(data_drop_column['Final_result'])
print(data_drop_column.iloc[2])
