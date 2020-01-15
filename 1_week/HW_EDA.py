import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


data = pd.read_csv("data/Auction_master_train.csv", sep=",")
print(data)

data_drop_column = data.dropna(axis=1)
print(data_drop_column)

sns.heatmap(data=data_drop_column.corr())
plt.show()

data_drop_column.hist(bins=30)
plt.show()