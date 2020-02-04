import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Anomaly detection(사기감지 데이터) 로드
data = pd.read_csv('anomaly-detection/creditcard.csv')

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

for i in y.values:
    if i != 0:
        print(i)