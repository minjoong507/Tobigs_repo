import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.metrics import *
import matplotlib.pyplot as plt


data = pd.read_csv("sampled_data.csv", sep=",")

# X, y 분리
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Train, test set 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# 스케일링
ss = StandardScaler()
X_train_s = ss.fit_transform(X_train)
X_test_s = ss.transform(X_test)

logreg = LogisticRegression()
logreg.fit(X_train_s, y_train)
logreg.score(X_test_s, y_test)

# confusion matrix
y_pred = logreg.predict(X_test)
conf_m = confusion_matrix(y_pred, y_test)
print(conf_m)
a = conf_m[0][0] + conf_m[1][1]
sum = 0
for row in conf_m:
    for index in row:
       sum += index

print(a/sum)

fpr, tpr, thresholds = roc_curve(y_test, logreg.decision_function(X_test_s))
x = fpr
y = tpr

plt.plot(x,y) # 간단하게 ROC 그려볼 수 있음.
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
plt.show()
print(thresholds)
res = [[fpr_i, tpr_i, thres_i] for fpr_i, tpr_i, thres_i in zip(fpr, tpr, thresholds)]
print(res)