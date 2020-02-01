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
print(logreg.score(X_test_s, y_test))

fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test_s)[:, 1])
x = fpr
y = tpr

plt.plot(x, y)
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
plt.show()
res = [[abs(fpr_i - tpr_i), thres_i] for fpr_i, tpr_i, thres_i in zip(fpr, tpr, thresholds)]
print(res)
d_point = 0
for i in res:
    if i[0] > d_point:
        d_point = i[0]
        cut_off = i[1]
print(cut_off)


# confusion matrix
y_pred = logreg.predict_proba(X_test_s)[:,1:]
a = []
for i in y_pred:
    if i < cut_off:
        a.append(0)
    else:
        a.append(1)
conf_m = confusion_matrix(a, y_test)
a = conf_m[0][0] + conf_m[1][1]
sum = 0
for row in conf_m:
    for index in row:
       sum += index

print(a/sum)