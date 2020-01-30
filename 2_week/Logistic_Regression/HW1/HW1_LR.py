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
print(confusion_matrix(y_pred, y_test))

fpr, tpr, thresholds = roc_curve(y_test, logreg.decision_function(X_test))
x = fpr
y = tpr

plt.plot(x,y) # 간단하게 ROC 그려볼 수 있음.
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
plt.show()
