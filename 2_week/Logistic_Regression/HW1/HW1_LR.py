import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.metrics import *


data = pd.read_csv("sampled_data.csv", sep=",")

X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], random_state=0)
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
classifier.score(X_test, y_test)
y_pred = classifier.predict(X_test)

print(classifier.score(X_test, y_test))
f1_score(y_test, y_pred)

scaler = StandardScaler()
new_data = pd.DataFrame(scaler.fit_transform(X_train))

X_train2, X_test2, y_train2, y_test2 = train_test_split(new_data.iloc[:, :-1], new_data.iloc[:, -1], random_state=0)

classifier2 = LogisticRegression()
classifier2.fit(X_train2, y_train2)
print(classifier2.score(X_test2, y_test2))

