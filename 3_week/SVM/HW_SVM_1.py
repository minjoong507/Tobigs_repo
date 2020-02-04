import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

iris = sns.load_dataset('iris') #data load
X = iris.iloc[:, :4]
y = iris.iloc[:, -1]

scal = StandardScaler() #scaling
X = scal.fit_transform(X)

# One Versus Rest
svm_1 = SVC(kernel ='rbf', gamma = 5, C = 100)
svm_2 = SVC(kernel ='rbf', gamma = 5, C = 100)
svm_3 = SVC(kernel ='rbf', gamma = 5, C = 100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=48)
y_train = pd.get_dummies(y_train) #one hot encoding
print(y_train)
print(y_train.iloc[:, 0])


svm_1.fit(X_train, y_train.iloc[:, 0]) # setosa
svm_2.fit(X_train, y_train.iloc[:, 1]) # versicolor
svm_3.fit(X_train, y_train.iloc[:, 2]) # virginica

# 부호가 모든 같은 경우가 있는가? < 모두 동점인 경우!!
for i in range(len(X_test)):
    if (np.sign(svm_1.decision_function(X_test)[i]) == np.sign(svm_2.decision_function(X_test)[i])) \
            and (np.sign(svm_2.decision_function(X_test)[i]) == np.sign(svm_3.decision_function(X_test)[i])):
        print(i)

print(np.sign(svm_1.decision_function(X_test)) == np.sign(svm_2.decision_function(X_test)))
print(np.sign(svm_1.decision_function(X_test)))

print(svm_1.decision_function(X_test))
print(svm_2.decision_function(X_test))
print(svm_3.decision_function(X_test))

print(svm_1.predict(X_test))
print(svm_2.predict(X_test))
print(svm_3.predict(X_test))

result = []
for i in range(len(X_test)):
    if (svm_1.decision_function(X_test)[i] >= svm_2.decision_function(X_test)[i]) \
            and (svm_1.decision_function(X_test)[i] >= svm_3.decision_function(X_test)[i]):
        result.append(1)
    elif (svm_2.decision_function(X_test)[i] >= svm_1.decision_function(X_test)[i]) \
            and (svm_2.decision_function(X_test)[i] >= svm_3.decision_function(X_test)[i]):
        result.append(2)
    else:
        result.append(3)
print(result)