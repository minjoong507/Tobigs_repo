import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn import metrics

# Anomaly detection(사기감지 데이터) 로드
data = pd.read_csv('anomaly-detection/creditcard.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

scal = StandardScaler()
X = scal.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=48)

labels = pd.DataFrame(y_train)
labels.columns = ['labels']
data_feature = data.drop("Class", axis=1)
data = pd.DataFrame(X_train)
data.columns = data_feature.columns
new_data = pd.concat([data, labels], axis=1)
X_samp, y_samp = RandomUnderSampler(random_state=0).fit_sample(X_train, y_train)

svc = SVC(kernel='linear', C = 100)
svc.fit(X_samp, y_samp)
y_pred=svc.predict(X_test)
print('Accuracy Score:')
print(metrics.accuracy_score(y_test,y_pred))

svm_model= SVC()
tuned_parameters = {
    'C': (np.arange(3, 5, 0.2)), 'kernel': ['linear', 'rbf'],
    'gamma': (np.arange(1, 5.1))
}
model_svm = GridSearchCV(svm_model, tuned_parameters, cv=10, scoring='accuracy', verbose = 3)
model_svm.fit(X_samp, y_samp)
print(model_svm.best_score_)
print(model_svm.best_params_)

svm_temp = SVC(kernel = 'linear', gamma = 1.0, C = 3.0)
svm_temp.fit(X_train,y_train)
y_pred = svm_temp.predict(X_test) # 훈련한 모델로 test셋을 시험해보자
print('Accuracy Score:')
print(metrics.accuracy_score(y_test,y_pred)) #스코어 확인