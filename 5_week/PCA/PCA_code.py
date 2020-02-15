import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml
from scipy import io
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


def new_coordinates(X,eigenvectors):
    for i in range(eigenvectors.shape[0]):
        if i == 0:
            new = [X.dot(eigenvectors.T[i])]
        else:
            new = np.concatenate((new, [X.dot(eigenvectors.T[i])]), axis=0)
    return new.T

# 모든 고유 벡터 축으로 데이터를 projection한 값입니다

def MYPCA(X, number):
    scaler = StandardScaler()
    x_std = scaler.fit_transform(X)
    features = x_std.T
    cov_matrix = np.cov(features)

    eigenvalues = lin.eig(cov_matrix)[0]
    eigenvectors = lin.eig(cov_matrix)[1]

    new_coordinates(x_std, eigenvectors)

    new_coordinate = new_coordinates(x_std, eigenvectors)

    index = eigenvalues.argsort()
    index = list(index)

    for i in range(number):
        if i == 0:
            new = [new_coordinate[:, index.index(i)]]
        else:
            new = np.concatenate(([new_coordinate[:, index.index(i)]], new), axis=0)
    return new.T


mnist = io.loadmat('mnist-original.mat')
X = mnist['data'].T
y = mnist['label'].T

# data를 각 픽셀에 이름붙여 표현
feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]
df = pd.DataFrame(X, columns=feat_cols)
df['y'] = y


X = df.drop('y', axis=1)
y = df['y']

# labels = []
# yList = y.tolist()
# for label in yList:
#     label = str(label)
#     if label[0] not in labels:
#         labels.append(label[0])
# print(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

features = X_test_s.T
cov_matrix = np.cov(features)
eigenvalues = lin.eig(cov_matrix)[0]
eigenvalues = np.sort(eigenvalues)[::-1]
dimension_num = [i for i in range(1, len(eigenvalues)+1)]

graph_dimension = pd.DataFrame()
graph_dimension['eigenvalues'] = eigenvalues
graph_dimension['dimension_num'] = dimension_num
print(graph_dimension)

fig, ax = plt.subplots()
ax.scatter(x = graph_dimension['dimension_num'], y = graph_dimension['eigenvalues'])
plt.ylabel('eigenvalues', fontsize=13)
plt.xlabel('dimension_num', fontsize=13)
plt.show()


pca = PCA(n_components=18)
X_18d = pca.fit_transform(X_train_s)

rnd_clf = RandomForestClassifier(random_state=0)
rnd_clf.fit(X_18d, y_train)

X_18d_test = pca.transform(X_test_s)

y_pred_18 = rnd_clf.predict(X_18d_test)
accuracy_18d = accuracy_score(y_test, y_pred_18)
print(accuracy_18d)


rnd_clf_2 = RandomForestClassifier(random_state=0)
rnd_clf_2.fit(X_train_s, y_train)
y_pred = rnd_clf_2.predict(X_test_s)
accuracy_d = accuracy_score(y_test, y_pred)
print(accuracy_d)
