import pandas as pd
import warnings
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

iris = datasets.load_iris()

print(type(iris))

labels = pd.DataFrame(iris.target)
labels.columns = ['labels']
data = pd.DataFrame(iris.data)
data.columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width']
data = pd.concat([data, labels], axis=1)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

grid_params = {
    'n_neighbors': [3, 5, 11, 19],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

gs = GridSearchCV(
    KNeighborsClassifier(), #estimator
    grid_params, # 찾고자하는 파라미터, 위에서 딕셔너리 선언
    verbose = 1, # verbose : integer    Controls the verbosity: the higher, the more messages. 뭔말이지?
    cv = 3, # k fold c-v 에서 k값?
    n_jobs = -1 # 병렬 처리갯수 (-1은 전부)
    )

gs_results = gs.fit(X_train, y_train)
print("Best Parameter: {}".format(gs.best_params_)) #가장 잘나오는 파라미터들을 출력
print("Best Cross-validity Score: {:.3f}".format(gs.best_score_)) #최종 스코어
print("Test set Score: {:.3f}".format(gs.score(X_test, y_test)))

# 여기서부턴 최적의 n_neighbors 찾기

grid_params_for_n = {
    'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}
gs_n = GridSearchCV(
    KNeighborsClassifier(),
    grid_params_for_n,
    cv = 5,
    return_train_score=True # 이게 디폴트가 false 던데 If ``False``, the ``cv_results_`` attribute will not include training scores. True로 한다는거는 cv 결과 속성이 트레이닝 스코어를 포함하게한다?
)
gs_n.fit(X_train, y_train)

print("Best Parameter: {}".format(gs_n.best_params_))
print("Best Cross-validity Score: {:.3f}".format(gs_n.best_score_))
print("Test set Score: {:.3f}".format(gs_n.score(X_test, y_test)))
result_grid = pd.DataFrame(gs_n.cv_results_)
print(result_grid)
plt.plot(result_grid['param_n_neighbors'], result_grid['mean_train_score'], label="Train")
plt.plot(result_grid['param_n_neighbors'], result_grid['mean_test_score'], label="Test")
plt.show()