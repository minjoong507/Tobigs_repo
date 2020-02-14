import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt
import pandas as pd
import random

#   기본 모듈들을 불러와 줍니다


# In[2]:


x1 = [95, 91, 66, 94, 68, 63, 12, 73, 93, 51, 13, 70, 63, 63, 97, 56, 67, 96, 75, 6]
x2 = [56, 27, 25, 1, 9, 80, 92, 69, 6, 25, 83, 82, 54, 97, 66, 93, 76, 59, 94, 9]
x3 = [57, 34, 9, 79, 4, 77, 100, 42, 6, 96, 61, 66, 9, 25, 84, 46, 16, 63, 53, 30]

#   설명변수 x1, x2, x3의 값이 이렇게 있네요


# In[3]:


X = np.stack((x1, x2, x3), axis=0)

#   설명변수들을 하나의 행렬로 만들어 줍니다


# In[4]:


X = pd.DataFrame(X.T,columns=['x1','x2','x3'])


# In[5]:


X.dot()


# 1-1) 먼저 PCA를 시작하기 전에 항상!!!!!! 데이터를 scaling 해주어야 해요
# 
# https://datascienceschool.net/view-notebook/f43be7d6515b48c0beb909826993c856/ 를 참고하시면 도움이 될거에요

# In[6]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_std = scaler.fit_transform(X)


# In[7]:


print(X_std)


# In[8]:


features = X_std.T


# In[9]:


print(features)


# 1-2) 자 그럼 공분산 행렬을 구해볼게요\
# 
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.cov.html 를 참고하시면 도움이 될거에요

# In[10]:


cov_matrix = np.COV()


# In[11]:


cov_matrix


# 1-3) 이제 고유값과 고유벡터를 구해볼게요
# 
# 방법은 실습코드에 있어요!!

# In[12]:


eigenvalues = '''?'''
eigenvectors = '''?'''


# In[13]:


print(eigenvalues)
print(eigenvectors)


# In[14]:


mat = np.zeros((3,3))


# In[15]:


mat


# In[16]:


mat[0][0] = eigenvalues[0]
mat[1][1] = eigenvalues[1]
mat[2][2] = eigenvalues[2]


# In[17]:


mat


# 1-4) 자 이제 고유값 분해를 할 모든 준비가 되었어요 고유값 분해의 곱으로 원래 공분산 행렬을 구해보세요
# 
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html 를 참고해서 행렬 끼리 곱하시면 됩니다
# 
# 행렬 곱으로 eigenvector x mat x eigenvector.T 하면 될거에요

# In[18]:


np.dot(np.dot(eigenvectors,mat),eigenvectors.T)


# 1-5) 마지막으로 고유 벡터 축으로 값을 변환해 볼게요
# 
# 함수로 한번 정의해 보았어요
# 
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.concatenate.html

# In[19]:


def new_coordinates(X,eigenvectors):
    for i in range(eigenvectors.shape[0]):
        if i == 0:
            new = [X.dot(eigenvectors.T[i])]
        else:
            new = np.concatenate((new,'''?'''),axis=0)
    return new.T

# 모든 고유 벡터 축으로 데이터를 projection한 값입니다


# In[20]:


new_coordinates(X_std,eigenvectors)

# 새로운 축으로 변환되어 나타난 데이터들입니다


# # 2) PCA를 구현해 보세요
# 
# 위의 과정을 이해하셨다면 충분히 하실 수 있을거에요

# In[21]:


from sklearn.preprocessing import StandardScaler

def MYPCA(X,number):
    scaler = StandardScaler()
    x_std = '''?'''
    features = x_std.T
    cov_matrix = '''?'''
    
    eigenvalues = '''?'''
    eigenvectors = '''?'''
    
    new_coordinates(x_std,eigenvectors)
    
    new_coordinate = new_coordinates(x_std,eigenvectors)
    
    index = eigenvalues.argsort()
    index = list(index)
    
    for i in range(number):
        if i==0:
            new = [new_coordinate[:,index.index(i)]]
        else:
            new = np.concatenate(([new_coordinate[:,index.index(i)]],new),axis=0)
    return new.T


# In[22]:


MYPCA(X,3)

# 새로운 축으로 잘 변환되어서 나타나나요?
# 위에서 했던 PCA랑은 차이가 있을 수 있어요 왜냐하면 위에서는 고유값이 큰 축 순서로 정렬을 안했었거든요


# # 3) sklearn이랑 비교를 해볼까요?
# 
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html 를 참고하시면 도움이 될거에요

# In[23]:


from sklearn.decomposition import PCA
pca = '''?'''


# In[24]:


'''?'''


# In[25]:


MYPCA(X,3)


# # 4) MNIST data에 적용을 해볼게요!
# 
# mnist data를 따로 내려받지 않게 압축파일에 같이 두었어요~!!!
# 
# mnist-original.mat 파일과 같은 위치에서 주피터 노트북을 열어주세요~!!!

# In[34]:


import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_mldata
from scipy import io
get_ipython().run_line_magic('matplotlib', 'inline')
from mpl_toolkits.mplot3d import Axes3D

# mnist 손글씨 데이터를 불러옵니다


# In[27]:


mnist = io.loadmat('mnist-original.mat') 
X = mnist['data'].T
y = mnist['label'].T


# In[28]:


# data information

# 7만개의 작은 숫자 이미지
# 행 열이 반대로 되어있음 -> 전치
# grayscale 28x28 pixel = 784 feature
# 각 picel은 0~255의 값
# label = 1~10 label이 총 10개인거에 주목하자


# In[29]:


# data를 각 픽셀에 이름붙여 표현
feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]
df = pd.DataFrame(X,columns=feat_cols)
df.head()


# In[30]:


# df에 라벨 y를 붙여서 데이터프레임 생성
df['y'] = y


# In[33]:


df


# # 지금까지 배운 여러 머신러닝 기법들이 있을거에요
# 
# 4-1) train_test_split을 통해 데이터를 0.8 0.2의 비율로 분할 해 주시고요
# 
# 4-2) PCA를 이용하여 mnist data를 축소해서 학습을 해주세요 / test error가 제일 작으신 분께 상품을 드리겠습니다 ^0^
# 
# 특정한 틀 없이 자유롭게 하시면 됩니다!!!!!!!!!

# In[ ]:




