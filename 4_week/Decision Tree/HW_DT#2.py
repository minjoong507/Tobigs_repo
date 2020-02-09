import pandas as pd
import numpy as np

df = pd.read_csv('https://raw.githubusercontent.com/AugustLONG/ML01/master/01decisiontree/AllElectronics.csv')
df.drop("RID",axis=1, inplace = True) #RID는 그냥 Index라서 삭제

