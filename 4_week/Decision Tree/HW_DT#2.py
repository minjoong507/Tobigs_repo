import pandas as pd
import numpy as np
import math
from itertools import combinations

df = pd.read_csv('https://raw.githubusercontent.com/AugustLONG/ML01/master/01decisiontree/AllElectronics.csv')
df.drop("RID",axis=1, inplace = True) #RID는 그냥 Index라서 삭제

print(math.log(10, 2))

def getEntropy(df, feature):
    yes_num = 0
    no_num = 0
    for data in df[feature]:
        if data == 'yes':
            yes_num += 1
        else:
            no_num += 1

    if yes_num == 0:
        entropy = - (no_num / (yes_num + no_num) * math.log(no_num / (yes_num + no_num), 2))
    elif no_num == 0:
        entropy = - (yes_num / (yes_num + no_num) * math.log(yes_num / (yes_num + no_num), 2))
    else:
        entropy = - (yes_num / (yes_num + no_num) * math.log(yes_num / (yes_num + no_num), 2) + no_num / (yes_num + no_num) * math.log(no_num / (yes_num + no_num), 2))

    return entropy

print(getEntropy(df, "class_buys_computer"))

def getGainA(df, feature) :
    info_D = getEntropy(df, feature) # 목표변수 Feature에 대한 Info(Entropy)를 구한다.
    columns = list(df.loc[:, df.columns != feature]) # 목표변수를 제외한 나머지 설명변수들을 리스트 형태로 저장한다.

    result = {}

    for context in columns:
        Info_context = [] # 인포값 담을 리스트
        col_list = [] # 칼럼이 가지는 값의 종류
        context_df = df[[context, feature]] # 필요변수만 가지고 새로운 데이터프레임 생성
        kkk = set(list(combinations(df[context], 1)))

        for i in kkk:
            col_list.append((list(i)[0]))  # 리스트 형변환

        for i in col_list:
            new_context_df = context_df[context_df[context] == i] # 엔트로피 계산을 위하여 다시 데이터프레임 분리
            Info_context.append(getEntropy(new_context_df, "class_buys_computer") / len(col_list))

        result[context] = info_D - sum(Info_context)

    return result

print(getGainA(df, "class_buys_computer"))
