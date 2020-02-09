import pandas as pd
import numpy as np
from itertools import combinations

pd_data = pd.read_csv('https://raw.githubusercontent.com/AugustLONG/ML01/master/01decisiontree/AllElectronics.csv')
pd_data.drop("RID", axis=1, inplace=True)

def get_gini(df, label):
    Yes_num = 0
    No_num = 0
    for i in df[label]:
        if i == 'yes':
            Yes_num += 1
        else:
            No_num += 1

    gini = 1 - (((Yes_num / (Yes_num + No_num)) ** 2 + (No_num / (Yes_num + No_num)) ** 2))
    return gini


def get_binary_split(df, attribute):
    result = []
    temp = []

    kkk = set(list(combinations(df[attribute], 1))) # 1개 조합 전부 만들기

    for i in kkk:
        temp.append((list(i)[0])) # 리스트 형변환

    x = len(temp)
    for index in range(1 << x):
        result.append([temp[j] for j in range(x) if (index & (1 << j))]) # 부분조합 만들기

    return result[1:-1]

print(get_binary_split(pd_data, "age"))


def get_attribute_gini_index(df, attribute, label):
    result = {}
    feature = get_binary_split(df, attribute)

    for context in feature:
        context_num = 0 # 해당 속성 갯수
        not_context_num = 0 # 해당 속성 제외 갯수

        context_yes = 0 # 해당 속성 yes 갯수
        context_no = 0 # 해당 속성 no 갯수

        not_context_yes = 0 # 해당 속성 제외 yes 갯수
        not_context_no = 0 # 해당 속성 제외 no 갯수

        context_df = df[[attribute, label]] # 필요변수만 가지고 새로운 데이터프레임 생성

        for i in context_df.values:
            if i[0] in context and i[1] == 'yes':
                context_num += 1
                context_yes += 1

            elif i[0] in context and i[1] == 'no':
                context_num += 1
                context_no += 1

            elif i[0] not in context and i[1] == 'yes':
                not_context_num += 1
                not_context_yes += 1
            else:
                not_context_num += 1
                not_context_no += 1

        s = ",".join(context) # 딕셔너리에 넣을 키값 만들기, 리스트 형식을 문자열로

        # 아래는 수식계산입니다.
        result[s] = (context_num / (context_num + not_context_num)) * (1 - (
                (context_yes / (context_yes + context_no)) ** 2 + (context_no / (context_yes + context_no)) ** 2)) + \
                             (not_context_num / (context_num + not_context_num)) * (1 - (
                (not_context_yes / (not_context_yes + not_context_no)) ** 2 + (not_context_no / (not_context_yes + not_context_no)) ** 2))

    return result


dic = get_attribute_gini_index(pd_data, "age", "class_buys_computer")
print(dic)

def f1(x):
    return dic[x]
key_min = min(dic.keys(), key=(lambda  k:dic[k]))

print(key_min)