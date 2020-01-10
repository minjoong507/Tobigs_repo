import pandas as pd


#series - 1차원배열
sr = pd.Series([17000, 18000, 1000, 5000],
       index=["피자", "치킨", "콜라", "맥주"])
print(sr.index)
print(sr.values)


#DataFrame - 2차원배열
values = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
index = ['one', 'two', 'three']
columns = ['A', 'B', 'C']

DataF = pd.DataFrame(values, index, columns)
print(DataF)

print(DataF.values)
print(DataF.index)
print(DataF.columns)


data_dic = { '학번' : ['1000', '1001', '1002', '1003', '1004', '1005'],
'이름' : [ 'Steve', 'James', 'Doyeon', 'Jane', 'Pilwoong', 'Tony'],
         '점수': [90.72, 78.09, 98.43, 64.19, 81.30, 99.14]}

DataF2 = pd.DataFrame(data_dic)
index_ex = ['a', 'b', 'c', 'd', 'e', 'f'] # 이거 index를 data의 index 개수 와 같게 해주면 변경가능, 다르면 에러뜸
DataF2 = pd.DataFrame(data_dic, index=index_ex)
print(DataF2)
