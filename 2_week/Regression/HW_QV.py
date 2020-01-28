import pandas as pd
import numpy as np
from numpy.linalg import inv

def estimate_beta(x, y):
    new_x = inv(np.dot(x.T, x))
    new_y = np.dot(x.T, y)
    beta_hat = np.dot(new_x, new_y)
    return beta_hat

def make_intercept(x):
    temp = np.ones((len(x), 1))
    x = np.hstack([temp, x])
    return x

def make_MSE(m1, m2, Y):
    beta_matrix = np.dot(m1, m2)
    e_matrix = np.subtract(Y, beta_matrix)
    SSE = np.dot(e_matrix.T, e_matrix)
    MSE = SSE / (len(Y) - len(m1.T) - 2)
    return MSE

# 데이터 입출력
data = pd.read_csv("Auction_master_train.csv", sep=",")
data = data.dropna(axis=1)

# 선언부
x_value = np.array(list(data['point.x'].values))
y_value = np.array(list(data['point.y'].values))
Hammer_Price = np.array(list(data['Hammer_price'].values))
total_addr_do = np.array(list(data['addr_do'].values))
total_addr_si = np.array(list(data['addr_si'].values))
total_compnay_list = np.array(list(data['Appraisal_company'].values))
Minimum_sales_price = np.array(list(data['Minimum_sales_price'].values))
Claim_price = np.array(list(data['Claim_price'].values))
Total_appraisal_price = np.array(list(data['Total_appraisal_price'].values))


company_list = []
for i in data['Appraisal_company'].values:
    if i not in company_list:
        company_list.append(i)

addr_do_list = []
for i in data['addr_do'].values:
    if i not in addr_do_list:
        addr_do_list.append(i)

addr_si_list = []
for i in data['addr_si'].values:
    if i not in addr_si_list:
        addr_si_list.append(i)



#1 X = 경도, 위도, 시/도 Y = 낙찰가

data_do = []

# 인덱싱 과정
for i, addr in enumerate(total_addr_do):
    # 부산이면 0
    if addr in addr_do_list[0]:
        data_do.append(0)
    # 서울이면 1
    else:
        data_do.append(1)

tmp = np.column_stack([x_value, y_value])
X_1 = np.column_stack([tmp, np.array(data_do)])
Y_1 = np.transpose(Hammer_Price)
result_1 = estimate_beta(make_intercept(X_1), Y_1)

print(make_MSE(make_intercept(X_1), result_1, Y_1))


#2 X = 청구금액, 최저매각가격, 시/군/구 Y = 낙찰가

data_si = {}
data_si_indexing = []

# 인덱싱 과정
for i, si in enumerate(addr_si_list):
    data_si[si] = i

for si in total_addr_si:
    data_si_indexing.append(data_si[si])

tmp = np.column_stack([Claim_price, Minimum_sales_price])
X_2 = np.column_stack([tmp, np.array(data_si_indexing)])
Y_2 = np.transpose(Hammer_Price)
result_2 = estimate_beta(make_intercept(X_2), Y_2)

print(make_MSE(make_intercept(X_2), result_2, Y_2))

#3 X = 총 감정가, 감정사 Y = 낙찰가

data_company = {}
data_company_indexing = []

# 인덱싱 과정
for i, cp in enumerate(company_list):
    data_company[cp] = i

for cp in total_compnay_list:
    data_company_indexing.append(data_company[cp])

X_3 = np.column_stack([Total_appraisal_price, data_company_indexing])
Y_3 = np.transpose(Hammer_Price)
result_3 = estimate_beta(make_intercept(X_3), Y_3)


print(make_MSE(make_intercept(X_3), result_3, Y_3))

