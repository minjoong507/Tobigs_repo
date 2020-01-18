import operator

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


'''
 문제 내용

1. 결측값이 있는 모든 열을 없애주세요
2. 모든 연속형 변수 간의 상관관계를 Heatmap을 통해 확인해 주세요
3. 모든 연속형 변수의 분포를 Histogram으로 확인해 주세요
4. Target 변수와 관련 있거나, 유의미한 Insight를 얻을 수 있는 시각화를 5개 이상 해주세요 (subplot활용)

2) 1-4에서 도출된 시각화 + 번뜩이는 Insight를 바탕으로 유의미한 Feature를 10개 이상 생성해 주세요
( Target 변수(=Hammer Price) 와 관련이 있으면 좋지만, 아니어도 괜찮습니다 )
'''



#데이터 입출력
data = pd.read_csv("data/Auction_master_train.csv", sep=",")
#print(data)


#결측값이 있는 모든 열 제거
data_drop_column = data.dropna(axis=1)


#Heatmap 출력
sns.heatmap(data=data_drop_column.corr())
#plt.show()


#Histogram 출력
data_drop_column.hist(bins=30)
#plt.show()


#DataFram index 접근
# print(data_drop_column['addr_do']) # 이렇게 출력하면 자료형이 series type 임을 확인 그리고 특정 칼럼이 값을 가지는 형태를 보려고 해
#
# print(type(data_drop_column['Final_result'].values)) # numpy 의 ndarray 자료구조로 나옴
#
# print(data_drop_column.iloc[2]) # 데이터 어떤 value 들을 가지는지 그냥 눈으로 확인하려고 해봄


# 칼럼 값들 뽑아보다가 주소도 뽑아봤는데 몇가지 없을 것 같아서 한번 확인해보니
# 여기서는 서울 부산밖에 안다루니까 두개를 구분지어서 또 활용하면 괜찮다고 생각했다.

addr_do_list = []
for i in data_drop_column['addr_do'].values:
    if i not in addr_do_list:
        addr_do_list.append(i)
#print(addr_do_list)


addr_si_list = []
for i in data_drop_column['addr_si'].values:
    if i not in addr_si_list:
        addr_si_list.append(i)
# print(addr_si_list)


# '+++'가 하나도 출력 안된걸 보니 유찰된 경매의 데이터는 들어가지 않았다.
# 혹시라도 낙찰 안된 데이터 있으면 무의미 하다고 생각되어 제거하려 했다.
# 그래서 Final_result 칼럼은 고려하지 않아도 된다는 사실 확인

for i in data_drop_column['Final_result'].values:
    if i != "낙찰":
        print("+++")
print("-------")


#시각화 5개 이상 (subplot 활용)
# plt.plot(data_drop_column['Hammer_price'])
# plt.show()
#
# plt.plot(data_drop_column['Auction_count'])
# plt.show()

'''
 일단 전체 row들의 특정 칼럼 별로 데이터를 본 결과로 매우 다양하게 있고 서로 연관관계가
 있을 법한 column을 선정해보고 그 column 을 시각화 해볼 예정이다.
 
 Claim_price : 경매 신청인의 청구 금액
 Auction_count : 경매 횟수
 Auction_miscarriage_count : 총 유찰 횟수
 Total_land_gross_area: 총 토지 전체 면적
 addr_do : 시_도
 addr_si : 주소_시군구
 Total_appraisal_price : 총 감정가
 Minimum_sales_price : 최저매각가격
 Hammer_price(target) : 낙찰가
'''

Claim_price = list(data_drop_column['Claim_price'].values)
Area = list(data_drop_column['Total_land_gross_area'].values)
Hammer_Price = list(data_drop_column['Hammer_price'].values)
Auction_count = list(data_drop_column['Auction_count'].values)
Auction_f_count = list(data_drop_column['Auction_miscarriage_count'].values)
Minimum_sales_price = list(data_drop_column['Minimum_sales_price'].values)
Total_appraisal_price = list(data_drop_column['Total_appraisal_price'].values)

# 1. 토지 면적당 가격 (단위 : 1 m^2)
# 변수 명 : m_per_price

m_per_price = []

for i, price in enumerate(Hammer_Price):
    if Area[i] == 0:
        m_per_price.append(0)
    else:
        temp = price / Area[i]
        m_per_price.append(temp)


# 2. 경매가 유찰 될 확률 (경매 횟수에 비례한 유찰 경우)
# 변수 명 : Auction_success

Auction_success = []

for i, tot_count in enumerate(Auction_count):
    if Auction_f_count[i] == 0:
        Auction_success.append(0)
    else:
        Auction_success.append(Auction_f_count[i] / tot_count)


# 3. 경매 신청인의 이익 계산 (낙찰가 - 경매 청구 금액)
# 변수 명 : final_revenue

final_revenue = []

for i, price in enumerate(Hammer_Price):
    final_revenue.append(price - Claim_price[i])


# 4. 경매가 활발한 지역 순위 지정 (시, 군으로 분류)
# 변수 명 : Auction_frequency_rank_list

Auction_frequency_rank_list = []

Auction_frequency_rank = {}
for si in addr_si_list:
    Auction_frequency_rank[si] = 0

for i in range(0, len(data_drop_column)):
    for si in addr_si_list:
        if data_drop_column.ix[i]['addr_si'] == si:
            Auction_frequency_rank[si] += int(data_drop_column.ix[i]['Auction_count'])

sortedArr = sorted(Auction_frequency_rank.items(), key=operator.itemgetter(1, 0), reverse=True)

for i in sortedArr:
    Auction_frequency_rank_list.append(i[0])


# 5. 경매 상승 금액 (최저 입찰가와 낙찰가의 차액)
# 변수 명 : margin_price

margin_price = []

for i, price in enumerate(Hammer_Price):
    margin_price.append(price - Minimum_sales_price[i])


# 6. 감정가와 낙찰가가 맞을 확률 (감정가가 낙찰가를 잘 예측했을 확률)
#  변수 명 : appraisal_price_accuracy

appraisal_price_accuracy = []

sub_price = []
for i, price in Hammer_Price:
    sub_price.append(abs(price - Total_appraisal_price[i]))
print(sub_price)
