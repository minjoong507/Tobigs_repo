##  # [Deep Residual Learning for Image Recognition] paper review

## 1. 요약

- ResNet 은 층이 깊은 NN 일 수록 학습하기 어렵다는 문제를 해결하기 위해 만들어졌다고 한다. 
- Residual(잔차)를 학습하는 방법으로 깊은 NN을 이전보다 더 쉽게 학습시키도록 만듬!



## 2. 소개

- 최근까지 네트워크들의 depth는 매우 중요하게 여겨졌고 깊을 수록 더 좋은 성능이 나온다 했지만 그렇지 않았다!
- depth 가 증가할 수록 gradient vanishing / exploding 문제가 발생, 어느 정도의 depth 도달 시 성능이 떨어지는 모습을 보임. 이 현상을 degradation 이라고 부름

- 기존의 모델말고 잔차 매핑을 통해서 degradation 현상을 해결 할 수 있다고 설명함.



## 3. Related work

- 어려운 내용...



## 4. Deep Residual Learning

- short connection, 한개 이상의 층을 skipping 할 때 기존의 H(x)로 F(x) + x 를 근사시키는 것이 더 낫다!
- 그래서 Plain Network 와 Residual Network를 비교하여 설명하고있음.
- ImageNet 데이터셋을 사용하여 평가하였다
- plain network 에서는 층이 많으면 에러율이 더 높게 나왔고 ResNet 에서는 층이 적으면 에러율이 더 높게 나온 사실을 확인할 수 있었다.



## 5. 이후 내용

- 세층을 뛰어넘는 아키텍쳐도 보이고 잔차를 통해서 degradation 문제를 해결했다는 내용? 추가적으로 '잔차'에 집중했다는 사실이 매우 인상적인 논문이다.

