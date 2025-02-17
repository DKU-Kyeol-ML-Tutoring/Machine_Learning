
- Overfitting, Underfitting
	일반화와 밀접한 관계


- Bias and Variance
Bias : 편향
Variance : 분산
기계학습 알고리즘을 선택할 때 핵심이 되는 개념



기계학습
1. 학습 데이터 수집
![](https://i.imgur.com/a6Wex9F.png)
데이터의 수 : N
input : x
output : y
-> x와 y를 N개 모은 것이 학습 데이터

예측할 y는 -1, +1로 가정



2. 모델 클래스 정의
Linear model 적용 -> 선형함수로 정의됨
모델 클래스를 정의하면 parameter 나옴 (아직 값이 결정되지 않고 학습을 통해 결정되는 것)



![](https://i.imgur.com/K897KOX.png)

1. 학습트레이닝에 잘 동작하는 것을 찾음
학습데이터 S에 대해 잘 동작하는 w,b를 찾는 게 목표 (Parameter Estimation)

>Model class를 정하면 모델 클래스 내에 parameter가 존재하게 되고, 그 parameter를 학습데이터에 대해 잘 동작하도록 결정을 내림




![](https://i.imgur.com/YPn1pcd.png)

2. 잘 동작한다는 것을 수치적으로 정의해야함
--> 손실함수 (=Loss function)
주어진 Input에 대해 모델이 예측 -> 모델의 예측 값과 정답 값이 틀리면 틀릴수록 큰 값을 반환하는 함수
	1. Squared Loss
	regression 문제에서 주로 사용
	예측 값과 정답 값이 틀리면 틀릴수록 2차의 함수로 패널티를 주는 것
	2. 0/1 Loss
	예측 값과 정답 값이 맞으면 Loss는 0, 틀리면 1




![](https://i.imgur.com/pvi77kr.png)
3. 학습을 최적화 문제로 결정
Loss function : L
모델의 함수값이 예측한 값 : h

-> Loss를 최소화하는 w,b를 찾는 것이 목적




기계학습에서의 일반화
: 기계학습 알고리즘이 학습 과정 동안 보지 못한 새로운 데이터에 대해 잘 동작하는 것
- overfitting(과적합)과 맞닿아 있음

![](https://i.imgur.com/mYlYmIV.png)

Overfitting 그래프
- 사소한 노이즈까지 고려 -> 알고리즘의 함수적 표현이 복잡함

Generalization 그래프
- 사소한 노이즈 고려하지 않고 전체적인 패턴만 파악 -> 알고리즘의 함수적 표현이 간단함
- Training Error가 매우 큼 (데이터 포인트의 y값과 실제 함수 값의 차이)

Training Error 관점에서는 왼쪽 모델이 적합
일반화 관점에서는 왼쪽 모델의 일반화 능력이 떨어질 수 있기 때문에 오른쪽 모델이 적합
--> 정확도는 희생하면서 일반화 능력을 높이기 위한 모델 선택
--> 많은 경우의 오른쪽 모델을 선택함



Generalization Error
- Overfitting
	Generalization error > Training error
	너무 과하게 학습데이터에 적합된 상태
- Underfitting
	Generalization error < Training error
	과소적합

-> Underfitting이 더 안 좋은 상태
	Overfitting이 나는 게 목표
	-> 적어도 training data에 대해 적합한 모델을 찾은 것이기 때문에
	과도한 overfitting이라면 수정 필요



모델의 용량
![](https://i.imgur.com/f1qi6sZ.png)

4. 선형 함수
	직선 밖에 표현하지 못하기 때문에 위와 같은 data set이 주어진다면 underfitting 발생
5. 2차 함수
	가장 적절
6. 9차 함수
	필요 이상으로 복잡한 모델
	training error는 줄일 수 있으나 데이터가 없는 구간의 굴곡이 심해서 실제로 up, down이 심할 확률이 적음

--> 확률적으로 고려하여 선택해야함

![](https://i.imgur.com/AjFR2BU.png)

모델의 용량이 커질수록 무조건 training error는 줄어듦
일반화 에러는 측정 불가, 예측만 가능함



Regularizaion(정규화)
학습과정 동안 최적화 문제를 풀게 되는데 그때 목적함수가 필요함. 목적함수는 학습데이터에 대해 손실함수를 정의해서 손실이 최소화 되도록 정의한다. 그 과정에서 과적합에 빠질 수 있기 때문에 term을 추가함 (Regularization term) 
-> 정규화 과정을 추가함으로써 일반화 에러를 낮추는 것이 목표 (training error를 낮추기 위함이 아님 / training error를 낮추려면 정규화 과정을 추가하면 안됨)



Bias, Variance
![](https://i.imgur.com/cpajekz.png)

Bias, Variance 모두 낮아야함
	Bias : 평균 값과 true 값의 차이
	Variance : 평균 값과 거리 값을 통해 구함

--> Test error = Bias + Variance



Overfitting, Underfitting
Overfitting
	과하게 적합된 것 (training data의 noise까지) -> 모델이 unstable -> Variance 증가
	모델의 복잡도, 용량이 증가할수록 Variance 증가 
	(Variance를 줄이기 위해서는 학습데이터를 많이 모으는 게 가장 좋음)
Underfitting
	과소하게 적합된 것 -> Bias 증가
	Training error를 어느 지점보다 더 줄이지 못함
	모델의 복잡도, 용량이 너무 낮다 -> 허접한 모델
	-> 모델의 복잡도를 증가시켜야함