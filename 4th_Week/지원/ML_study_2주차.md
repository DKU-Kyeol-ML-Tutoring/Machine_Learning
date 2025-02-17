#ML_study 

[[#^7e128b|딥러닝과 클라우드]]
[[#^1cd226|인공지능]]


---



**머신러닝 모델 구축 과정** ^7e128b
1. 데이터 준비
2. 데이터 분할
	학습 데이터와 데스트 데이터로 나눔
3. 모델 학습
	머신러닝 알고리즘을 이용하여 모델 학습
4. 모델 평가
5. 실제 환경에 적용


![](https://i.imgur.com/0nfEXEG.png)




**단순 선형 회귀**
: 하나의 독립변수(x)와 하나의 종속 변수(y) 간의 선형 관계를 찾는 방법
- 목표 : 독립 변수(x)를 이용하여 종속 변수(y)를 예측하는 모델을 만들기
- 수식 : $y = Wx + b$
	$W$ : 기울기
	$b$ : 절편
	-> 상수인 $W$와 $b$를 찾는 것이 학습 목표
- 예제 : 기온(x)와 아이스크림 판매량(y)의 관계 분석



**선형 회귀 모델 학습 방법**
$W$(기울기)와 $b$(절편)을 찾는 과정이 머신러닝의 학습이다
- 최소제곱법 (Least Squares Method) : 가장 작은 오차를 만드는 $W$와 $b$ 찾기
- 경사 하강법(Gradient Descent) : 손실 함수를 미분하여 기울기를 조정하며 최적의 값 찾기



**단순 선형 회귀 코드 구현**
- 예제 데이터: `cars.csv`
    - 속도(`speed`, mph) → 제동 거리(`dist`, ft)
- 회귀식을 통해 속도가 빠를수록 제동 거리가 길어지는 패턴을 찾음

	`데이터 로드 및 전처리`
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 데이터 로드
cars = pd.read_csv('D:/data/cars.csv')

# 데이터 확인
print(cars.head())

# 데이터 분리
speed = cars[['speed']]  # X 변수 (2차원 배열 형태)
dist = cars['dist']      # Y 변수
train_X, test_X, train_y, test_y = train_test_split(speed, dist, test_size=0.2, random_state=123)
```
- `to_frame()` : 1차원 데이터(Series)를 2차원 데이터(dataframe)으로 변경
- `train_test_split()`을 사용해 80% 학습, 20% 테스트 데이터 분리


	`모델 학습`
```python
# 모델 정의
model = LinearRegression()

# 모델 학습 (Fitting)
model.fit(train_X, train_y)
```


	모델 예측
```python
# 예측 수행
pred_y = model.predict(test_X)
print(pred_y)

# 특정 속도 값에 대한 예측
print(model.predict([[13]]))  # speed=13일 때 예측
print(model.predict([[20]]))  # speed=20일 때 예측
```
- `model.predict(test_X)`:테스트 데이터에 대한 예측값 출력
- `model.predict([[13]])`: 속도가 13mph일 때 예상 제동 거리 출력


	`모델 평가`
```python
from sklearn.metrics import mean_squared_error, r2_score

# 계수 출력
print('Coefficients:', model.coef_[0])
print('Intercept:', model.intercept_)

# MSE 및 R² 점수
print('Mean Squared Error:', mean_squared_error(test_y, pred_y))
print('R² Score:', r2_score(test_y, pred_y))
```
- MSE(Mean squared error) : 오차의 평균 제곱 값(낮을수록 좋음)
- R$^2$ Score : 1에 가까울수록 좋은 모델


	`모델 시각화`
```python
import matplotlib.pyplot as plt

plt.scatter(test_X, test_y, color='black')  # 실제 데이터
plt.plot(test_X, pred_y, color='blue', linewidth=3)  # 예측된 회귀선
plt.xlabel('Speed')
plt.ylabel('Stopping Distance')
plt.show()
```
![](https://i.imgur.com/rVXGBRh.png)





**다중 선형 회귀**
: 단순 선형 회귀와 달리 여러 개의 독립 변수를 포함하는 형태
- 수식 : $y = W_1x_1 + W_2x_2 + ... + W_nx_n + b$
- 예제 : 연봉 예측 모델
	변수 : 교육 연수, 여성 비율, 직업 평판 등
	종속 변수 : 연봉

	`코드 구현`
```python
import pandas as ps
import numpy as np
from sklearn.linear_model import LinearRegression
form sklearn.metrics import mean_squared_error, r2_score
form sklearn.model_selection import train_test_split

#데이터 로드
df = pd.read_csv('D:/data/prestige.csv')

df_X = df[['education', 'women', 'prestige']]
df_y = df['income']

#데이터 분리
train_X, test_X, train_y, test_y = train_test_split(df_X, df_y, test_size=0.2, random_state=123)

#모델 정의
model = LinearRegression()

#모델 학습
model.fit(train_X, train_y)

#모델 예측
pred_y = model.predict(test_X)
```
- 다중 변수 입력 데이터를 `DataFrame` 형태로 불러와 학습




**로지스틱 회귀**
: 회귀 분석 중 하나지만 예측 대상이 수치데이터(연속 값)이 아니라 범주형 데이터(yes or no)
- 예제 : iris 품종 예측

```python
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
form sklearn.model_selection import train_test_split
form sklearn.metrics import accuracy_score

#데이터 로드
iris_X, iris_y = datasets.load_iris(return_X_y=True)

#데이터 분리
train_X, test_X, train_y, test_y = train_test_split(iris_X, iris_y, test_size=0.3, random_state=1234)

#모델 정의
model = LogisticRegression()

#모델 학습
model.fit(train_X, train_y)

#모델 예측
pred_y = medel.predict(test_X)
print(pred_y)

#모델 정확성 평가
acc = accuracy_score(test_y, pred_y)
print('Accuracy : {0:3f}'.format(acc))
```

$$Accuracy = \frac{예측과 정답이 일치하는 instance 수}{전체 test instance 수}$$
- 로지스틱 회귀는 종속 변수가 숫자여야 하기 때문에 문자형으로 되어 있는 범주데이터는 숫자로 변환한 후 작업해야한다



---


**선형 회귀**
회귀 : 데이터의 패턴을 분석해서 가장 적합한 선형식 또는 비선형식을 찾는 것 ^1cd226
- $y = f(x)$에서 출력 $y$가 실수이고 입력 $x$도 실수일 때 함수 $f(x)$를 예측
![](https://i.imgur.com/0bACerg.png)


**손실 함수**
: 실제 데이터와 예측 값의 차이를 측정하는 함수
- 직선과 데이터 사이의 간격을 제곱하여 합한 값


**손실 함수 최소화**
- 경사 하강법 사용
	$W=W−α⋅$ $∂Loss \over ∂W$
	- `α` = 학습률(Learning Rate)

	`코드 구현`
```python
import numpy as np
import matplotlib.pyplot as plt 

# 입력 데이터
X = np.array([0.0, 1.0, 2.0])
y = np.array([3.0, 3.5, 5.5])

# 초기 값 설정
W = 0       # 기울기
b = 0       # 절편
lrate = 0.01  # 학습률
epochs = 1000  # 반복 횟수
n = float(len(X))  # 데이터 개수

# 경사 하강법 실행
for i in range(epochs): 
    y_pred = W * X + b  # 예측값 계산
    dW = (2/n) * sum(X * (y_pred - y))  # 기울기 변화량
    db = (2/n) * sum(y_pred - y)  # 절편 변화량
    W = W - lrate * dW  # 기울기 업데이트
    b = b - lrate * db  # 절편 업데이트

# 학습된 W, b 출력
print("기울기(W):", W, "절편(b):", b)

# 그래프 출력
plt.scatter(X, y)
plt.plot([min(X), max(X)], [min(y_pred), max(y_pred)], color='red')
plt.show()
```
- 학습이 완료되면 최적의 직선이 출력



**과잉적합, 과소적합**
- 과잉 적합(overfitting)
	: 학습 데이터에서는 성능이 뛰어나지만 새로운 데이터는 성능이 떨어지는 경우
	모델이 너무 복잡해서 불필요한 패턴까지 학습한 상태
- 과소적합(underfitting)
	: 학습 데이터에서도 성능이 좋지 않은 경우
	모델이 너무 단순해서 패턴을 제대로 학습하지 못한 상태



