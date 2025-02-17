

- 기계학습
인공지능의 한 분야
알고리즘을 설계하고 개발하는 분야



- 머신러닝
어떤 작업 (T)에 대해 경험데이터 (E)를 기반으로 특정 performance metric (P)를 개선하는 것 

>T : 체스를 두는 것
>P : 상대방을 많이 이길수록 좋음 (승률)
>E  : 경험 데이터 




![](https://i.imgur.com/jYzv80B.png)

- 전통적인 프로그래밍
	코드와 input 정보를 컴퓨터에 주면 프로그램에 정의된 순서대로 연산을 수행하여 최종 결과 값인 output을 출력

- 머신러닝
	input과 원하는 output을 모아서 컴퓨터에게 주면 그에 맞는 최종적인 program이 나옴
	ex) spell checker
	input과 output 쌍을 학습데이터로 알고리즘에게 전달
	학습된 알고리즘을 사용하여 spell shceking 수행





기계학습 하는 목적 : **일반화**

- 일반화
training data에게서 패턴을 배워서, 새로운 데이터가 오더라도 데이터를 잘 이해할 수 있게 하기 위함
학습 데이터를 완벽하게 이해한다면, 학습데이터를 생성할 수도 있음 = 생성형 인공지능


- No Free Lunch Theorem
새로운 task, data를 모을 때마다 최적의 알고리즘을 찾는 작업이 필요함
어떤 기계학습 알고리즘도 다른 기계학습 알고리즘 보다 항상 좋다고 할 수 없음
= **하나의 알고리즘이라도 하더라도 모든 경우에 다 좋을 수는 없음**






기계학습의 종류
1. Supervised learning 지도 학습
2. Unsupervised learning 비지도 학습
3. Semi-supervised learning 
4. Reinforcement learning 강화 학습



Supervised learning 지도 학습
기계학습 알고리즘에 대해 input에 따른 output을 명시적으로 알려줌
	Classification : y가 범주형 변수
	Regression : Output이 실수로 예측됨



Unsupervised learning 비지도 학습
학습 데이터가 Input data로만 이루어져 있음



Semi-supervised learning 
학습데이터를 줄 때 몇몇은 input만, 몇몇은 Input, output 둘 다 줌
-> Supervised learning과 Unsupervised learning의 중간

label(output)은 사람이 작업
->몇몇 데이터는 label 된 데이터, 몇몇 데이터는 unlabel 된 데이터 
	LU learning : 몇몇 데이터는 label 된 데이터, 몇몇 데이터는 unlabel 된 데이터
	PU learning : 정상 클래스, 비정상 클래스로 나뉠 때 정상클래스에 대해서만 label된 data를 줌

![](https://i.imgur.com/2fmSIWL.png)

label이 주어지지 않았지만 label을 확률적으로 줄 수 있음
- 빨간색이랑 가까운 unlabel data는 빨간색일 확률이 높음
- Decision Boundary를 훨씬 정확하게 예측 가능 -> classification 성능 높일 수 있음



Reinforcement learning 강화 학습
데이터셋이 사전에 주어지지 않고 환경이 주어짐
알고리즘이 환경을 보고 알아서 배우게 하는 것
보상이 바로 오지 않을 수 있기 때문에 가장 어려움


