# 머신러닝 시스템의 분류
## <font color="#c0504d">훈련 지도 방식에 따른 분류</font>
## 지도 학습 supervised learning
훈련 데이터에 <span style="background:rgba(240, 200, 0, 0.2)">레이블이 포함.</span>
1) **분류** classification
2) **특성** feature을 사용해 **타깃** target 수치 예측 => 회귀 regression
 - 회귀 알고리즘을 분류에 사용할 수도 있고 분류 알고리즘을 회귀에 사용할 수도 있음.

### 비지도 학습 unsupervised learning
훈련 데이터에 <span style="background:rgba(240, 200, 0, 0.2)">레이블이 없음.</span>
1) **군집** clustering  알고리즘 - 그룹으로 묶는다.
2) **계층 군집** hierarchical 알고리즘 - 더 작은 그룹으로 세분화.
	- 분할 군집 divisive clustering (하향식), 병합 군집 agglomerative clustering (상향식)
3) **시각화** visualization 알고리즘 - 도식화가 가능한 2D, 3D 표현 제공
4) **차원 축소** - 특성 합치기 = **특성 추출** feature extraction
5) **이상치 탐지** outlier detection
6) **특이치 탐지** novelty detection
7) **연관 규칙 학습** association rule learning

### 준지도 학습 semi-supervised learning
<span style="background:rgba(240, 200, 0, 0.2)">레이블이 일부만</span> 있는 데이터.
- 대부분 지도 학습 + 비지도 학습


### 자기 지도 학습self-supervised learning
<span style="background:rgba(240, 200, 0, 0.2)">레이블이 없는 데이터셋에서 레이블이 완전히 부여된 데이터셋 생성</span>
- 비지도 학습보단 지도 학습에 가깝
- 비지도 학습 - 군집, 차원 축소, 이상치 탐지
- 자기 지도 학습 - 분류, 회귀

### 강화 학습 reinforcement learning
- 학습하는 시스템 agent. 환경 environment을 관찰해 행동action을 실행하고 그 결과로 <span style="background:rgba(240, 200, 0, 0.2)">보상reward or 벌점penalty</span> 부과
- 시간이 지나면서 가장 큰 보상을 얻기 위해 <span style="background:rgba(240, 200, 0, 0.2)">정책policy</span> 라 부르는 최상의 전략을 스스로 학습. 



## <font color="#c0504d">입력 데이터의 스트림으로부터 점진적으로 학습할 수 있는지 여부에 따른 분류</font>
### 배치 학습 batch learning
<span style="background:rgba(240, 200, 0, 0.2)">가용한 데이터를 모두 사용해 훈련</span>
- 시간과 자원(cpu, 메모리 공간, 디스크 공간...) 많이 소모 -> 오프라인 수행
**오프라인 학습** offline learning
- 모델의 성능은 시간에 따라 감소
	-> 모델 부패 model rot or 데이터 드리프트 drift
		-> 정기적 재훈련 필요
		
### 온라인 학습 online learning (점진적 학습 incremental learning)
데이터를 순차적으로 한 개씩 또는 미니배치 mini-batch라 부르는 <span style="background:rgba(240, 200, 0, 0.2)">작은 묶음 단위로 주입.</span>
- 매 학습 단계가 빠르고 비용이 적게 든다. 
어디에 적합?
	- 빠른 변화에 적응해야 하는 시스템
	- 컴퓨팅 자원이 제한된 경우 (모바일)
**외부 메모리 학습**out-of-core learning
	- 보통 오프라인으로 실행됨. '온라인' 학습과 헷갈리지 말 것.
- 중요한 파라미터: **학습률**learning rate
	- 변화하는 데이터에 얼마나 빠르게 적응할 것인지
	- 학습률 high: 데이터에 빠르게 적응/ 금방 잊어버림
	- 학습률 low: 데이터에 느리게 학습/ 덜 민감



## <font color="#c0504d">어떻게 일반화되는가에 따른 분류</font>
### 사례 기반 학습instance-based learning
<span style="background:rgba(240, 200, 0, 0.2)">단순히 기억하는 것.</span>
- 훈련 샘플 기억 -> 유사도 측정을 사용해 new and trained 비교하는 식으로 일반화

### 모델 기반 학습model-based learning
<span style="background:rgba(240, 200, 0, 0.2)">샘플들의 모델</span>을 만들어 예측에 사용
- 모델 선택model selection - ex) [[선형 모델linear model]] 채택
	- 선형 모델의 기반의 예측과 비슷한 값을 내는 k-최근접 이웃k-nearest neighbors 회귀도 있음.- 사례 기반의 학습 알고리즘
- 모델 파라미터 조정



# 머신러닝의 주요 도전과제 (문제&해결법)
### 주요 작업 <모델을 선택해 어떤 데이터에 훈련시키는 것>
### 문제될 수 있는 것: '나쁜 모델', '나쁜 데이터'

1. 충분하지 않은 양의 훈련 데이터
2. 대표성이 없는 훈련 데이터
	대표하는 훈련 세트를 사용하기 어려운 이유
		1. 샘플이 작으면 **샘플링 잡음**sampling noise (우연에 의한 대표성 없는 데이터) 발생
		2. **샘플링 편향**sampling bias 매우 큰 샘플도 표본 추출 방법이 잘못되면 대표성을 띠지 못할 수 있다.
3. 낮은 품질의 데이터
	- 오류, 이상치, 잡음 가득
	- 훈련 데이터 정제 필요
4. 관련없는 특성
	- 특성 공학feature engineering: 훈련에 좋은 특성 찾기
		- 특성 선택feature selection
		- 특성 추출feature extraction: 특성 결합
		- 데이터 수집
5. 훈련 데이터 과대적합overfitting
	훈련 데이터의 양과 잡음에 비해 모델이 너무 복잡할 때
		  ex) 고차원 다항 회귀 모델 vs 선형 모델
		 - 단순화: 파라미터 수가 적은 모델 선택, 특성 수 줄이기, **규제**regulation - 모델에 제약
			 - 규제의 양은 **[[하이퍼파라미터hyperparameter]]가 결정
		 - 더 많은 데이터
		 - 잡음 줄이기
6. 훈련 데이터 과소적합underfitting
	모델이 너무 단순해서 데이터의 내재된 구조를 학습하지 못할 때
	- 모델 파라미터가 더 많은 모델 선택
	- 학습 알고리즘에 더 좋은 특성 제공(특성 공학)
	- 모델의 제약을 줄인다. 하이퍼파라미터 감소



- [[믿기 힘든 데이터의 효과]]


