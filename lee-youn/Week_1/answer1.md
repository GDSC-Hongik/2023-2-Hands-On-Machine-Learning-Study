# Question Set

### 1. 모델이 훈련데이터에서의 성능은 좋지만 새로운 샘플에서의 일반화 성능이 나쁘다면 어떤 문제가 있는 건가요? 해결책 세 가지는 무엇인가요?

모델이 훈련데이터에서의 성능은 좋지만 새로운 샘플에서는 일반화 성능이 나쁜 것은, 훈련데이터 대한 과대적합이 발생했기 때문입니다. 이 문제를 해결하기 위해선 모델에 규제를 추가하거나 복잡한 모델을 좀 더 단순화 시키거나 데이터양을 늘리는 방법이 있습니다.


### 2. 데이터의 불균형한 특성을 고려할 때, StratifiedKFold 교차 검증을 적용하는 과정에서 발생할 수 있는 문제점은 무엇이며, 이를 어떻게 해결할 수 있을까요?

StratifideKFold 교차 검증은 기존의 KFold 교차 검증과 다르게 폴드 당 계층별 분포비율을 동일하게 하는 특징이 있습니다. 교차 검증을 적용하는 과정에서 발생할 수 있는 문제점은 아무래도 폴드마다 동일하게 분포 비율이 들어가야 하기 때문에 특정 계층(레이블)의 데이터가 부족하면 충분히 학습시킬 수 없는 것이라 추측됩니다. 이 문제를 해결하기 위해선 데이터 양을 늘리는 과정을 거치거나 폴드 당 분배(?)하는 과정에서 추가적인 처리 작업이 필요할 것이라 생각합니다..


### 3. 정확도를 분류기의 성능 측정 지표로 선호하지 않는 이유를 작성해주세요. (Precision과 Recall이라는 키워드를 써서 작성해주세요)

target이 편향되어 있거나 하는 경우에는 정확도 만으로 모델 성능을 파악하기 어렵습니다. 예를 들어, 이진 분류의 경우 90개의 레이블이 1이고 10개의 레이블이 0이면 예측결과를 1로 찍어도 정확도가 90%가 됩니다. 즉, 아무것도 하지 않고도 높은 정확도 수치가 나타날 수 있습니다.

더 좋은 방법인  '오차 행렬'을 이용한 정밀도(Precision)와 재현율(recall)은 양성 예측의 정확도 및 분류기가 정확하게 감지한 양성 샘플의 비율을 알 수 있어 정확도보다 분류기의 성능을 평가하기 더 좋습니다.


### 4. 경사하강법을 사용하여 모델을 학습할 때, 조기 종료 규제가 무엇을 의미하며, 이 규제가 왜 필요한지에 대해 설명해주세요.(예측오차와 과대적합이라는 키워드를 써서 설명해주세요)

조기 종료 규제는 검증 오차가 최솟값에 도달했을 때 바로 훈련을 중지시키는 것이며, 예측오차가 감소하여 멈추었다가 다시 상승하는 것과 같은 과대적합을 해결하기 위해 예측오차가 최소에 도달하는 즉시 훈련을 멈추는 조기 종료 규제가 필요합니다.