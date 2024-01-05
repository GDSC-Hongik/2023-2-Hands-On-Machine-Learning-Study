# Question&Answer Set

### Q1. 모델이 훈련데이터에서의 성능은 좋지만 새로운 샘플에서의 일반화 성능이 나쁘다면 어떤 문제가 있는 건가요? 해결책 세 가지는 무엇인가요?
### A1. 훈련데이터셋에 과적합이 되었을 것이다. 과적합이란 데이터셋수에 비해 그래프가 복잡해졌다는 것을 의미하며 이를 해결하기위해서는 1. 데이터 수를 늘린다. 2.그래프를 단순하게 만든다. 3.데이터 복잡성을 추가한다. 

### Q2. 데이터의 불균형한 특성을 고려할 때, StratifiedKFold 교차 검증을 적용하는 과정에서 발생할 수 있는 문제점은 무엇이며, 이를 어떻게 해결할 수 있을까요? 
### A2. 데이터가 불균형하다는 것은 특정 클래스의 데이터 수가 상대적으로 부족하다는 뜻이다. StratifiedKFold를 하는 것은 안그래도 부족한 특정클래스를 k만큼 나누어 각 fold마다 학습 데이터셋과 같은 비율의 class를 유지해주었다는 것인데, 이렇게 되면 그 소수 class에 대한 학습이 진행이 어렵다는 문제가 있다. 이런 class imbalance문제는 그 불균형한 특정 데이터셋 수를 단순히 늘리는 oversampling을 진행하여 충분히 학습할 수 있도록 해주어 해결할 수 있다. 

### Q3. 정확도를 분류기의 성능 측정 지표로 선호하지 않는 이유를 작성해주세요. (Precision과 Recall이라는 키워드를 써서 작성해주세요)
### A3. 정확도는 단순하게 accuracy는 모든 경우에서 내가 얼만큼 맞췄는지에 관심이 있는데 이렇게 되면 내가 1개의 답을 맞추기 위해 다트를 100번 던지더라도, 100점, 1개의 답을 맞추기 위해 다트를 1번 던지더라도 100점인 같은 점수가 나오기때문에 Precision을 고려해줘야하고, Precision만 고려하다보면 10개의 정답을 맞추었는데 알고보니 10개만 정답인 경우에도, 정답이 총 100개있는 경우에도 둘다 같은 precision이 나오다보니 recall도 중요해졌다. 따라서 precision과 recall모두 조화롭게 높은 값을 원하는 것 같아서 F1 score와 같은 다른 Metric으로 평가를 하는 것 같다. (비유가 부족 -> 이해는 가지만 Negative라고 예측한 경우는 예시에서 찾을 수 없다는 점)

### Q4. 경사하강법을 사용하여 모델을 학습할 때, 조기 종료 규제가 무엇을 의미하며, 이 규제가 왜 필요한지에 대해 설명해주세요.(예측오차와 과대적합이라는 키워드를 써서 설명해주세요)
### A4. 모델을 학습하다보면 train dataset의 평가 metric은 줄어드는데 test dataset의 평가 metric은 줄어들지 않는 경우 모델이 train dataset에 너무 과대적합이 되어있다고 보다 조기 종료를 하는 것을 말합니다. 다시 말해 예측오차가 커지는 경우에 조기종료규제를 진행합니다. 