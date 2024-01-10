Q1. Non-Linearity라는 말의 의미와 그 필요성은?(3줄 분량으로 작성해주세요)

A1. Non-Linearity라는 단어는 "비선형성"이라는 의미를 갖고 있다.
신경망의 activation function을 구성할 때, linear한 함수들로만 구성하게 된다면 모델 자체가 결론적으로 하나의 linear function으로서의 역할 밖에 하지 못한다. 비선형성을 가진 activation function의 사용을 통해 더욱 복잡한 뉴런간의 관계에서 의미있는 연산을 가능케 할 수 있다.

Q2. 미니배치를 작게 할때의 장단점은?(2줄 분량으로 작성해주세요)

A2. 장점 : 미니배치를 작게 할 수록 iteration을 한번 수행하는데 걸리는 시간이 적어진다.
단점 : 배치가 작아질수록 업데이트되는 기울기의 안정성이 떨어진다. (이리저리 튈 수 있다)

Q3. 배치 정규화의 장점과 단점은 무엇인가요?

A3. 장점 : 입력의 분포를 안정화할 수 있다. -> 최적화 과정이 개선됨
단점 : 정규화하는 과정을 거쳐야하니 그만큼 계산 시간(cost)가 더 발생한다..?

Q4. 학습률 스케줄링이란 무엇이며, 왜 사용되나요?

A4. 학습률 스케줄링이란, 훈련을 진행하는데 매우 중요한 역할을 하는 "학습률"을 컨트롤 하기위하여 여러 전략을 통해 학습률을 추적 및 관리하는 것이다. 보통 훈련하는 동안 학습률을 감소시키는 방향의 적절한 전략이 존재한다.