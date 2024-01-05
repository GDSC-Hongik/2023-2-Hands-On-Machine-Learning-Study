# 10장 케라스를 사용한 인공 신경망 소개
## 퍼셉트론

단순 퍼셉트론은 결정 경계가 선형이므로 복잡한 패턴을 가진 문제, 데이터가 비선형인 문제, XOR문제를 풀 수 없었다.

이때 퍼셉트론을 여러 개 쌓아 올린 다층 퍼셉트론(MLP)이 XOR문제를 해결할수 있고 계단 함수대신 시그모이드 함수나 ReLU 함수를 사용함으로써 역전파 알고리즘으로 학습가능하며 비선형성이 추가되어 복잡한 문제또한 해결할수 있다.

### 회귀에서의 MLP

- 손실 함수로 보통 mse를 사용하지만 이상치가 많은 경우 mae 또는 두가지를 조합한 후버 손실을 사용한다.

### 분류에서의 MLP

- 분류의 종류에 따라 출력 뉴런수와 출력층의 활성화 함수를 바꿔 사용한다.

## 케라스를 사용하여 MLP 구현하기

- 케라스는 텐서플로 이외에 여러 백엔드를 지원하였지만 현재는 텐서플로 전용으로 사용되고 있다. 사실상 케라스와 텐서플로는 하나이다.

### 모델 만들기

- Sequential() 를 사용해서 시퀀셜 모델을 만들수 있습니다. 이는 적어둔 층을 순서대로 실행합니다.
- 특이한 점은 입력 차원을 처음에만 적어주고 그 뒤 부터는 출력 차원만 적어주는 것입니다.

### 모델 컴파일

- 다중 분류에서 레이블이 정수라면 sparse_categorical_crossentropy 사용하고 원핫벡터라면 categorical_crossentropy를 사용해야합니다.

### 모델 훈련 및 평가

- fit() 메서드를 사용해서 모델 훈련을 진행한다.
- fit() 메서드가 반환하는 History 객체에는 훈련동안의 정보가 담겨있습니다.
    - 훈련 파라미터
    - 훈련 세트와 검증 세트에 대한 손실과 측정한 지표
- 검증 손실이 여전히 감소한다면 모델 수렴을 위해 학습을 추가로 진행합니다
- 모델 성능 상승을 위해서는 하이퍼 파라미터를 튜닝해야합니다.
    - 학습률
    - 옵티마이저
    - 모델 구조 등
- evaluate() 메서드를 사용하여 테스트를 진행합니다.

### 모델 예측

- predict() 메서드를 사용해서 샘플에 대한 예측을 만듭니다.
    - 출력 결과가 클래스마다 각각의 확률이라면 argmax() 메서드로 가장 높은 확률의 클래스를 얻습니다.

### 서브클래싱 API로 만드는 동적 모델

- Model 클래스를 상속받아 원하는 모델을 작성합니다.
- call() 메서드에 단순히 층을 나열하는 것뿐 아니라 if 문, for 문, 저수준 텐서플로 연산 등을 작성할 수 있습니다.

### 모델 저장

- save() 메서드를 사용해서 모델을 저장할 수 있습니다.
    - 저장 포맷으로는 tf, h5 가 있는데 대부분의 텐서플로 배포 도구는 tf를 사용하므로 이것이 선호됩니다.

# 11장 심층 신경망 훈련
> 대규모 신경망을 학습시키며 발생하는 4가지 문제점을 해결하는 방법을 알아봅시다.
> 

## 1. 그레디언트 소실 & 폭주 문제 해결

### 불안정한 그레디언트

- 신경망이 깊어지면서 출력층과 입력층과의 거리가 멀어져 그레디언트가 하위층으로 갈수록 작아지거나 커지는 현상이 발생하여 훈련을 어렵게 만듭니다.
    - 이유: 시그모이드 활성화 함수와 N(0,1) 가중치 초기화 방식을 사용할때 출력의 분산 > 입력의 분산 → 위층으로 갈수록 분산이 커져 활성화 함수가 0,1 로 수렴하여 기울기가 0에 가까워져 전파할 그레디언트가 없다.

### 초기화 방법

적절한 신호가 흐르기 위해서는 입력의 분산과 출력의 분산이 같아야한다.

층의 입력과 출력 연결 개수를 각각 fan_in, fan_out이라고 하고 둘의 산술평균을 fan_avg라고 하자

- 세이비어 초기화(Xavier)
    - 평균이 0이고 분산이 1/fan_avg 인 정규분포
    - 활성화 함수: 시그모이드, 하이퍼볼릭탄젠트 함수를 사용할때 효율적이다.
    - ReLU 계열 함수를 사용할경우 성능이 좋지 않다.
- He 초기화(카이밍 초기화)
    - 평균이 0이고 분산이 2/fan_in 인 정규분포
    - 활성화 함수: ReLU 계열 함수를 사용할때 효율적이다.
- 파이토치에서 nn.linear 이나 nn.Conv2d에서는 가중치를 균등분포로 카이밍 초기화 방법을 사용하고 있다.
    - linear([링크](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py))
        
        ```python
        def reset_parameters(self) -> None:
                # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
                # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
                # https://github.com/pytorch/pytorch/issues/57109
                init.kaiming_uniform_(self.weight, a=math.sqrt(5))
                if self.bias is not None:
                    fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    init.uniform_(self.bias, -bound, bound)
        ```
        
    - CovNd([링크](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/conv.py#L40))
        
        ```python
        def reset_parameters(self) -> None:
                # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
                # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
                # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
                init.kaiming_uniform_(self.weight, a=math.sqrt(5))
                if self.bias is not None:
                    fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
                    if fan_in != 0:
                        bound = 1 / math.sqrt(fan_in)
                        init.uniform_(self.bias, -bound, bound)
        ```
        

### 배치 정규화

- 입력을 원점에 맞추고 정규화 한다음, 파라미터로 결과값의 스케일을 조정합니다.
- 배치 정규화라고는 하지만 표준화에 가깝다.
- 방법
    1. 먼저 미니배치에 대하여 평균과 표준편차를 구하고 N(0,1)를 따르도록 표준화 한다.
    2. 학습가능한 파라미터 a, b가 있다고 하면 표준화를 거친 x에 대해 결과 z = a * x + b를 계산한다.
    3. 이제 N(b, a^2)를 따르는 미니 배치 인풋이 생성되었다.

## 2. 대규모 신경망을 위한 레이블이 있는 데이터가 적은 경우 학습 방법

## 3. 훈련 속도를 높이는 최적화 기법

## 4. 대규모 신경망 규제 기법