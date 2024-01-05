# 10. 케라스를 사용한 인공 신경망 소개

## 후버손실

- 평균 제곱 오차와 평균 절대 오차를 조합한 손실값

## 활성화함수

## 소프트플러스 함수


ReLU의 부드러운 버전이며 0에서 미분가능한 ReLU라고 받아들이면 된다.

하이퍼파라미터 미세튜닝

- GridSearchCV
    - 사이킷런(Scikit-learn) 라이브러리에서 제공하는 하이퍼파라미터 미세 조정 tool
    - 방법은 특히 모델의 성능에 큰 영향을 미치는 하이퍼파라미터가 많고, 최적의 조합을 예측하기 어려운 경우에 유용
    - 그러나 가능한 조합의 수가 매우 많을 경우, `GridSearchCV`는 많은 시간과 계산 자원을 필요
- RandomizedSearchCV
    - 모든 가능한 하이퍼파라미터 조합을 시도하는 대신 무작위로 선택된 조합을 평가

배치 크기는 모델 성능과 훈련 시간에 영향을 미친다. 

큰 배치를 사용하면 GPU와 같은 하드웨어 가속기를 효율적으로 활용할 수 있다. 

- GPU RAM에 맞는 가장 큰 배치를 사용하기를 권장
- 실전에서 큰 배치를 사용하면 훈련 초기에 불안정하게 훈련되고 작은 배치 크기로 훈련된 모델이 차라리 일반화 성능이 더 좋음 .

## GPU RAM은 뭐야

GPU RAM은 "Graphics Processing Unit Random Access Memory"의 약자로, GPU(그래픽 처리 장치)에 내장된 메모리를 의미합니다.

1. **고속 데이터 처리**: **GPU**는 **병렬 처리**에 최적화되어 있어 대량의 데이터를 동시에 빠르게 처리할 수 있습니다. **GPU RAM**은 이러한 **고속 처리를 지원**하기 위해 설계되었습니다.
2. **그래픽 관련 작업**: GPU는 원래 비디오 게임의 복잡한 그래픽을 렌더링하거나 비디오를 처리하는 데 사용되었습니다. 이러한 작업에 필요한 데이터는 GPU RAM에 저장됩니다.
3. **딥 러닝 및 머신 러닝**: 최근에는 GPU의 강력한 병렬 처리 능력이 인공 지능, 특히 딥 러닝과 머신 러닝 분야에서 중요하게 활용됩니다. 이러한 계산에서 GPU RAM은 **모델의 가중치, 중간 계산 결과, 입력 데이터 등을 저장**하는 데 사용됩니다.
4. **메모리 용량과 대역폭**: GPU RAM은 일반적으로 높은 대역폭을 제공하여 대량의 데이터를 빠르게 GPU 코어로 이동할 수 있습니다. **메모리 용량이 클수록 더 많은 데이터를 동시에 처리**할 수 있습니다.

## 그래픽카드와 GPU는 같은 말이야?

그래픽카드에 GPU가 속하는 개념. GPU가 아닌 그래픽 카드에는 메모리(VRAM)이나 쿨링시스템 정도가 있다. 

## 배치를 크게 할때는 worker라는 CPU 코어도 함께 수를 올려줘야 한다.

이미지 전처리 작업이 CPU에서 수행되는 경우가 많은 이유

1. **작업의 성격**: 이미지 전처리는 종종 순차적이고 분기가 많은 작업을 포함합니다. 예를 들어, 크기 조정, 자르기, 색상 조정 같은 작업은 병렬 처리보다는 순차적 처리에 더 적합할 수 있습니다. 이러한 종류의 작업은 CPU의 성격과 잘 맞습니다.
2. **자원 분배 최적화**: GPU는 고도의 병렬 처리가 필요한 작업, 특히 대규모 수치 계산이나 딥 러닝 모델의 훈련과 같은 작업에 최적화되어 있습니다. 이미지 전처리와 같은 상대적으로 가벼운 작업을 CPU에서 수행함으로써, GPU의 자원을 이러한 복잡한 작업에 집중할 수 있습니다.

## Warming up

- 작은 학습률로 훈련을 시작해서 점점 학습률을 크게 한다.

### **학습률 감소 (Learning Rate Decay):**

1. 학습 과정에서 점차적으로 학습률을 줄이는 것은 모델이 최적의 솔루션에 더 세밀하게 접근하도록 돕기 위해 사용. 초기에 높은 학습률을 사용하면 빠른 학습이 가능하지만, 시간이 지남에 따라 손실 함수의 최소값에 더 가까이 접근하기 위해 학습률을 낮춘다.
2. 학습률 감소는 여러 방법으로 구현될 수 있다. 예를 들어, 일정한 간격으로 학습률을 점차적으로 줄이거나, 성능이 향상되지 않을 때 학습률을 줄이는 방법이 있다.
3. 이 방법은 모델이 초기에 빠르게 학습하고 나중에는 **더 안정적으로 수렴**하도록 돕습니다.  과적합(overfitting)의 위험을 줄이고, 모델의 일반화 성능을 향상시킬 수 있습니다.

### **워밍업 (Warm-Up):**

1. 워밍업은 학습률을 처음에는 낮게 시작하여 점차적으로 증가시키는 방법이다. 모델의 가중치가 초기에 불안정할 때 너무 큰 업데이트를 방지하기 위해 사용된다.
2. 워밍업 기간 동안, 학습률은 낮은 값에서 시작하여 점차적으로 목표 학습률까지 증가함. 이 초기 단계는 모델이 데이터의 기본적인 특성을 학습하고, 최적화 과정을 안정화시키는 데 도움이 된다.
3. 워밍업은 특히 큰 학습률을 사용하는 모델에서 중요. 모델이 처음부터 너무 큰 학습률로 인해 발생할 수 있는 발산(divergence)이나 오버슈팅(overshooting)을 방지.

# 11장

## 그래디언트 소실 및 폭주 문제

- 알고리즘이 하위 층으로 진행될수록 그레이디언트가 점점 작아지는 경우가 생긴다. 경사하강법이 하위 층의 연결 가중치를 변경되지 않은 채로 둔다면 좋은 솔루션으로 수렴하지 않을 것.
- 그레이디언트가 점점 커져서 여러 층이 비정상적으로 큰 가중치로 갱신되면 알고리즘은 발산한다.

## 글로럿과 He 초기화

**<아이디어>** 양방향 신호가 적절하게 흐르기 위해서는

- 각 층의 출력에 대한 분산이 입력에 대한 분산과 같아야 한다.
- 역방향에서 층을 통과하기 전과 후의 그레디언트 분산이 동일해야 한다.

## ****글로럿 초기화****

- 입력과 출력의 노드 수를 고려하여 가중치의 분산을 조절
- 글로럿 초기화는 특히 하이퍼볼릭 탄젠트(tanh)나 로지스틱 시그모이드 같은 대칭적이고 제한된 활성화 함수와 잘 어울린다.
- ReLU 활성화 함수와 같이 비대칭적인 활성화 함수를 사용할 때는 He 초기화(또는 Kaiming 초기화)와 같이 다른 초기화 방법을 고려하는 것이 좋다.

## 고급 활성화 함수

- LeakyReLU
    - 훈련하는 동안 일부 뉴런은 0이외의 값을 출력하지 않아서 dying ReLU라고 알려진 문제를 해결하려 등장
    - ReLU의 변형
    - 속도는 ReLU가 굉장히 빠름.
    - 대규모 이미지 셋에서 ReLU보다 성능이 크게 앞서지만 소규모 데이터셋에서는 과대적합될 위험이 있음.
        
- ELU 활성화함수
    - ReLU 의 변형 성능을 앞지름
    - 지수함수를 사용하므로 ReLU보다 계산이 느림
    - 훈련하는 동안 **수렴속도가 빨라서 느린 계산이 상쇄**되지만 테스트시에는 느릴 것임.
        
        ![스크린샷 2024-01-02 오후 10.38.44.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/f1b77830-6c23-45e2-9256-e7bac5ca202d/54e4f224-a1a6-486f-a1e1-8cd195ffc81e/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-01-02_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_10.38.44.png)
        
        ![스크린샷 2024-01-02 오후 10.38.28.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/f1b77830-6c23-45e2-9256-e7bac5ca202d/b41116d3-edef-45d8-8cb1-a3184116b0e3/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-01-02_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_10.38.28.png)
        
- SiLU
- Swish
- GELU
- Mish

## 배치 정규화

각 층에서 활성화함수를 통과하기 전이나 후에 모델에 연산을 하나 추가 (스케일 조정 및 이동)

- 배치 정규화로 인해 그레이디언트 소실 문제 감소하여 tanh, 시그모이드 같은 수렴성을 가진 활성화함수 사용가능.

## 그래디언트 클리핑

- 역전파시 특정 임계값을 넘어서지 못하게 그레이디언트를 잘라내는 것

비지도 사전 훈련

고속옵티마이저

- 모멘텀최적화
- 네스테로프 가속경사
- AdaGrad
- RMSProp
- Adam
- Adamax
- Nadam
- AdamW

학습률스케쥴링

- 거듭제곱 기반 스케쥴링
- 지수기반 스케쥴링
- 구간별 고정 스케쥴링
- 성능 기반 스케쥴림
- 1사이클 스케쥴링

## 규제로 과대적합 피하기

- L_1과 L_2규제
- 드롭아웃
- 몬테 카를로 드롭아웃
- 맥스-노름 규제