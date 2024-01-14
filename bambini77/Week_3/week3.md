# 3주차 Summary

생성일: December 17, 2023 1:15 AM
주차: 3주차
깃허브 커밋: No
사람: 최윤창
완료 여부: 진행 중

<aside>
📌 키워드 요약

</aside>

# 12장 텐서플로를 사용한 사용자 정의 모델과 훈련

## 텐서플로 훑어보기

텐서플로는 강력한 수치 계산용 라이브러리

- 핵심 구조는 넘파이와 매우 비슷하지만 GPU를 지원한다.
- (여러 장치와 서버에 대해서) 분산 컴퓨팅을 지원한다.
- 일종의 JIT 컴파일러를 포함한다. 속도를 높이고 메모리 사용량을 줄이기 위해 계산을 최적화한다. 이를 위해 파이썬 함수에서 계산 그래프를 추출한 다음 최적화하고 효율적으로 실행한다.
- 계산 그래프는 플랫폼에 중립적인 포맷으로 내보낼 수 있다.
- 텐서플로는 자동 미분 기능과 RMSProp, Nadam 같은 고성능 옵티마이저를 제공하므로 모든 종류의 손실 함수를 최소화할 수 있다

## 넘파이처럼 텐서플로 사용하기

텐서는 일반적으로 다차원 배열 (넘파이 ndarray와 비슷)

- tf.constant()로 텐서를 만듦
    - 크기와 데이터 타입 가진다
    - 인덱스 참조가 넘파이와 매우 비슷하게 작동
    - 모든 종류의 텐서 연산이 가능
    - 스칼라값 가질 수 있음
    - 기본수학 연산과 넘파이에서 볼 수 있는 대부분의 연산 제공
- 넘파이 배열로 텐서 만들고 그 반대도 가능. 서로 연산 적용도 가능
- 타입 변환은 성능을 크게 감소시킬 수 있으므로 자동으로 수행하지 않는다. 필요하다면 tf.cast()함수 사용
- tf.Tensor는 변경이 불가능한 객체 → 일반적인 텐서로는 역전파로 변경되어야 하는 신경망의 가중치 구현 x
    
    → tf.Variable() 사용
    
- **tf.Variable()**
    - assign() 메서드를 사용하여 변숫값 변경
- 희소 텐서, 텐서 배열, 래그드 텐서, 문자열 텐서, 집합, 큐와 같은 여러 데이터 구조 지원

## 사용자 정의 모델과 훈련 알고리즘

### 사용자 정의 손실 함수

- 모델 저장 easy, 모델 로드할 때는 함수 이름과 실제 함수를 매핑한 딕셔너리를 전달
- 사용자 정의 객체를 포함한 모델을 로드할 때는 그 이름과 객체를 매핑해야 한다.

### 활성화 함수, 초기화, 규제, 제한을 커스터마이징하기

```python
def my_softplus(z):
return tf.math.log(1.0 + tf.exp(z))

def my_glorot_initializer(와ape, dtype=tf.float32):
stddev = tf.sqrt(2. / (shape[0] + shape[l]))
return tf.random.normal(shape, stddev=stddev, dtype=dtype)

def my_ll_regularizer(weights):
return tf.reduce_sum(tf.abs(0.01 * weights))

def my_positive_weights(weights): # tf.nn.relu(veights)와 반환값이 같습니다•
return tf.where(weights < 0., tf•zeros_like(weights), weights)
```

만들어진 사용자 정의 함수는 보통의 함수와 동일하게 사용

### 사용자 정의 지표

- **손실 vs 지표**
    - **손실**은 모델을 훈련하기 위해 경사 하강법에서 사용되므로 미분 가능해야 하고 그레디언트가 모든 곳에서 0이 아님
    - **지표는** 모델을 평가할 때 사용. 미분이 가능하지 않거나 모든 곳에서 그레디언트가 0이어도 괜찮다. (주로 평균 절댓값 오차나 평균 제곱 오차를 많이 사용)
    - 대부분 사용자 손실 함수를 만드는 것과 사용자 지표 함수를 만드는 것은 동일
- **스트리밍 지표**

### 사용자 정의 층

- 가중치가 없는 층 (ex>tf.keras.layers.Flatten이나 tf.keras.layers.ReLU)
    - 파이썬 함수를 만든 후 tf.keras.layers.Lambda층으로 감싸기
- 훈련과 테스트에서 다르게 작동하는 층이 필요하다면 call()메서드에 training 매개변수를 추가하여 훈련인지 테스트인지 결정

### 사용자 정의 모델

![Untitled](3%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20Summary%20ac3a3a65235840eb8caabef7621f320a/Untitled.png)

위와 같은 예제 모델

keras.Model 클래스를 상속하여 생성자에서 층과 변수를 만들고 모델이 해야 할 작업을 call() 메서드에 구현

잔차 블록?

### 모델 구성 요소에 기반한 손실과 지표

- 은닉 층의 가중치나 활성화 함수 등과 같이 모델의 구성 요소에 기반한 손실을 정의해야 할 경우도 있음
- 모델 구성 요소에 기반한 손실을 정의하고 계산하여 add_loss() 메서드에 그 결과를 전달

### 자동 미분으로 그레디언트 계산하기

- 각 파라미터가 매우 조금 바뀔 때 함수의 출력이 얼마나 변하는지 측정하여 도함수의 근삿값을 계산하는 것 → 대규모 신경망에서는 적용하기 어려움

```python
def f(w1, w2):
    return 3 * w1 ** 2 + 2 * w1 * w2

w1, w2 = 5, 3; eps = 1e-6
# 각 파라미터가 바뀔 때마다 함수의 출력이 얼마나 변하는지 측정하여 도함수의 근삿값을 계산함
print((f(w1 + eps, w2) - f(w1, w2)) / eps) # 36.000003007075065
print((f(w1, w2 + eps) - f(w1, w2)) / eps) # 10.000000003174137
```

- 후진 모드 자동 미분을 사용

ex. 변수 w1, w2 정의 tf.GradientTape 블록을 만들어 이 변수 와 관련된 모든 연산을 자동으로 기록

```python
w1, w2 = tf.Variable(5.), tf.Variable(3.)
with tf.GradientTape() as tape:
    z = f(w1, w2)
    
gradients = tape.gradient(z, [w1, w2]) 
# tape.gradient(z, [w1, w2])는 출력 z에 대한 가중치 w1과 w2의 그래디언트를 계산
# z가 w1과 w2에 대해 얼마나 민감한지를 나타내는 값입니다.
gradients
# [<tf.Tensor: shape=(), dtype=float32, numpy=36.0>,
#  <tf.Tensor: shape=(), dtype=float32, numpy=10.0>]
```

- 기본적으로 테이프는 변수가 포함된 연산만을 기록. 변수가 아닌 다른 객체에 대한 그레디언트는 None이 반환.
    
    모든 연산을 기록하도록 강제 → watch()함수 사용
    
- tf.stop_gradient()함수 → 신경만의 일부분에 그레디언트가 역전파되지 않도록 막을때 사용.
- 안정적인 함수에도 수치적으로 불안정한 그레디언트 존재할 수 있음
    
    → 자동 미분을 사용하지 않고 그레디언트 계산을 위해 사용할 식을 텐서플로에 알려준다
    
    이를 위해 @tf.custom_gradient 데코레이터를 사용하고 
    

### 사용자 정의 훈련 반복

## 텐서플로 함수와 그래프

![Untitled](3%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20Summary%20ac3a3a65235840eb8caabef7621f320a/Untitled%201.png)

![Untitled](3%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20Summary%20ac3a3a65235840eb8caabef7621f320a/Untitled%202.png)

![Untitled](3%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20Summary%20ac3a3a65235840eb8caabef7621f320a/Untitled%203.png)

![Untitled](3%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20Summary%20ac3a3a65235840eb8caabef7621f320a/Untitled%204.png)

![Untitled](3%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20Summary%20ac3a3a65235840eb8caabef7621f320a/Untitled%205.png)

batch normalization 추가

![Untitled](3%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20Summary%20ac3a3a65235840eb8caabef7621f320a/Untitled%206.png)

배치 정규화 2개 추가 → f1 0.518

![Untitled](3%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20Summary%20ac3a3a65235840eb8caabef7621f320a/Untitled%207.png)

배치정규화 하나 빼고 드롭아웃 추가 →  f1 0.514

![Untitled](3%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20Summary%20ac3a3a65235840eb8caabef7621f320a/Untitled%208.png)

상위 레이어 일부 훈련 가능하게 설정 → f1 0.525

![Untitled](3%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20Summary%20ac3a3a65235840eb8caabef7621f320a/Untitled%209.png)

l2규제 추가 f1→ 0.554