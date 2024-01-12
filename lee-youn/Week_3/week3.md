# 3주차 Summary

생성일: 2023년 12월 17일 오전 1:14
주차: 3주차
깃허브 커밋: No
사람: 윤정 이
완료 여부: 완료

<aside>
📌 키워드 요약

</aside>

## **12장.** 텐서플로를 사용한 사용자 정의 모델과 훈련

---

### 넘파이처럼 텐서플로 사용하기

- **텐서와 연산**
    - tf.constant() 함수 → 텐서 만들 수 있음
    - 필요한 모든 기본 수학 연산과 넘파이에서 볼 수 있는 대부분의 연산을 제공함
        - tf.add()
        - tf.multiply()
        - tf.square()
        - tf.exp()
        - tf.sqrt()
        - tf.reshape()
        - tf.squeeze()
        - tf.tile()
- **타입 변환**
    - 타입 변환은 성능을 크게 감소시킬 수 있다
    - 텐서플로는 어떤 타입 변환도 자동으로 수행하지 않는다
    - 진짜 타입 변환이 필요할 때 ⇒ tf.cast() 함수 사용
- **변수**
    - 일반적인 텐서로는 역전파로 변경되어야 하는 신경망의 가중치를 구현할 수 없음
    - tf.Variable 사용
    - assign() 메서드를 사용하여 변숫값 바꿀 수 있음
    - 직접 수정은 불가능
- **다른 데이터 구조**
    - 희소 텐서(tf.SparseTensor)
        - 대부분 0으로 채워진 텐서를 효율적으로 나타냄
    - 텐서 배열(tf.TensorArray)
        - 텐서의 리스트
        - 기본적으로 고정된 길이를 가지지만 동적으로 바꿀 수 있음
    - 래그드 텐서(tf.RaggedTensor)
        - 텐서의 크기가 달라지는 차원 ⇒ 래그드 차원
    - 문자열 텐서
        - tf.string 타입의 텐서
        - 유니코드 아니라 **바이트 문자열** 나타냄
    - 그 밖에 집합 큐 등등

### 사용자 정의 모델과 훈련 알고리즘

- 사용자 정의 손실 함수
    - 후버 손실 사용
        - 레이블과 모델의 예측을 매개변수로 받는 함수 만들고,
        - 텐서플로 연산을 사용해 (각 샘플의) 손실을 모두 담은 텐서를 계산
- 사용자 정의 요소를 가진 모델을 저장하고 로드하기
    - 모델을 로드할 때는 함수 이름과 실제 함수를 매핑한 딕셔너리를 전달해야 함
- 활성화 함수, 초기화, 규제, 제한을 커스터마이징하기
    - 함수가 모델과 함께 저장해야 할 하이퍼파라미터를 가지고 있다면 적절한 클래스를 상속함
- 사용자 정의 지표
    - 손실은 모델을 훈련하기 위해 경사 하강법에서 사용되므로 미분 가능해야 하고 그레이디언트가 모든 곳에서 0이 아니어야 함
    - 지표는 모델을 평가할 때 사용되며 훨씬 이해하기 쉬워야 함
    - 미분이 가능하지 않거나 모든 곳에서 그레이디언트가 0이어도 괜찮음
    - **스트리밍 지표** : 배치마다 점진적으로 업데이트 됨
    - reset_states() ⇒ 변수를 초기화하는 메서드

### 사용자 정의 모델과 훈련 알고리즘

- 텐서플로에는 없는 특이한 층을 가진 네트워크를 만들어야 할 때 ⇒ 사용자 정의 층 만듦
- 훈련과 테스트에서 다르게 작동하는 층이 필요할 때
    - call()메서드에 training 매개변수를 추가하여 훈련인지 테스트인지를 결정해야 함

- 자동 미분으로 그레이디언트 계산하기
    - gradient() 메서드를 한 번 이상 호출해야 한다면 지속 가능한 테이프를 만들고 사용이 끝난 후 테이프를 삭제하여 리소스를 해제해야 함
    - 대부분의 경우 그레이디언트 테이프는 여러 값에 대한 한 값의 그레이디언트를 계산하는 데 사용
    - 벡터에 있는 각 손실마다 후진 자동 미분 수행
    - 신경망의 일부분에 그레이디언트가 역전파되지 않도록 막을 필요가 있다면 ⇒ tf.stop_gradient() 함수 사용
    
- 사용자 정의 훈련 반복
    - 모델이 훈련과 테스트 중에 다르게 작동하는 경우, 훈련 반복안에서 모델을 호출할 때 training=True를 지정하는 것을 잊지 말기!
    

### 텐서플로 함수와 그래프

## KAGGLE. Flower Classification

**1) VGG16 → ResNet50으로 변경**

**2) start_lr, min_lr, max_lr 변경**

```python
def exponential_lr(epoch,
                   start_lr = 0.1, min_lr = 0.001, max_lr = 0.05,
                   rampup_epochs = 3, sustain_epochs = 0,
                   exp_decay = 0.8):
```

![Untitled](3%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20Summary%20bad8e9be57214c60a588e05a75c87f63/Untitled.png)

![Untitled](3%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20Summary%20bad8e9be57214c60a588e05a75c87f63/Untitled%201.png)

⇒ f1은 0.120으로 높아졌으나 그래도 목표치(0.9)보다는 낮은 수치

⇒ accuracy그래프와 loss그래프 추이를 봤을 때 epoch를 더 증가시키면 나은 성과가 있을것이라 추측

**3) epoch 12 → 50**

![Untitled](3%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20Summary%20bad8e9be57214c60a588e05a75c87f63/Untitled%202.png)

![Untitled](3%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20Summary%20bad8e9be57214c60a588e05a75c87f63/Untitled%203.png)

⇒ f1높아졌긴 했는데,,시간 소요 대비 결과가 그닥 좋진 않

⇒ 개망!

**4) ResNet50 → VGG16으로 변경(아마, ResNet50은 fully connect(?) layer가 필요없어서 이렇게 모델 가져다가 쓰는 걸로는 좀 까다로운 듯,,)**

**5) min_lr을 0.01로 변경 (증가 추이가 너무 느리게 진행되어서!)**

```python
def exponential_lr(epoch, start_lr = 0.1, min_lr = 0.01, max_lr = 0.05,
		rampup_epochs = 3, sustain_epochs = 0, exp_decay = 0.8):
```

```python
EPOCHS = 12

with strategy.scope():
    pretrained_model = tf.keras.applications.VGG16(
        weights='imagenet',
        include_top=False ,
        input_shape=[*IMAGE_SIZE, 3]
    )
    pretrained_model.trainable = False
    
    model = tf.keras.Sequential([
        # To a base pretrained on ImageNet to extract features from images...
        pretrained_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(CLASSES), activation='softmax')
    ])
```

**6) 이것저것 ㅊ추가 및 변경**

```python
# Learning Rate Schedule for Fine Tuning #
def exponential_lr(epoch,
                   start_lr = 0.05, min_lr = 0.001, max_lr = 0.01,
                   rampup_epochs = 5, sustain_epochs = 0,
                   exp_decay = 0.5):

    def lr(epoch, start_lr, min_lr, max_lr, rampup_epochs, sustain_epochs, exp_decay):
        # linear increase from start to rampup_epochs
        if epoch < rampup_epochs:
            lr = ((max_lr - start_lr) /
                  rampup_epochs * epoch + start_lr)
        # constant max_lr during sustain_epochs
        elif epoch < rampup_epochs + sustain_epochs:
            lr = max_lr
        # exponential decay towards min_lr
        else:
            lr = ((max_lr - min_lr) *
                  exp_decay**(epoch - rampup_epochs - sustain_epochs) +
                  min_lr)
        return lr
    return lr(epoch,
              start_lr,
              min_lr,
              max_lr,
              rampup_epochs,
              sustain_epochs,
              exp_decay)

lr_callback = tf.keras.callbacks.LearningRateScheduler(exponential_lr, verbose=True)

rng = [i for i in range(EPOCHS)]
y = [exponential_lr(x) for x in rng]
plt.plot(rng, y)
print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))
```

![Untitled](3%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20Summary%20bad8e9be57214c60a588e05a75c87f63/Untitled%204.png)

```python
EPOCHS = 12

with strategy.scope():
    pretrained_model = tf.keras.applications.VGG16(
        weights='imagenet',
        include_top=False ,
        input_shape=[*IMAGE_SIZE, 3]
    )
    pretrained_model.trainable = False
    
    model = tf.keras.Sequential([
        # To a base pretrained on ImageNet to extract features from images...
        pretrained_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(256,activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(CLASSES), activation='softmax')
    ])
```

⇒ 배치 정규화 레이어 추가

![Untitled](3%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20Summary%20bad8e9be57214c60a588e05a75c87f63/Untitled%205.png)

정확도가 처음에 확 높아지다가 점차 차이 폭(?)이 줄어드는게 보임

⇒ epoch를 조금 증가시키고 lr을 좀 더 높여보자

```python
def exponential_lr(epoch,
                   start_lr = 0.05, min_lr = 0.001, max_lr = 0.01,
                   rampup_epochs = 6, sustain_epochs = 0,
                   exp_decay = 0.5):
```

rampup_epochs를 6으로 변경함

![Untitled](3%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20Summary%20bad8e9be57214c60a588e05a75c87f63/Untitled%206.png)

너무 정확도가 높게 시작하는 것 같더니,,갑자기 하락하네 lr 다시 변경,,

![Untitled](3%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20Summary%20bad8e9be57214c60a588e05a75c87f63/Untitled%207.png)

lr 변경후에도 값의 차이가 크게 생기지 않아서 lr를 크게 다시 변경

너무 느리게 accuracy가 올라가는 이슈로 다시 lr 변

dropout 레이어를 넣어서 문제가 된듯

dropout 레이어 제거, epoch 30으로 변

```python
EPOCHS = 12

with strategy.scope():
    pretrained_model = tf.keras.applications.VGG16(
        weights='imagenet',
        include_top=False ,
        input_shape=[*IMAGE_SIZE, 3]
    )
    pretrained_model.trainable = False
    
    model = tf.keras.Sequential([
        # To a base pretrained on ImageNet to extract features from images...
        pretrained_model,
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(len(CLASSES), activation='softmax')
    ])
```

```python
# Learning Rate Schedule for Fine Tuning #
def exponential_lr(epoch,
                   start_lr = 0.1, min_lr = 0.001, max_lr = 0.05,
                   rampup_epochs = 7, sustain_epochs = 0,
                   exp_decay = 0.5):
```

```python
# Define training epochs
EPOCHS = 30
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

history = model.fit(
    ds_train,
    validation_data=ds_valid,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    callbacks=[lr_callback],
)
```

![Untitled](3%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20Summary%20bad8e9be57214c60a588e05a75c87f63/Untitled%208.png)

7) epoch를 30으로 동일하게 변경하고, 이미지 크기가 작은 데이터 사용

## KAGGLE. Flower Classification - 최종

---

### **[기존 BaseLine과 달라진 점]**

1. 이미지 데이터 사이즈 변경 : 224x224 사이즈로 변경

```python
IMAGE_SIZE = [224, 224]
GCS_PATH = GCS_DS_PATH + '/tfrecords-jpeg-224x224'
```

1. model 변경 : DenseNet121모델 사용

                          : 배치정규화 레이어 추가

                    : dropout 레이어 추가

```python
with strategy.scope():
    pretrained_model = tf.keras.applications.densenet.DenseNet121(
        weights='imagenet',
        include_top=False ,
        input_shape=[*IMAGE_SIZE, 3]
    )
    pretrained_model.trainable = True
    
    model = tf.keras.Sequential([
        # To a base pretrained on ImageNet to extract features from images...
        pretrained_model,
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(len(CLASSES), activation='softmax')
    ])
```

1. learning rate 변경

```python
# Learning Rate Schedule for Fine Tuning #
def exponential_lr(epoch,
                   start_lr = 0.001, min_lr = 0.0001, max_lr = 0.01,
                   rampup_epochs = 30, sustain_epochs = 0,
                   exp_decay = 0.8):
```

![Untitled](3%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20Summary%20bad8e9be57214c60a588e05a75c87f63/Untitled%209.png)