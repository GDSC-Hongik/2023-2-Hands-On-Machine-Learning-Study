# 대회 회고

## 📌변경한 점

---

1. 데이터 사이즈 변경

```python
IMAGE_SIZE = [224, 224]
GCS_PATH = GCS_DS_PATH + '/tfrecords-jpeg-224x224'
AUTO = tf.data.experimental.AUTOTUNE
```

- DenseNet121 모델이 224x224 사이즈의 이미지로 학습되었기 때문에 이에 맞춰 image size를 224x224로 선택하였다.
- image size가 작을수록 학습 속도가 더 빨랐다!

1. DenseNet121 모델 변경, model.trainable = True로 변경, 배치정규화&드롭 아웃 레이어 추가

```python
EPOCHS = 100

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

- 여러 모델들을 실험해본 결과 DenseNet121 모델이 가장 좋게 나옴
- model.trainable을 True로 바꿨을 때, False일 때보다는 처음 정확도가 낮게 학습되었으나 epoch가 증가할 수록 정확도가 높아짐
- 배치 정규화와 드롭아웃을 추가하였을 때 정확도가 높아짐

1. Learning Rate Schedule 내 값 변경

```python
def exponential_lr(epoch,
                   start_lr = 0.001, min_lr = 0.0001, max_lr = 0.01,
                   rampup_epochs = 30, sustain_epochs = 0,
                   exp_decay = 0.8):
```

- 기존 start_lr이 너무 작아 0.001로 변경
- min_lr이 0.001일 때는 정확도가 0.86-0.87 이상으로 올라가지 않아 0.0001로 낮춤
- max_lr도 처음엔 0.05로 했으나 값이 너무 큰 것 같아 0.01로 변경

1. earlyStopping 추가

```python
from tensorflow.keras.callbacks import EarlyStopping

# Define training epochs
EPOCHS = 100
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

# Define Early Stopping Callback
early_stopping_callback = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=3,            # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True,  # Restore model weights from the epoch with the best value of the monitored quantity
)

history = model.fit(
    ds_train,
    validation_data=ds_valid,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    callbacks=[lr_callback],
)
```

- epoch을 100으로 놓고 3번 epoch가 진행되는 동안 값이 증가하지 않으면 멈추도록 설정
- 이상하게 잘 작동은 안하고 걍 100번 항상 다 돈 듯,,?

![Untitled](%E1%84%83%E1%85%A2%E1%84%92%E1%85%AC%20%E1%84%92%E1%85%AC%E1%84%80%E1%85%A9%2031a27a9739c94621894e0304661b3cce/Untitled.png)