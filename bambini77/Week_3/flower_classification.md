# 4주차 Summary


## 베이스라인에서 바꾼것

```python
def data_augment(image, label):
# Thanks to the dataset.prefetch(AUTO)
# statement in the next function (below), this happens essentially
# for free on TPU. Data pipeline code is executed on the "CPU"
# part of the TPU while the TPU itself is computing gradients.
image = tf.image.random_flip_left_right(image)
image = tf.image.random_flip_up_down(image)
image = tf.image.random_brightness(image, max_delta=0.1)
image = tf.image.random_saturation(image, lower=0.7, upper=1.3)
image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
#image = tf.image.random_saturation(image, 0, 2)
return image, label
```

data augementation을 추가적으로 여러가지를 추가해주었다.

```python
with strategy.scope():
pretrained_model = tf.keras.applications.DenseNet201(
weights='imagenet',
include_top=False ,
input_shape=[*IMAGE_SIZE, 3]
)
#vgg 말고 다른거 해보기
pretrained_model.trainable = False
#batch normalization 추기 + True로 바꿔보기
for layer in pretrained_model.layers[-3:]:
layer.trainable = True
#dropout 추가
model = tf.keras.Sequential([
# To a base pretrained on ImageNet to extract features from images...
pretrained_model,
# ... attach a new head to act as a classifier.
tf.keras.layers.GlobalAveragePooling2D(),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.Dense(512, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.001)),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.Dropout(rate=0.2),
tf.keras.layers.Dense(len(CLASSES), activation='softmax')
])
```

1. pretraidned_model에서 마지막 3개 layer를 훈련 가능하게 바꾸어주었다
2. 배치 정규화층을 추가하였다
3. dense layer층을 추가하였다 (여기에 l2규제를 추가해주었다). 이후에 배치 정규화 layer을 하나 더 추가 하였다
4. dropout을 추가하였다.

```python
model.compile(
optimizer='nadam',
loss = 'sparse_categorical_crossentropy',
metrics=['sparse_categorical_accuracy'],
)
#optimizer lion 사용 찾아보기
model.summary()
```

optimizer를 adam에서 nadam으로 변경하였다

```python
def exponential_lr(epoch,
start_lr = 0.0001, min_lr = 0.0001, max_lr = 0.0005,
rampup_epochs = 20, sustain_epochs = 0,
exp_decay = 0.8):
```

rampup_epochs를 20으로 변경하였다

```python
early_stop = EarlyStopping(
monitor='val_loss',     # 성능을 모니터링할 지표
patience=5,             # 성능 향상이 없는 에포크를 얼마나 기다릴 것인지
verbose=1,              # 얼리 스톱핑이 시작될 때 로그를 출력할지 여부
restore_best_weights=True  # 가장 좋은 모델의 가중치를 복원할 것인지
)
```

epoch 값을 100으로 맞춘뒤 earlystopping을 걸어주었다

![Untitled](4%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20Summary%20e88458b5f6f0483d98a94fa645acd671/Untitled.png)

최고 f1_score ⇒ 0.865

## 알게된 점?

추가적으로 layer를 추가할수록 오히려 점수 떨어짐

lr 맞추기가 굉장히 힘듦 → 최적?

전이학습을 통해 진행되는데 pretrained layer가 훈련이 안되게 False로 맞춰져 있는데 일부를 통해 True로 변경
