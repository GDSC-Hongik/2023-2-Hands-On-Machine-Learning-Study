# 1주차 Summary

생성일: December 17, 2023 1:15 AM
주차: 1주차
깃허브 커밋: No
사람: 최윤창
완료 여부: 진행 중

<aside>
📌 **키워드 요약**

</aside>

# 1장 한눈에 보는 머신러닝

머신러닝이란?

데이터에서 학습하도록 컴퓨터를 프로그래밍하는 과학

## 머신러닝 시스템의 종류

### 훈련 지도 방식

- **지도 학습**
    
    훈련 데이터에 레이블 값 포함
    
    분류 - ex.스팸 필터 → 샘플 이메일과 클래스로 훈련, 새 메일을 어떻게 분류할지 학습
    
    로지스틱 회귀 - 클래스에 속할 확률 출력
    
    회귀 - 특성을 이용해 타깃 수치를 예측 (특성과 타깃이 포함된 데이터가 많이 필요)
    
    타깃은 회귀에서, 레이블은 분류에서 많이 사용
    
    특성 = 예측변수, 속성이라 부름
    
- **비지도 학습**
    
    훈련 데이터에 레이블 X
    
    - 계층 군집
        
        데이터 사이의 연결고리 찾기 → 더 작은 그룹으로 세분화
        
    - 시각화
        
        도식화 가능한 2D나 3D 표현을 만들어준다
        
    - 차원 축소
        
        적은 정보를 잃으면서 데이터를 간소화
        
        EX.특성 추출 - 상관관계가 있는 여러 특성을 하나로 합치기
        
        데이터 주입 전 차원 축소를 하는것이 유용함 - 실행속도 up, 메모리 공간 down
        
    - 이상치 탐지
        
        학습 알고리즘에 주입하기 전에 데이터셋에서 이상한 값을 자동으로 제거
        
    - 특이치 탐지
        
        
    - 연관 규칙 학습
        
        특성 간 흥미로운 관계 찾기
        
- **준지도 학습**
    
    데이터에 레이블이 일부만 있음
    
- **자기 지도 학습**
    
    레이블이 전혀 없는 데이터셋에서 레이블이 완전히 부여된 데이터셋을 생성
    
- **강화 학습**
    
    학습 시스템 = 에이전트
    
    환경을 관찰해서 행동을 실행하고 그 결과로 보상 또는 벌점을 받는다.
    
    가장 큰 보상을 얻기 위해 정책이라고 부르는 최상의 전략을 스스로 학습
    

### 배치 학습과 온라인 학습

- **배치 학습**
    
    점진적으로 학습 X
    
    모델 부패, 데이터 드리프트 - 시간이 지남에 따라 모델의 성능이 천천히 감소
    
    해결: 최신 데이터에서 모델을 정기적으로 재훈련
    
- **온라인 학습**
    
    데이터를 순차적으로 한 개씩 또는 미니배치라 부르는 작은 묶음단위로 주입하여 훈련
    
    메 학습 단계가 빠르고 비용이 적게 든다
    
    새로운 데이터가 입력되면 즉시 학습한다
    
    **학습률 -** 변화하는 데이터에 얼마나 빠르게 적응할 것인지
    
    학습률 up - 시스템이 데이터에 빠르게 적응 , but 예전 데이터를 금방 잊어버릴 것
    
    학습률 down - 느리게 학습, but 새로운 데이터에 있는 잡음이나 대표성 없는 데이터 포인트에 덜 민감
    
    **단점**
    
    나쁜 데이터가 주입되었을 때 시스템 성능이 감소
    
    성능 감소 → 학습 중단, 이상치 탐지 알고리즘 사용 등
    

### 사례 기반 학습과 모델 기반 학습

어떻게 일반화 되는가?

- **사례 기반 학습**
    
    시스템이 훈련 샘플을 기억함으로써 학습
    
    유사도 측정을 사용해 새로운 데이터와 학습한 샘플을 비교하여 일반화
    
- **모델 기반 학습**
    
    샘플들의 모델을 만들어 예측에 사용하는 것
    
    모델 → 함수 느낌(?)
    
    모델에는 모델 파라미터가 존재하며 최적의 파라미터를 찾아야한다. (비용함수를 최소화 시키는 파라미터 찾기)
    
    모델의 성능이 얼마나 좋은지 측정하는 효용함수(fitness function)와 나쁜지 측정하는 비용함수 (cost function)로 모델의 성능을 측정한다.
    
- **과정**
    1. 데이터 분석
    2. 데이터에 적합한 모델 선택
    3. 훈련 데이터로 모델을 훈련 → 비용 함수를 최소화하는 모델 파라미터를 찾는다
    4. 새로운 데이터에 모델을 적용해 예측을 만든다.
    
    더 좋은 예측을 위해서:
    
    더 많은 특성 사용
    
    좋은 훈련 데이터를 더 많이 모으기
    
    더 강력한 모델을 선택
    

### 데이터

- 충분하지 않은 양의 훈련 데이터
    
    데이터가 많이 있어야 잘 작동한다
    
    알고리즘보다 데이터가 더 중요하다(물론 알고리즘 무시하지 말자)
    
- 대표성 없는 훈련 데이터
    
    샘플 small : sampling noise 발생(우연에 의한 대표성 없는 데이터)
    
    샘플 big: 샘플링 편향(sampling bias)가 발생할 수 있음(대표성을 띠지 못할 수 있다)
    
    →정확한 예측을 하지 못할 수 있다
    
- 낮은 품질의 데이터
    
    훈련 데이터 정제 중요
    
    이상치라는 것이 명확 → 샘플 무시 or 고치기
    
    특성 몇 개 누락 → 특성을 모두 무시? 샘플 무시? 빠진 값을 채울지? 넣은 모델 or 뺀 모델을 쓸지 결정
    
- 관련없는 특성
    
    훈련에 사용할 좋은 특성들을 찾아야한다
    
- overfitting 과대적합,  underfitting 과소적합
    - **Overfitting**
        
        훈련 데이터에는 너무 잘 맞지만 일반성이 떨어진다
        
        훈련 데이터의 양과 잡음에 비해 모델이 너무 복잡할 때 일어남.
        
        **해결방안**
        
        1. 파라미터 수가 적은 모델을 선택 or regularization을 통해 모델에 제약을 가한다. 
            
            즉, 모델을 단순화 시킨다.
            
            규제의 양은 **하이퍼파라미터**가 결정
            
        2. 훈련 데이터를 더 많이 모으기
        3. 훈련 데이터의 잡음을 줄이기
        
        하이퍼파라미터란?
        
        모델x, 학습 알고리즘의 파라미터. 
        
        훈련 전에 미리 지정, 훈련하는 동안에는 상수로 남아 있음.
        
    - Underfitting
        
        모델이 너무 단순해서 데이터의 구조를 학습하지 못할 때
        
        **해결방안**
        
        1. 모델 파라미터가 많은 강력한 모델을 선택
        2. 학습 알고리즘에 더 좋은 특성을 제공
        3. 모델의 제약을 줄인다.

### 테스트와 검증

훈련 데이터를 훈련 세트와 테스트 세트 2개로 나누어서 진행.

테스트 세트를 통해 일반화 오차(generalization error)추정값을 얻음.

훈련 오차가 작지만 일반화 오차가 크다면 이 모델은 과적합된것임

- 하이퍼파라미터 튜닝과 모델 선택
    
    모델과 하이퍼파라미터가 테스트 세트에 최적화되어 과적합됨.
    
    **홀드아웃 검증**(holdout validation)
    
    훈련 세트 일부를 추출(**검증 세트**)하여 여러 후보 모델을 평가하고 가장 좋은 하나를 선택
    
    (전체 훈련 세트 - 검증 세트) 에서 다양한 하이퍼파라미터 값을 가진 여러 모델을 훈련
    
    검증 세트에서 가장 높은 성능을 내는 모델을 선택
    
    최선의 모델을 전체 훈련세트에서 다시 훈련하여 최종 모델을 만든뒤 테스트세트에서 평가하여 일반화 오차를 추정
    
    **교차 검증**
    
    검증 세트가 작거나 커서 문제가 발생할 수 있음
    
    해결 → 검증 세트 여러개를 사용 반복적으로 성과 측정, 결과를 평균
    
- 데이터 불일치
    
    데이터가 실제 사용될 데이터를 완벽하게 대표하지 못할 수 있음
    
    모델 성능이 떨어짐 → 과대적합 때문? or 데이터 불일치 때문?
    
    어떻게 구분해내는가?
    
    1. 훈련 데이터의 일부를 떼어내 **훈련-개발 세트**를 먼저 만든다.
    2. 모델을 훈련세트에서 훈련한 다음 훈련-개발 세트에서 평가
    3. 
        1. 모델이 잘 작동되지 않는다 → 훈련세트에 과대적합
        2. 모델이 잘 작동 → 검증 세트에서 평가 → 성능이 나쁘다면 데이터 불일치
    

# 3장 분류

### MNIST

숫자 이미지 데이터셋 (70000개) 

sklearn을 통해 데이터셋을 내려받을 수 있다.

```python
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', as_frame=False) 
// False로 지정하여 넘파잉로 데이터 받음
```

fetch_ : 데이터셋 다운로드

load_ : 번들로 포함된 소규모 데이터셋을 로드

make_ : 테스트에 유용한 가짜 데이터셋을 생성

생성된 데이터셋은 일반적으로 numpy 배열이고 입력과 타깃 데이터를 담은 (X, y) 튜플로 반환

일반적인 속성으로 다음과 같다

DESCR: 데이터셋 설명

data : 입력 데이터, 일반적으로 2D 넘파이 배열

target:  레이블, 일반적으로 1D 넘파이 배열

70000개 이미지, 각 이미지에는 784개 특성 (28*28픽셀)

**하나의 이미지 확인**

```python
import matplotlib as mpl
import matpotlib.pyplot as plt

def plot_digit(image_data):
	img = image_data.reshape(28,28) //특성 벡터를 추출해서 28*28배열로 크기 바꾸기
	
	plt.imshow(image, cmap = "binary")
	// imshow()함수를 이용해서 그린다.
	// cmap="binary"로 지정해 0을 흰색, 255를 검은색으로 나타내는 흑백 컬러맵 사용

	plt.axis("off")
```

**테스트 세트 분리**

```python
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[:60000]
```

### 이진 분류기 훈련

ex. 이미지가 5이냐 아니냐 2개의 클래스를 구분

target vector 만들기

```python
y_train_5 = (y_train == '5') // 5는 True고, 다른 숫자는 모두 False
y_test_5 = (y_test == '5')
```

**stochastic gradient descent**분류기로 시작

→ 훈련 샘플을 독립적으로 처리, 속도가 빨라서 매우 큰 데이터셋을 다루는데 효과적

### 성능 측정

- **교차 검증을 사용한 정확도 측정**

cross_val_score()함수로 sgdclassifier 평가 

→ 95% 나옴

모든 이미지를 가장 많이 등장하는 클래스로 분류하는 더미 분류기 만들어 비교

→정확도 90% 나옴

불균형한 데이터셋을 다룰 때 정확도를 분류기의 성능 측정 지표로 선호하지 않는다.

오차 행렬을 조사하는것이 더 좋다.

- **오차 행렬**

클래스 A의 샘플이 클래스 B로 분류된 횟수를 세는 것

(ex. 숫자 8의 이미지를 0으로 잘못 분류한 횟수는 오차 행렬에서 8번행 0번 열을 보면 된다)

실제 타깃과 비교할 수 있도록 예측값을 만들어야함

test set로도 가능하지만 여기서 사용하면 안됨 → cross_val_predict() 함수를 사용

cross_val_predict()

k-폴드 교차 검증을 수행하지만 평가점수 반환이 아닌 각 테스트 폴드에서 얻은 예측을 반환

```python
from sklearn.model_selection import  cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv = 3)
```

confusion_matrix()를 사용해 호출

```python
from sklearn.metrics import confusion_matrix

confusion_matrix(y_train_5, y_train_pred) #분류결과표

###결과값###
>>
array([[53892,   687],
       [ 1891,  3530]])

// 완벽한 분류기이 경우
>> 
array([[54579,     0],  
       [    0,  5421]])

```

행은 **실제 클래스**, 열은 **예측한 클래스** 

**True Positive** - 실제 이미지 5, 예측도 5

**True Negative -** 실제 이미지 5가 아님, 예측도 5가 아님

**False Positive -** 실제 이미지 5, 예측은 5가 아님

**False Negative -** 실제 이미지 5가 아님, 예측값은 5

**정밀도**

5라고 예측한 데이터 중 5인 데이터는 얼마나 있는지

![Untitled](1%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20Summary%20026a57418bf54230a3e91185e4ade714/Untitled.png)

**재현율**

5안 데이터들 중 예측값도 5라고 얼마나 잘 예측하였는지

![Untitled](1%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20Summary%20026a57418bf54230a3e91185e4ade714/Untitled%201.png)

- **정밀도와 재현율**
    
    ```python
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    precision_score(y_train_5, y_train_pred)
    >>
    0.8370879772350012
    
    recall_score(y_train_5,y_train_pred)
    >>
    0.6511713705958311
    
    f1_score(y_train_5, y_train_pred)
    >>
    0.7325171197343846
    ```
    
    **F1 score** : 정밀도와 재현율을 하나의 숫자(조화평균)로 표현한 지표
    
    ![Untitled](1%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20Summary%20026a57418bf54230a3e91185e4ade714/Untitled%202.png)
    
    정밀도와 재현율이 비슷하다면 F1 score가 높다.
    
    상황에 따라 더 중요한 지표가 있을 수 있음
    
- ****Precision/Recall trade-off****
    
    sgd classifier는 결정 함수를 사용하여 각 샘플의 점수를 계산함. 점수가 임곗값(threshold)보다 크면 양성 클래스를 할당, 그렇지 않다면 음성 클래스를 할당
    
     
    
    ![Untitled](1%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20Summary%20026a57418bf54230a3e91185e4ade714/Untitled%203.png)
    
    threshold를 높이면 정밀도가 100%이지만 재현율은 50%가 된다.
    
    threshold를 낮추면 정밀도가 낮아지지만 재현율은 높아진다.
    
    sklearn에서 예측에 사용한 점수는 확인할 수 있다.
    
    **decision_function()**를 호출하여 각 샘플의 점수를 얻는다.
    
    ```python
    y_scores = sgd_clf.decision_function([some_digit])
    y_scores
    
    >>
    array([2164.22030239])
    
    #threshold가 0일때
    
    threshold = 0
    y_some_digit_pred = (y_scores > threshold)
    y_some_digit_pred
    
    >>
    array([True])
    
    #threshold 3000일때
    
    threshold = 3000
    y_some_digit_pred = (y_scores > threshold)
    y_some_digit_pred
    
    >>
    array([False])
    ```
    
    적절한 임곗값(threshold)은 어떻게 정할까?
    
    1. cross_val_predict() 함수를 사용하여 훈련 세트에 있는 모든 샘플의 점수를 구한다
    2. precision_recall_curve() 를 이용하여 가능한 모든 임곗값에 대한 정밀도와 재현율을 계산한다
    3. matplotlib을 이용해 함수로 그린다
    
    **정밀도와 재현율 곡선**
    
    ![Untitled](1%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20Summary%20026a57418bf54230a3e91185e4ade714/Untitled%204.png)
    
    **재현율에 대한 정밀도 곡선**
    
    ![Untitled](1%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20Summary%20026a57418bf54230a3e91185e4ade714/Untitled%205.png)
    
    재현율이 80% 근처에서 정밀도가 급격하게 감소
    
    → 이 하강 지점 직전을 ****Precision/Recall trade-off****로 선택하는 것이 좋음
    
    정밀도 90%가 목표라면?
    
    argmax()를 사용
    
    → 최댓값의 첫번째 인덱스를 반환
    
    정밀도가 최소 90%가 되는 최소 임곗값을 찾는것
    
    ```python
    idx_for_90_precision = (precisions >= 0.90).argmax()
    threshold_for_90_precision = thresholds[idx_for_90_precision]
    threshold_for_90_precision
    
    >>
    3370.0194991439557
    
    y_train_pred_90 = (y_scores >= threshold_90_precision)
    # 훈련세트에 대한 예측 만들기
    ```
    
- ROC 곡선
    
    **False Positive 비율에 대한 True Positive 비율의 곡선**
    
    FPR은 1 - TNR이다
    
    어떻게 그리는가?
    
    1. roc_curve()를 이용하여 여라 임곗값에서 TPR과 FPR을 계산한다
    2. matplotlib 을 이용하여 TPR에 대한 FPR곡선을 나타낸다
    3. 
    
    ![Untitled](1%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20Summary%20026a57418bf54230a3e91185e4ade714/Untitled%206.png)
    
    TPR이 높을수록 분류기가 만드는 FPR이 늘어난다.
    
    점선은 완전한 랜덤 분류기의 ROC곡선
    
    즉, 좋은 분류기일수록 점선에서 떨어져 있어야한다
    
    **곡선 아래의 면적(AUC)**을 측정해 분류기들을 비교한다
    
    ```python
    from sklearn.metrics import roc_auc_score
    
    roc_auc_score(y_train_5, y_scores)
    >>
    0.9604938554008616
    ```
    
    완벽한 분류기의 AUC는 1, 완전한 랜덤 분류기는 0.5
    
    RandomForestClassifier를 훈련시켜 SGDClassifier의 PR곡선과 F1 점수를 비교
    
    RandomForestClassifier에는 작동 방식의 차이로 decision_function() 대신 **predict_proba()** 함수 사용 
    
    → 데이터가 행, 클래스가 열이고 데이터가 주어진 클래스에 속할 확률을 담은 배열을 반환
    
    ```python
    from sklearn.ensemble import RandomForestClassifier
    
    forest_clf = RandomForestClassifier(random_state = 42)
    y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method = "predict_proba")
    # 교차검증으로 RandomForestClassifier를 훈련하고 모든 이미지에 대한 클래스 확률 예측
    
    y_probas_forest[:2]
    
    >> 결과
    array([[0.11, 0.89],
           [0.99, 0.01]])
    ```
    
    ```python
    y_scores_forest= y_probas_forest[:, 1]
    precisions_forest, recalls_forest, thresholds_forest= precision_recall_curve(
        y_train_5, y_scores_forest)
    #precision_recall_curve()함수에 양수 클래스 확률을 전달
    ```
    
    ![Untitled](1%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20Summary%20026a57418bf54230a3e91185e4ade714/Untitled%207.png)
    
    오른쪽 모서리에 곡선이 가깝다 → SGD 보다 Random Forest가 훨씬 더 좋아 보인다
    

### 다중 분류

여러개의 이진 분류기를 이용하여 둘 이상의 클래스를 구별

1. **OvR(one vs rest) 혹은 OvA(one vs all) 전략**
    
    이미지 분류 시 Decision score가 가장 높은 것을 클래스로 선택한다. 
    
2. **OvO 전략**
    
    0과 1 구별, 0과 2 구별, 1과 2 구별 등 모든 숫자의 조합에 대해 이진 분류기를 학습시키는 것
    
    - OvO의 장점
        1. 각 Classifier를 훈련시킬 때 전체 데이터가 아닌 구별을 위한 두가지 클래스에 대한 데이터만 들어감
        2. training set의 크기에 민감하여 작은 training set을 선호하는SVM에서 효과적

대부분의 2진 분류에서는 OvR선호

sklearn으로 Multiclass Classification 구현 시 Binary Classifier를 선택하면 알고리즘에 따라 자동으로 OvR 또는 OvO를 실행

```python
from sklearn.svm import SVC

svm_clf = SVC(random_state=42)
svm_clf.fit(X_train[:2000], y_train[:2000])  # y_train_5가 아니고 y_train을 사용

svm_clf.predict([some_digit])

###결과값###
>>
array([5], dtype=object)
```

sklearn이 실제로 OvO 방법을 사용하는지 확인하기 위해 **decision_function()**을 사용하면 1개가 아니라, 데이터 당 10개의 점수를 반환

```python
some_digit_scores= svm_clf.decision_function([some_digit])
some_digit_scores.round(2)

#결과
>>
array([[ 3.79,  0.73,  6.06,  8.3 , -0.29,  9.3 ,  1.75,  2.77,  7.21,
         4.82]])
```

sklearn에서 OvO나 OvR을 강제하고 싶다면 'OneVsRestClassifier'나 'OneVsOneClassifier'를 사용

```python
from sklearn.multiclass import OneVsRestClassifier

ovr_clf = OneVsRestClassifier(SVC(random_state=42))
ovr_clf.fit(X_train[:2000], y_train[:2000])

ovr_clf.predict([some_digit])

#결과
>>
array(['5'], dtype='<U1')

len(ovr_clf.estimators_)
>>
10

```

SGDClassifier을 훈련하고 예측을 만드는것도 마찬가지

```python
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train)
sgd_clf.predict([some_digit])

#결과
>>
array(['3'], dtype='<U1')

sgd_clf.decision_function([some_digit]).round()

#결과
>>
array([[-31893., -34420.,  -9531.,   1824., -22320.,  -1386., -26189.,
        -16148.,  -4604., -12051.]])
```

대부분의 점수가 큰 음수라는 것은 Classifier가 예측 결과에 대한 강한 확신을 보인다

입력의 스케일을 조정하여 accuracy를 높여본다

### 오류 분석

가능성 높은 모델 하나를 찾았다고 가정하고 성능을 향상시킬 방법?

→  만들어진 에러의 종류를 분석하는 것

- 오차 행렬 살펴보기

### 다중 레이블 분류(Multi-Label Classification)

분류기에 따라 하나의 데이터가 여러 개의 클래스를 출력해야 할 때도 있다

여러 개의 이진 클래스를 출력하는 분류 시스템을 'Multi-Label Classification'이라고 함.

```python
from sklearn.neighbors import KNeighborsClassifier

y_train_large = (y_train >= 7)
y_train_odd = (y_train.astype('int8') % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)

knn_clf.predict([some_digit])

###결과값###
>>
array([[False,  True]])
# 2개의 레이블이 출력됨
```

각 숫자 이미지에 두 개의 타깃 레이블이 담긴 y_multilabel 배열을 만든다

첫 번째 레이블은 숫자가 큰 값(7, 8, 9) 중 하나인지 나타내고, 두 번째는 홀수인지 나타냄.

 그리고 그 다음줄에서 KNeighborsClassifier에 이 다중 타깃 배열을 사용하여 훈련

### 다중 출력 분류(****Multi-output Classification)****

 Multi-Label Classification에서 하나의 레이블이 다중 클래스가 될 수 있도록 일반화한 것

**ex.** 이미지에서 잡음을 제거하기 

잡음이 많은 이미지를 입력으로 받고 깨끗한 숫자 이미지를 MNIST 이미지처럼 픽셀의 강도를 담은 배열로 출력 

특히 하나의 픽셀이 하나의 레이블이기 때문에 분류기의 출력은 다중 레이블이고, 각 레이블은 0~255사이의 값을 가집니다.

# 4장 모델 훈련

## 선형 회귀

**일반적인 linear model**

y=θ0+θ1∗x1+θ2∗x2+θ3∗x3+...+θn∗xn

가중치 합과 bias라는 상수를 더해 예측을 만든다

![Untitled](1%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20Summary%20026a57418bf54230a3e91185e4ade714/Untitled%208.png)

<벡터를 이용한 표현>

θ는 모델의 파라미터 벡터

x는 샘플의 특성 벡터

![Untitled](1%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20Summary%20026a57418bf54230a3e91185e4ade714/Untitled%209.png)

선형 모델에서 사용하는 loss function은 MSE이다.

### 정규 방정식

모델 파라미터를 최적화하는 하나의 방식

![Untitled](1%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20Summary%20026a57418bf54230a3e91185e4ade714/Untitled%2010.png)

```python
import numpy as np

np.random.seed(42)# 코드 예제를 재현 가능하게 만들기 위해m= 100# 샘플 개수
X = 2 * np.random.rand(m, 1)# 열 벡터
y = 4 + 3* X+ np.random.randn(m, 1)# 열 벡터
```

![Untitled](1%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20Summary%20026a57418bf54230a3e91185e4ade714/Untitled%2011.png)

랜덤으로 linear한 데이터 생성하고 정규 방정식을 통해 **θ 계산**

```python
from sklearn.preprocessing import add_dummy_feature

X_b= add_dummy_feature(X) # 각 샘플에 x0 = 1을 추가합니다
theta_best= np.linalg.inv(X_b.T@ X_b)@ X_b.T@ y

theta_best

##결과값##
>>
array([[3.93328217],
       [3.08032243]])
```

**sklearn에서 선형 회귀Lienar Regression 수행**

```python
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)

lin_reg.intercept_, lin_reg.coef_ 
# .intercept_:추정된 상수항, .coef_ : 추정된 weight 벡터

>>
(array([4.27746389]), array([[2.91097464]]))

lin_reg.predict(X_new)
>>
array([[ 4.27746389],
       [10.09941316]])
```

특잇값 분해를 통해 유사역행렬이 계산된다.

## 경사 하강법(gradient descent)

 학습률 하이퍼파라미터로 결정 →  적절한 학습률 찾아야함

랜덤 초기화 때문에 **전역 최솟값**보다 덜 좋은 **지역 최솟값**에 수렴할 수 있다

![Untitled](1%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20Summary%20026a57418bf54230a3e91185e4ade714/Untitled%2012.png)

모델 훈련: 비용 함수를 최소화하는 모델 파라미터의 조합을 찾는 일

- ****Batch Gradient Descent****

![Untitled](1%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20Summary%20026a57418bf54230a3e91185e4ade714/Untitled%2013.png)

비용함수의 편도함수

![Untitled](1%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20Summary%20026a57418bf54230a3e91185e4ade714/Untitled%2014.png)

비용함수의 gradient vector

Batch gradient descent는 θ를 조정하기 위한 기울기를 구할 때 훈련 데이터 전체를 사용 → training set이 너무 크다면 시간이 오래 걸림

- 확률적 경사하강법 (stochastic gradient descent)
    
    하나의 데이터를 랜덤으로 선택하여 그 데이터에 대한 기울기를 계산 → 속도가 빠르다
    
    빠르기 때문에 불안정하다
    
    해결?
    
    1. 학습률을 점진적으로 감소시킨다(시작할 때 크게 하고 점차 작게 줄여간다)
    2. epoch마다 훈련 세트를 섞는다
    
- ****Mini-Batch Gradient descent****
    
    미니배치라고 불리는 작은 데이터 세트에 대한 기울기를 계산
    
    sgd보다 덜 불규칙하고 최솟값에 더 가까이 도달
    

![Untitled](1%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20Summary%20026a57418bf54230a3e91185e4ade714/Untitled%2015.png)

3가지 방식의 훈련 과정을 그래프로 표현

## 다항회귀

비선형 데이터를 학습하는 데 선형 모델을 사용할 수 있게 해줌

훈련 세트에 있는 각 특성르 제곱하여 새로운 특성으로 추가

→ 차수가 높아지면서 비선형 데이터에 선형 모델을 사용할 수 있게 해줌

특성 사이의 관계를 찾을 수 있음

## 학습 곡선

모델의 일반화 성능을 추정하기 위해 학습 곡선을 확인

학습곡선이란?

모델의 훈련 오차와 검증 오차를 훈련 반복 횟수의 함수로 나타낸 그래프

## 규제가 있는 선형 모델

### 릿지 회귀

규제가 추가된 선형 회귀

### 라쏘 회귀

### 엘라스틱넷

### 조기 종료

## 로지스틱 회귀

분류에 사용.

샘플이 특정 클래스에 속할 확률을 추정하는 데 널리 사용

- **확률 추정**
    
    로지스틱 회귀의 작동 방법>
    
    각 특성의 가중치 합을 계산. 결괏값의 로지스틱을 출력
    
    ![Untitled](1%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20Summary%20026a57418bf54230a3e91185e4ade714/Untitled%2016.png)
    
    시그모이드 함수로 감싸서 0과 1사이 값(확률)이 나오도록 해줌
    
    확률이 0.5이상이면 1, 이하면 0이라고 예측
    
- **훈련과 비용 함수**
    
    로지스틱 회귀 모델에서 훈련의 목적
    
    1. 양성 샘플에 대해서는 높은 확률을 추정
    2. 음성 샘플에 대해서는 낮은 확률을 추정
    
    하는 모델의 파라미터 벡터θ를 찾는것
    
    ![Untitled](1%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20Summary%20026a57418bf54230a3e91185e4ade714/Untitled%2017.png)
    
    하나의 훈련 샘플 비용함수
    
    ![Untitled](1%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20Summary%20026a57418bf54230a3e91185e4ade714/Untitled%2018.png)
    
    전체 데이터에 대한 비용
    
    볼록 함수이므로 전역 최솟값 찾는것을 보장
    
- **결정 경계**
    
    ex. 세 가지의 서로 다른 꽃을 로지스틱 회귀로 분류
    
    ![Untitled](1%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20Summary%20026a57418bf54230a3e91185e4ade714/Untitled%2019.png)
    
    1) Iris-Verginica의 꽃잎 너비는 1.4cm~2.5cm 사이입니다.
    
    2) 꽃잎의 너비가 2cm가 이상이면 Iris-Verginica라고 강하게 확신합니다.
    
    3) 양쪽의 확률이 똑같이 **50%**가 되는 1.6cm 근방에서 **결정 경계**가 만들어짐. 
    
- **소프트맥스 회귀**
    
    2가지 이상의 클래스를 분류하는데 사용하는 로지스틱 회귀
    
    1. 데이터가 주어지면 소프트맥스 회귀 모델이 각 클래스에 대한 점수를 계산
        
        ![Untitled](1%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20Summary%20026a57418bf54230a3e91185e4ade714/Untitled%2020.png)
        
    2. **softmax 함수**를 이용하여 점수를 확률 값으로 변경. 가장 높은 확률값을 가진 클래스로 데이터 x를 분류
    3. 훈련 시 **크로스 엔트로피 비용함수**를 사용
        
        타깃 클래스에 대해서는 높은 확률을 그 외에는 낮은 확률을 추정하도록 만들기