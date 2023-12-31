# 1장 - 한눈에 보는 머신러닝

---

# 머신러닝 분류 체계

1. **지도 형태**에 따른 분류
- 지도 학습
    - 레이블이 제공됨
    - 다량의 데이터 필요
- 비지도 학습
    - 레이블이 제공되지 않음 (스스로 학습)
    - 관련 알고리즘 : 계층 군집, 시각화, 차원 축소, 이상치 탐지-특이치 탐지, 연관 규칙 학습
- 준지도 학습
    - 레이블이 일부만 있음
    - 지도 학습 + 비지도 학습
- 자기 지도 학습
    - 레이블이 제공되지 않음 → 직접 레이블을 매핑함
- 강화 학습
    - 보상과 벌점

1. **점진적으로** 학습할 수 있는지의 여부에 따른 분류
- 배치 학습
    - 점진적으로 학습할 수 없음 (=가용 데이터를 **한번에, 모두** 사용해서 훈련)
    - 리소스 큼
    - 모델 부패 이슈
- 온라인 학습
    - 점진적 (순차적)으로 학습 or 미니배치 단위로 학습
    - 리소스 적음
    - 데이터 품질에 크게 영향 받음

1. 어떻게 **일반화** 되는지에 따른 분류
- 사례 기반 학습
    - 유사도 측정
    - 훈련 샘플을 기억
- 모델 기반 학습
    - 모델을 만들어 예측
    - 대부분의 머신러닝

# 인공지능 프로젝트 망하는 법

- 나쁜 모델 사용
- 나쁜 (훈련) 데이터 사용
    - 양이 적음
    - 대표성이 없음
        - 샘플링 잡음, 샘플링 편향
    - 품질이 낮음
    - 관련없는 특성에 집중함
    - 훈련 데이터에 과대적합
        - 해결 : 파라미터 수 줄이기, 특성 수 줄이기, 제약 걸기, 데이터 수 늘리기
    - 훈련 데이터에 과소적합
        - 해결 : 파라미터 수 늘리기, 제약 줄이기

# 테스트, 검증

일반적으로, 훈련 데이터를 **훈련 세트**와 **테스트 세트**로 나눔

검증 세트를 따로 빼서 **홀드아웃 검증** 과정을 거치는 것도 좋은 방법

# 2장 - 머신러닝 프로젝트 처음부터 끝까지

공부가 필요

# 3장 - 분류

---

# 분류하려는 가짓수에 따른 분류

- 이진 분류 (A 인지, 아닌지)
    - 확률적 경사 하강법(SGD)
    - 성능 측정 : k-폴드 교차 검증
        - 성능 판단 : 정확도 X → **정밀도**(precision), **재현율**(recall), **F1 score** O →
            - 정밀도와 재현율은 **트레이드오프** 관계
    - ROC 곡선
- 다중 분류
    - 다중 분류 알고리즘
        - LogisticRegression, RandomForestClassifier, GaussianNB
    - 이진 분류 알고리즘을 여러개 사용해서 다중 분류
        - OvR, OvA, OvO 전략

# 분류기의 약점

- 이미지의 위치, 회전 방향에 매우 민감
    - 이미지 이동, 회전된 변형 이미지로 보강

# 다중 레이블 분류

이진 분류의 결과가 여러 개 출력되는 시스템

- 다중 출력 다중 클래스 분류

# 4장 -모델 훈련

---

# 선형 회귀

- 선형 데이터
- 구성 요소 (파라미터)
    - 가중치 합
    - 편향
- 파라미터 설정
    - 오차 줄이기
        - 평균 제곱 오차(MSE)
        - 평균 제곱근 오차(RMSE)
- 정규 방정식

# 경사 하강법

- MSE 비용 함수에서 전역 최솟값에 다가가기
- 학습률 설정
- 배치 경사 하강법
    - 매 경사 하강법 스텝에서 전체 훈련 데이터 사용
- 확률적 경사 하강법
    - 매 스탭에서 한개의 샘플 랜덤 선택해서 사용
- 미니배치 경사 하강법
    - 미니배치 (작은 샘플 세트)를 사용

# 다항 회귀

- 비선형 데이터
- 선형 모델 사용 (각 특성의 거듭제곱 → 새로운 특성으로 추가)

# 회귀 할 때의 문제점과 해결

- 학습 곡선 분석
    - 훈련 데이터에 과소적합
        - 학습 곡선에서 두 오차 곡선이 꽤 높은 오차에서 매우 가까이 근접해 있음
        - 복잡한 모델, 더 좋은 특성 사용
    - 훈련 데이터에 과대적합
        - 두 곡선 사이에 공간이 있음
        - 검증 오차가 훈련 오차에 근접할 때까지 많은 훈련 데이터 추가
        

# 규제가 있는 선형 모델

- 릿지 회귀
    - 가중치 작게 유지
    - 규제항
        - 훈련하는 동안에만 사용됨
- 라쏘 회귀
    - 희소 모델 만들기 (덜 중요한 특성의 가중치 제거)
- 엘라스틱넷 회귀
    - 릿지 회귀와 라쏘 회귀의 비율을 섞어서 사용 (r, 1-r)
- 조기 종료

# 로지스틱 회귀

공부가 필요