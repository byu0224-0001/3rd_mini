# 3rd_mini
3차 미니 프로젝트
---
# 🎯 주식 차트 패턴 CNN 프로젝트 - 완전 가이드

## 📋 프로젝트 개요

이 프로젝트는 **100만 개 이상의 주식 차트 이미지**를 학습하여 다음날 주가의 **상승/하락을 예측**하는 딥러닝 시스템입니다.

### 핵심 목표
1. ✅ CNN을 활용한 차트 패턴 인식
2. ✅ 이진 분류 (Up/Down) 예측 모델 구축
3. ✅ 실시간 예측 시스템 구현
4. 🔄 (향후) 14종 차트 패턴 분류로 확장

---

## 📊 데이터셋 정보

### 현재 데이터
- **위치**: `dataset-2021/`
- **구조**:
  - `up/`: 다음날 상승 차트 (499,498개)
  - `down/`: 다음날 하락 차트 (516,231개)
- **총 이미지**: 1,015,729개
- **이미지 크기**: 224x224 픽셀 (RGB)
- **클래스 비율**: 거의 균형 (49.18% vs 50.82%)
- **종목**: 중국 주식 시장 (XSHE, XSHG)
- **기간**: 2021년 데이터

---

## 🏗️ 프로젝트 구조

```
Study/project/
│
├── dataset-2021/              # 메인 데이터셋 (1M+ 이미지)
│   ├── up/                   # 상승 차트
│   └── down/                 # 하락 차트
│
├── models/                   # 저장된 모델
│   ├── best_stock_chart_model.h5      # 최고 성능 모델
│   └── test_stock_chart_model.h5      # 테스트 모델
│
├── results/                  # 결과 파일
│   ├── training_history.png           # 학습 곡선
│   ├── confusion_matrix.png           # 혼동 행렬
│   ├── sample_charts.png              # 샘플 차트
│   ├── data_distribution.png          # 데이터 분포
│   └── model_summary.txt              # 모델 구조
│
├── stock_chart_cnn.py        # 📌 메인 학습 스크립트
├── predict.py                # 📌 예측 스크립트
├── explore_data.py           # 📌 데이터 탐색 스크립트
├── train_quick_test.py       # 📌 빠른 테스트 스크립트
│
├── requirements.txt          # 필요 패키지
├── README.md                # 프로젝트 설명
└── PROJECT_GUIDE.md         # 📌 이 파일
```

---

## 🚀 단계별 실행 가이드

### STEP 1: 환경 설정

```bash
# 가상환경 생성 (선택사항)
python -m venv venv
venv\Scripts\activate  # Windows

# 필수 패키지 설치
pip install -r requirements.txt
```

### STEP 2: 데이터 탐색

```bash
python explore_data.py
```

**출력 결과**:
- 데이터셋 통계 (개수, 비율, 크기)
- 샘플 이미지 시각화
- 이미지 통계 분석
- 데이터 분포 차트

**생성 파일**:
- `results/sample_charts.png`
- `results/data_distribution.png`

### STEP 3: 빠른 테스트 (추천)

**목적**: 전체 데이터로 학습하기 전에 모델이 제대로 작동하는지 확인

```bash
python train_quick_test.py
```

**특징**:
- 각 클래스당 5,000개씩 사용 (총 10,000개)
- 20 에포크 학습 (약 30분 소요, GPU 기준)
- 빠른 프로토타입 검증

**생성 파일**:
- `models/test_stock_chart_model.h5`
- `results/training_history_test.png`
- `results/confusion_matrix_test.png`
- `results/model_summary_test.txt`

### STEP 4: 전체 모델 학습 (시간 소요 큼)

```bash
python stock_chart_cnn.py
```

**주의사항**:
- ⚠️ 100만 개 이미지 학습은 매우 오래 걸립니다
- 💾 메모리: 최소 16GB RAM 권장
- 🖥️ GPU: NVIDIA GPU (CUDA 지원) 강력 권장
- ⏱️ 예상 시간: 
  - GPU: 10-20시간
  - CPU: 100+ 시간

**생성 파일**:
- `models/best_stock_chart_model.h5` (최고 성능 모델)
- `results/training_history.png`
- `results/confusion_matrix.png`
- `results/model_summary.txt`

### STEP 5: 예측 수행

#### 5-1. 단일 이미지 예측

```bash
python predict.py --image dataset-2021/up/000001.XSHE-20201229-1.jpg --visualize
```

#### 5-2. 배치 예측

```bash
python predict.py --batch dataset-2021/up --model models/best_stock_chart_model.h5
```

#### 5-3. Python 코드로 예측

```python
from predict import StockChartPredictor

# 모델 로드
predictor = StockChartPredictor(model_path='models/best_stock_chart_model.h5')

# 예측
result = predictor.predict('my_chart.jpg')
print(f"예측: {result['prediction']}")
print(f"상승 확률: {result['up_probability']:.2%}")
print(f"하락 확률: {result['down_probability']:.2%}")
```

---

## 🔧 모델 아키텍처

### CNN 구조

```
Input: (100, 100, 3)
  ↓
[Conv Block 1]
Conv2D(32) → Conv2D(32) → BatchNorm → MaxPool(2x2) → Dropout(0.25)
  ↓
[Conv Block 2]
Conv2D(64) → Conv2D(64) → BatchNorm → MaxPool(2x2) → Dropout(0.25)
  ↓
[Conv Block 3]
Conv2D(128) → Conv2D(128) → BatchNorm → MaxPool(2x2) → Dropout(0.25)
  ↓
[Conv Block 4]
Conv2D(256) → Conv2D(256) → BatchNorm → MaxPool(2x2) → Dropout(0.25)
  ↓
[FC Layers]
Flatten → Dense(512) → BatchNorm → Dropout(0.5)
        → Dense(256) → BatchNorm → Dropout(0.5)
        → Dense(1, sigmoid)
  ↓
Output: [0, 1] (0=Down, 1=Up)
```

### 주요 기술

1. **Data Augmentation**
   - 회전 (±10도)
   - 이동 (±10%)
   - 수평 플립
   - 줌 (±10%)

2. **정규화 기법**
   - Batch Normalization
   - Dropout (0.25, 0.5)
   - L2 Regularization (선택)

3. **최적화**
   - Optimizer: Adam (lr=0.001)
   - Loss: Binary CrossEntropy
   - Metrics: Accuracy, Precision, Recall, AUC

4. **콜백**
   - EarlyStopping (patience=10)
   - ModelCheckpoint (최고 성능 저장)
   - ReduceLROnPlateau (학습률 감소)

---

## 📈 평가 지표

모델은 다음 지표로 평가됩니다:

### 1. Accuracy (정확도)
전체 예측 중 맞춘 비율

### 2. Precision (정밀도)
상승 예측 중 실제 상승 비율

### 3. Recall (재현율)
실제 상승 중 상승으로 예측한 비율

### 4. F1-Score
Precision과 Recall의 조화평균

### 5. AUC (ROC 곡선 아래 면적)
모델의 전반적인 분류 성능

### 6. Confusion Matrix
|              | 예측: Down | 예측: Up |
|--------------|-----------|----------|
| **실제: Down** | TN        | FP       |
| **실제: Up**   | FN        | TP       |

---

## 💡 최적화 및 개선 아이디어

### 현재 구현 ✅
- [x] CNN 기반 이미지 분류
- [x] Data Augmentation
- [x] 정규화 (BatchNorm, Dropout)
- [x] Early Stopping
- [x] 학습률 스케줄링
- [x] 예측 시스템

### 향후 개선 사항 🔄

#### 1. 라벨링 고도화
현재: 다음날 상승/하락 (노이즈 많음)
개선:
- N일 후 수익률 기반 (3일, 5일, 10일)
- 변동성 고려 (2% 이상 상승/하락만 라벨링)
- 거래량 패턴 결합

#### 2. 데이터 확장
현재: 2021년 중국 주식만
개선:
- 다양한 연도 데이터 추가 (2015-2024)
- 국내 주식 (KOSPI, KOSDAQ)
- 해외 주식 (S&P500, NASDAQ)
- 암호화폐 차트

#### 3. 모델 앙상블
- CNN + LSTM (시계열 패턴)
- ResNet, EfficientNet 등 고급 아키텍처
- 앙상블 (투표, 스태킹)

#### 4. 멀티모달 접근
차트 이미지 + 추가 데이터:
- 거래량 데이터
- 재무제표 (PER, PBR, ROE 등)
- 뉴스 감성 분석
- 매크로 경제 지표

#### 5. 패턴 분류 확장
현재: 상승/하락 (2분류)
확장: 14종 차트 패턴 분류
- Head & Shoulders
- Double Top/Bottom
- Triangle (Ascending, Descending, Symmetrical)
- Flag & Pennant
- Wedge
- Cup and Handle
- 등등...

#### 6. 실전 배포
- REST API 서버 (Flask/FastAPI)
- 웹 대시보드 (Streamlit)
- 모바일 앱
- 실시간 차트 크롤링 및 예측

---

## 🎓 학습 팁

### GPU 사용 확인

```python
import tensorflow as tf
print("GPU 사용 가능:", tf.config.list_physical_devices('GPU'))
```

### 메모리 부족 시
1. 배치 크기 줄이기: `batch_size=32` → `batch_size=16`
2. 이미지 크기 줄이기: `(100, 100)` → `(64, 64)`
3. 모델 크기 줄이기: 필터 개수 감소

### 학습 중단 후 재개

```python
# 체크포인트 로드
model = keras.models.load_model('models/checkpoint.h5')

# 학습 재개
model.fit(train_generator, initial_epoch=20, epochs=50, ...)
```

---

## ⚠️ 주의사항

### 1. 과적합 방지
- ✅ Dropout 사용
- ✅ Data Augmentation
- ✅ Early Stopping
- ✅ Validation 모니터링

### 2. 클래스 불균형
현재 데이터셋은 거의 균형잡혀 있음 (49% vs 51%)
불균형 시 해결책:
- Class Weights 사용
- 언더샘플링/오버샘플링
- SMOTE

### 3. 데이터 리크 방지
- Train/Val/Test 완전 분리
- 시계열 고려 (과거 → 미래)
- 같은 종목 데이터 분리

### 4. 투자 리스크
⚠️ **이 모델은 교육/연구 목적입니다**
- 실제 투자 결정에 단독 사용 금지
- 과거 데이터 기반 (미래 보장 없음)
- 시장 환경 변화에 취약
- 펀더멘털 분석 병행 필요

---

## 📞 문제 해결 (Troubleshooting)

### 1. ImportError: No module named 'tensorflow'
```bash
pip install tensorflow
```

### 2. CUDA Error (GPU 관련)
```bash
# CPU 버전 사용
pip install tensorflow-cpu
```

### 3. Out of Memory
- 배치 크기 줄이기
- 이미지 크기 줄이기
- 작은 서브셋으로 테스트

### 4. 낮은 정확도
- 더 많은 에포크 학습
- 학습률 조정
- 모델 구조 개선
- 데이터 전처리 개선

---

## 📚 참고 자료

### 논문
- ImageNet Classification with Deep CNNs (AlexNet)
- Very Deep CNNs for Large-Scale Image Recognition (VGG)
- Deep Residual Learning (ResNet)

### 기술 문서
- TensorFlow 공식 문서: https://www.tensorflow.org/
- Keras 가이드: https://keras.io/
- Scikit-learn: https://scikit-learn.org/

### 관련 프로젝트
- Chart Pattern Detection using Deep Learning
- Stock Price Prediction with LSTM
- Technical Analysis with Machine Learning

---

## 📝 라이선스 및 면책

이 프로젝트는 **교육 및 연구 목적**으로 제작되었습니다.

### 면책 조항
- 이 모델의 예측은 투자 조언이 아닙니다
- 실제 투자 손실에 대한 책임은 투자자 본인에게 있습니다
- 과거 성과가 미래를 보장하지 않습니다
- 충분한 리스크 관리와 분산 투자를 권장합니다

---

## 🎉 프로젝트 완료 체크리스트

- [x] ✅ 데이터셋 준비 (1M+ 이미지)
- [x] ✅ 데이터 탐색 및 분석
- [x] ✅ CNN 모델 설계
- [x] ✅ 학습 파이프라인 구축
- [x] ✅ 평가 시스템 구현
- [x] ✅ 예측 시스템 구현
- [x] ✅ 시각화 및 리포팅
- [ ] 🔄 전체 데이터 학습
- [ ] 🔄 하이퍼파라미터 튜닝
- [ ] 🔄 모델 앙상블
- [ ] 🔄 웹 서비스 배포
- [ ] 🔄 14종 패턴 분류 확장

---

## 📧 문의 및 기여

프로젝트 개선 아이디어나 버그 리포트는 언제든 환영합니다!

**Happy Coding & Trading! 🚀📈**
