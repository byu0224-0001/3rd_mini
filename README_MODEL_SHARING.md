# 주식 차트 패턴 CNN 모델 - 공유 가이드

## 📦 필요한 파일 목록

다른 사람과 모델을 공유하려면 다음 파일들을 함께 공유해야 합니다:

### 1️⃣ **필수 파일**

```
stock_chart_cnn_candlestick_aug.py  # 메인 학습 스크립트
requirements.txt                     # 필요한 패키지 목록
README_MODEL_SHARING.md             # 이 가이드 파일
```

### 2️⃣ **데이터셋 구조**

공유받는 사람은 다음과 같은 데이터셋 구조를 준비해야 합니다:

```
dataset-2021/  (또는 다른 이름)
├── up/        # 상승 패턴 이미지들
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── down/      # 하락 패턴 이미지들
    ├── image1.png
    ├── image2.png
    └── ...
```

**중요:** 이미지 파일 형식은 `.png`, `.jpg`, `.jpeg` 중 하나여야 합니다.

---

## 🚀 설치 및 실행 방법

### Step 1: Python 환경 설정

```bash
# Python 3.8 이상 필요
python --version

# 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows
```

### Step 2: 패키지 설치

```bash
pip install -r requirements.txt
```

### Step 3: 데이터셋 준비

1. 자신의 이미지 데이터를 다음 구조로 준비:
   ```
   my-dataset/
   ├── up/
   └── down/
   ```

2. 스크립트에서 데이터셋 경로 수정:
   ```python
   # stock_chart_cnn_candlestick_aug.py 파일의 main() 함수에서
   data_dir = create_subset_if_needed(
       source_dir='my-dataset',  # ← 여기를 자신의 데이터셋 경로로 변경
       target_dir='dataset-subset-5k',
       num_samples_per_class=2500
   )
   ```

### Step 4: 학습 실행

```bash
python stock_chart_cnn_candlestick_aug.py
```

---

## ⚙️ 설정 변경 방법

### 데이터셋 크기 조절

`main()` 함수에서 `num_samples_per_class` 값을 변경:

```python
# 5,000장 (각 클래스 2,500장)
num_samples_per_class=2500

# 10,000장으로 변경하려면
num_samples_per_class=5000
```

### 하이퍼파라미터 조정

`main()` 함수에서 다음 값들을 변경할 수 있습니다:

```python
# 배치 크기
batch_size = 128  # GPU 사용 시
batch_size = 64   # CPU 사용 시

# 이미지 크기
img_size=(128, 128)  # 기본값
img_size=(224, 224)  # 더 큰 이미지

# 에포크 수
epochs=50  # Stage 1, 2 각각

# Early Stopping patience
patience=15
```

### GPU 사용 설정

- **자동 감지**: 스크립트가 GPU를 자동으로 감지합니다
- **GPU 있으면**: 배치 크기 128로 자동 설정
- **GPU 없으면**: 배치 크기 64로 자동 설정, CPU 최적화 적용

---

## 📊 출력 파일

학습 완료 후 다음 파일들이 생성됩니다:

```
models/candlestick_aug/
├── stage1_best.keras  # 1단계 최고 모델
└── stage2_best.keras  # 2단계 최고 모델 (최종)

results/
└── candlestick_aug_results.png  # 학습 결과 시각화
```

---

## 🔧 문제 해결

### 1. GPU 메모리 부족 에러

```python
# batch_size를 줄이기
batch_size = 64  # 또는 32
```

### 2. 데이터셋을 찾을 수 없음

```
❌ 데이터셋을 찾을 수 없습니다: dataset-2021
```

→ `source_dir='dataset-2021'`을 자신의 데이터셋 경로로 변경

### 3. 메모리 부족 (RAM)

```python
# 서브셋 크기를 줄이기
num_samples_per_class=1000  # 총 2,000장
```

---

## 📋 데이터 증강 방법 (논문 기반)

### ✅ 적용된 증강

1. **캔들 무작위 이동**
   - 100% 캔들, 0.003 이동
   - 100% 캔들, 0.002 이동
   - 50% 캔들, 0.003 이동
   - 50% 캔들, 0.001 이동
   - 10% 캔들, 0.00025 이동

2. **가우시안 노이즈**
   - 평균: 0
   - 분산: 0.01

### ❌ 제거된 증강 (주가 차트에 부적합)

- 회전 (rotation)
- 좌우 반전 (horizontal flip)
- 상하 반전 (vertical flip)
- 크기 조절 (zoom)

---

## 💡 참고 사항

1. **학습 시간**: GPU 사용 시 약 40-50분, CPU 사용 시 약 1-1.5시간
2. **Early Stopping**: 성능이 개선되지 않으면 자동으로 조기 종료됩니다
3. **재현성**: `random.seed(42)`로 동일한 결과를 재현할 수 있습니다
4. **중간 저장**: 각 Stage의 최고 성능 모델이 자동 저장됩니다

---

## 📞 문의사항

모델 공유 시 문제가 발생하면:
1. Python 버전 확인 (3.8 이상 필요)
2. TensorFlow 버전 확인 (2.10.0 이상 필요)
3. 데이터셋 구조가 올바른지 확인
4. `requirements.txt`의 모든 패키지가 설치되었는지 확인

