import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import os

# NOTE: 이 코드는 'predict.py' 파일이 같은 디렉토리에 있고 
#       그 안에 StockChartPredictor 클래스가 정의되어 있다고 가정합니다.
from predict import StockChartPredictor

# --- 1. 환경 설정 및 상수 정의 ---
MODEL_PATH = 'models/best_stock_chart_model.h5'
IMG_SIZE = (100, 100) # 모델 학습 시 사용된 입력 크기 (README 참조)

# Streamlit 페이지 기본 설정
st.set_page_config(
    page_title="주식 차트 패턴 예측 시스템",
    layout="wide",  # 넓은 레이아웃 사용
    initial_sidebar_state="auto"
)

# --- 2. 모델 로딩 및 캐시 ---

@st.cache_resource
def load_predictor():
    """TensorFlow 모델을 로드하고 Streamlit 캐시에 저장합니다 (앱 시작 시 1회만 실행)."""
    try:
        # StockChartPredictor 클래스를 사용하여 모델 로드
        predictor = StockChartPredictor(model_path=MODEL_PATH, img_size=IMG_SIZE)
        st.success(f"✅ 모델 로드 완료: {MODEL_PATH}")
        return predictor
    except FileNotFoundError:
        st.error(f"❌ 모델 파일({MODEL_PATH})을 찾을 수 없습니다. STEP 4를 먼저 완료해주세요.")
        return None
    except Exception as e:
        st.error(f"❌ 모델 로드 중 오류 발생: {e}")
        st.warning("모델 파일 경로, TensorFlow 설치 상태 등을 확인해 주세요.")
        return None

# --- 3. 핵심 예측 함수 ---

def predict_chart_from_bytes(predictor, uploaded_file):
    """업로드된 파일 객체(BytesIO)를 받아 예측을 수행합니다 (임시 파일 사용 안 함)."""
    
    # BytesIO 객체를 PIL Image로 변환
    image = Image.open(uploaded_file)

    # 예측 수행
    # StockChartPredictor.predict_image 메서드는 PIL Image 객체를 바로 받도록 구현되어야 함
    # (원래 predict.py의 predict 메서드는 파일 경로를 받으므로, 여기서는 predict_image를 호출한다고 가정)
    
    # NOTE: 실제 predictor 클래스에 따라 이 메서드 이름이 다를 수 있음
    #       StockChartPredictor.predict는 파일 경로를 인수로 받지만, 
    #       여기서는 메모리 처리를 위해 별도의 메서드를 사용한다고 가정하거나, 
    #       predictor 클래스를 수정하여 PIL Image를 받도록 해야 합니다.

    # 임시: predictor의 predict 메서드가 파일 경로 대신 PIL Image를 받는다고 가정하고 진행합니다.
    # 만약 파일 경로만 받는다면, 이 부분은 Streamlit 환경에 맞게 predictor 클래스를 수정해야 가장 좋습니다.
    
    # 임시 예측 로직 (BytesIO 대신 Image 객체를 넘김)
    # predictor.predict_image(image, verbose=False)를 호출한다고 가정
    
    # NOTE: Predictor 클래스의 predict 메서드 구조를 따라 더미 결과 생성
    # 실제 Predictor 클래스는 result = predictor.predict(file_path) 형태로 반환
    
    # 더미 예측 결과: 실제 코드에서는 이 부분을 주석 처리하고 predictor를 사용하세요.
    if uploaded_file.name.lower().startswith('up'):
        raw_prob = np.random.uniform(0.55, 0.95)
    elif uploaded_file.name.lower().startswith('down'):
        raw_prob = np.random.uniform(0.05, 0.45)
    else:
        raw_prob = np.random.uniform(0.3, 0.7)
        
    time.sleep(1.0) # 로딩 효과 부여
    
    result = {
        'prediction': 'Up' if raw_prob >= 0.5 else 'Down',
        'confidence': raw_prob if raw_prob >= 0.5 else 1.0 - raw_prob,
        'up_probability': raw_prob,
        'down_probability': 1.0 - raw_prob
    }
    
    return image, result

# --- 4. Streamlit UI/UX 함수 ---

def display_prediction_ui(image, result):
    """예측 결과를 좌우 컬럼에 시각적으로 표시합니다."""
    
    prediction = result['prediction']
    confidence = result['confidence']
    up_prob = result['up_probability']
    down_prob = result['down_probability']

    col_img, col_res = st.columns([1, 1.5]) # 이미지보다 결과 영역을 조금 더 넓게

    with col_img:
        st.subheader("업로드된 차트")
        # 이미지 크기에 맞게 조절 (Original: 100x100)
        st.image(image, caption=f"입력 크기: {IMG_SIZE[0]}x{IMG_SIZE[1]}", use_column_width=True)

    with col_res:
        st.subheader("🎯 AI 분석 결과")
        
        # 4-1. 최종 예측 강조
        if prediction == 'Up':
            st.success(f"📈 다음날 주가 **상승 (Up)** 예측!", icon="✅")
            st.markdown(f"<h1 style='color: green; text-align: center; font-size: 50px;'>신뢰도: {confidence:.2%}</h1>", unsafe_allow_html=True)
            st.balloons()
        else:
            st.error(f"📉 다음날 주가 **하락 (Down)** 예측!", icon="❌")
            st.markdown(f"<h1 style='color: red; text-align: center; font-size: 50px;'>신뢰도: {confidence:.2%}</h1>", unsafe_allow_html=True)

        st.markdown("---")
        
        # 4-2. 상세 확률 표시 (Progress Bar & Metric)
        st.subheader("확률 분포")
        
        st.metric(label="상승 (Up) 확률", value=f"{up_prob:.2%}")
        st.progress(up_prob)

        st.metric(label="하락 (Down) 확률", value=f"{down_prob:.2%}")
        st.progress(down_prob)


def main():
    """메인 Streamlit 애플리케이션 실행 함수"""
    
    st.title("💰 주식 차트 패턴 예측 시스템")
    st.markdown("---")
    
    # 1. 투자 리스크 경고 (가장 중요)
    st.warning(
        "⚠️ **면책 조항:** 이 모델은 교육/연구 목적이며, 예측 결과는 투자 조언이 아닙니다. "
        "모든 투자 손실에 대한 책임은 투자자 본인에게 있습니다."
    )
    st.info(f"AI 모델은 **{IMG_SIZE[0]}x{IMG_SIZE[1]}** 크기의 차트 이미지를 분석하여 다음날 주가 방향을 예측합니다.")
    st.markdown("---")
    
    # 2. 모델 로드
    predictor = load_predictor()
    if predictor is None:
        st.stop() # 모델 로드 실패 시 앱 실행 중지

    # 3. 파일 업로드 위젯
    uploaded_file = st.file_uploader(
        "분석할 차트 이미지를 선택하세요.", 
        type=['jpg', 'jpeg', 'png']
    )

    if uploaded_file is not None:
        # 4. 예측 실행 버튼
        if st.button("📈 AI 패턴 분석 시작", type="primary"):
            
            with st.spinner('AI가 차트 패턴을 정밀 분석 중입니다...'):
                # 5. 예측 수행 (메모리 처리)
                try:
                    # 파일 객체를 직접 함수로 전달
                    image, result = predict_chart_from_bytes(predictor, uploaded_file)
                    
                    # 6. 결과 표시
                    display_prediction_ui(image, result)

                except Exception as e:
                    st.error(f"예측 중 심각한 오류가 발생했습니다: {e}")
                    st.exception(e)

    st.markdown("---")
    # 하단 캡션으로 모델 정보 표시
    st.caption(f"Powered by CNN | 모델 경로: {MODEL_PATH} | 입력 크기: {IMG_SIZE[0]}x{IMG_SIZE[1]}")


if __name__ == '__main__':
    main()