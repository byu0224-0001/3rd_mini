import streamlit as st
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras

# predict.py 파일에서 StockChartPredictor 클래스를 가져옵니다.
# (predict.py 파일이 같은 디렉토리에 있다고 가정합니다.)
# 예측 클래스 로직을 직접 복사하거나, predict.py에서 import 해야 합니다.
# 여기서는 predict.py의 로직을 직접 통합하는 방식으로 작성합니다.

# -----------------------------------------------------
# Predictor 클래스 로직 (predict.py에서 가져옴)
# -----------------------------------------------------
from PIL import Image
from predict import StockChartPredictor

# 글로벌 변수
MODEL_PATH = 'models/best_stock_chart_model.h5'
IMG_SIZE = (100, 100) # 모델 학습 시 사용된 크기

@st.cache_resource
def load_predictor():
    """모델을 로드하고 Streamlit 캐시에 저장합니다 (앱 시작 시 1회만 실행)."""
    try:
        # StockChartPredictor 클래스를 사용하여 모델 로드 (predict.py 파일 참조)
        predictor = StockChartPredictor(model_path=MODEL_PATH, img_size=IMG_SIZE)
        return predictor
    except FileNotFoundError:
        st.error(f"❌ 모델 파일({MODEL_PATH})을 찾을 수 없습니다. STEP 4를 먼저 완료해주세요.")
        return None
    except Exception as e:
        st.error(f"❌ 모델 로드 중 오류 발생: {e}")
        return None

# -----------------------------------------------------
# Streamlit UI/UX 함수
# -----------------------------------------------------

def display_prediction_result(result):
    """예측 결과를 시각적으로 표시합니다."""
    
    prediction = result['prediction']
    confidence = result['confidence']
    up_prob = result['up_probability']
    down_prob = result['down_probability']

    st.subheader("🎯 예측 결과")

    if prediction.startswith('Up'):
        st.balloons()
        st.success(f"📈 다음날 주가 **{prediction}** 예측!")
        st.markdown(f"<h1 style='color: green; text-align: center;'>{confidence:.2%}</h1>", unsafe_allow_html=True)
        color_code = '#4CAF50' # Green
    else:
        st.error(f"📉 다음날 주가 **{prediction}** 예측!")
        st.markdown(f"<h1 style='color: red; text-align: center;'>{confidence:.2%}</h1>", unsafe_allow_html=True)
        color_code = '#F44336' # Red

    st.markdown("---")
    
    # 확률 게이지 표시
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="상승 확률 (Up Probability)", value=f"{up_prob:.2%}")
        st.progress(up_prob)

    with col2:
        st.metric(label="하락 확률 (Down Probability)", value=f"{down_prob:.2%}")
        st.progress(down_prob)


def main_app():
    """메인 Streamlit 애플리케이션"""
    st.set_page_config(
        page_title="주식 차트 패턴 예측 시스템",
        layout="centered"
    )

    st.title("📈 주식 차트 패턴 예측 시스템")
    st.markdown("---")
    
    # 1. 모델 로드
    predictor = load_predictor()
    
    if predictor is None:
        return

    # 2. 파일 업로드 위젯
    st.subheader("1. 차트 이미지 업로드 (.jpg, .png)")
    uploaded_file = st.file_uploader(
        "분석할 캔들스틱 차트 이미지를 선택하세요.", 
        type=['jpg', 'jpeg', 'png']
    )

    if uploaded_file is not None:
        # 3. 이미지 표시
        image = Image.open(uploaded_file)
        st.image(image, caption='업로드된 차트 이미지', use_column_width='auto', width=200)

        # 4. 예측 버튼
        if st.button("분석 시작 (예측)", type="primary"):
            
            with st.spinner('AI가 차트 패턴을 분석 중입니다...'):
                # 5. 예측 수행 (Streamlit은 FileUploader의 BytesIO 객체를 처리해야 함)
                # Predictor.predict 메서드는 파일 경로를 받으므로, 임시 파일로 저장 후 사용
                
                # 임시 파일 경로 설정
                temp_file_path = os.path.join("temp_chart.jpg")
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # 예측 실행
                try:
                    # StockChartPredictor.predict는 verbose=True를 기본으로 가지므로
                    # 여기서는 predict_image를 사용하거나 predict 메서드를 수정해야 합니다.
                    # 임시 파일 경로를 사용하여 예측 클래스의 메서드 호출
                    result = predictor.predict(temp_file_path, verbose=False)
                    
                    # 6. 결과 표시
                    display_prediction_result(result)

                except Exception as e:
                    st.error(f"예측 중 심각한 오류가 발생했습니다: {e}")
                finally:
                    # 임시 파일 삭제
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)

    st.markdown("---")
    st.caption(f"시스템 정보: ResNet 기반 CNN 모델 사용 | 입력 크기: {IMG_SIZE}")


if __name__ == '__main__':
    # Streamlit 앱 실행 시 GPU/CPU 설정은 이미 메인 환경에서 처리됨
    main_app()