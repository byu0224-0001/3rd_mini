"""
주식 차트 패턴 예측 웹 애플리케이션
- ResNet50 기반 Transfer Learning 모델
- 주식 차트 이미지 업로드 및 상승/하락 예측
"""

import streamlit as st
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import io
import platform

# 한글 폰트 설정 (matplotlib)
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')  # 맑은 고딕
elif platform.system() == 'Darwin':  # macOS
    plt.rc('font', family='AppleGothic')
else:  # Linux
    plt.rc('font', family='NanumGothic')

plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 페이지 설정
st.set_page_config(
    page_title="주식 차트 패턴 예측기",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #0066cc;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #0052a3;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,102,204,0.4);
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .up-prediction {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .down-prediction {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #0066cc;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# 모델 로딩 함수 (캐싱)
@st.cache_resource
def load_model(model_path):
    """학습된 모델 로딩 및 입력 크기 자동 감지"""
    try:
        if os.path.exists(model_path):
            model = keras.models.load_model(model_path)
            
            # 모델의 입력 shape 확인
            input_shape = model.input_shape  # (None, height, width, channels)
            img_height = input_shape[1]
            img_width = input_shape[2]
            
            print(f"[INFO] 모델 로드 완료: {model_path}")
            print(f"[INFO] 모델 입력 크기: {img_height}x{img_width}")
            
            return model, (img_width, img_height)
        else:
            st.error(f"모델 파일을 찾을 수 없습니다: {model_path}")
            return None, None
    except Exception as e:
        st.error(f"모델 로딩 중 오류 발생: {e}")
        return None, None

# 이미지 전처리 함수
def preprocess_image(image, target_size=(128, 128)):
    """업로드된 이미지를 모델 입력 형식으로 전처리"""
    try:
        # PIL Image를 RGB로 먼저 변환 (ResNet50은 RGB 필요)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 크기 조정
        img = image.resize(target_size)
        img_array = np.array(img)
        
        # shape 확인 및 디버깅
        print(f"[DEBUG] 전처리 전 이미지 shape: {img_array.shape}")
        
        # RGB 확인 (안전장치)
        if len(img_array.shape) == 2:
            # Grayscale인 경우 RGB로 변환
            img_array = np.stack([img_array] * 3, axis=-1)
            print(f"[DEBUG] Grayscale -> RGB 변환: {img_array.shape}")
        elif img_array.shape[2] == 4:
            # RGBA인 경우 RGB로 변환
            img_array = img_array[:, :, :3]
            print(f"[DEBUG] RGBA -> RGB 변환: {img_array.shape}")
        
        # 정규화 (0-255 -> 0-1)
        img_array = img_array.astype('float32') / 255.0
        
        # 배치 차원 추가 [1, height, width, 3]
        img_array = np.expand_dims(img_array, axis=0)
        
        print(f"[DEBUG] 최종 입력 shape: {img_array.shape}")
        print(f"[DEBUG] 값 범위: min={img_array.min():.4f}, max={img_array.max():.4f}")
        
        # shape 검증
        expected_shape = (1, target_size[1], target_size[0], 3)
        assert img_array.shape == expected_shape, f"잘못된 shape: {img_array.shape}, 예상: {expected_shape}"
        
        return img_array
    
    except Exception as e:
        print(f"[ERROR] 이미지 전처리 중 오류: {e}")
        raise

# 예측 함수
def predict_stock_chart(model, image, target_size):
    """주식 차트 이미지 예측"""
    try:
        # 이미지 전처리 (모델에 맞는 크기로)
        processed_img = preprocess_image(image, target_size=target_size)
        
        print(f"[DEBUG] 모델 입력 shape: {processed_img.shape}")
        
        # 예측 수행
        prediction_proba = model.predict(processed_img, verbose=0)[0][0]
        
        print(f"[DEBUG] 예측 확률값: {prediction_proba}")
        
        # 이진 분류 결과
        prediction_class = 1 if prediction_proba > 0.5 else 0
        
        print(f"[DEBUG] 예측 클래스: {prediction_class} ({'UP' if prediction_class == 1 else 'DOWN'})")
        
        return prediction_class, prediction_proba
    
    except Exception as e:
        print(f"[ERROR] 예측 중 오류: {e}")
        import traceback
        traceback.print_exc()
        raise

# 신뢰도 게이지 차트 생성
def create_confidence_gauge(confidence, prediction_class):
    """
    신뢰도 게이지 차트 생성
    
    Args:
        confidence: 예측된 클래스에 대한 신뢰도 (0~1)
        prediction_class: 예측 클래스 (0: DOWN, 1: UP)
    """
    fig, ax = plt.subplots(figsize=(8, 2))
    
    # 배경
    ax.barh(0, 1, height=0.3, color='#e0e0e0', edgecolor='none')
    
    # 확률 바 (예측 결과에 따라 색상 결정)
    color = '#667eea' if prediction_class == 1 else '#f5576c'
    ax.barh(0, confidence, height=0.3, color=color, edgecolor='none')
    
    # 중앙선 (50% 신뢰도)
    ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2, alpha=0.5)
    
    # 레이블
    ax.text(0, -0.5, '낮음 (0%)', ha='left', va='top', fontsize=10, fontweight='bold')
    ax.text(1, -0.5, '높음 (100%)', ha='right', va='top', fontsize=10, fontweight='bold')
    ax.text(0.5, -0.5, '50%', ha='center', va='top', fontsize=9, alpha=0.7)
    
    # 현재 값 표시
    ax.text(confidence, 0.5, f'{confidence*100:.1f}%', 
            ha='center', va='center', fontsize=12, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=color, edgecolor='none'))
    
    # 예측 클래스 표시
    class_text = "상승(UP)" if prediction_class == 1 else "하락(DOWN)"
    ax.text(0.5, 1.2, f'{class_text} 예측 신뢰도', 
            ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 1.5)
    ax.axis('off')
    
    plt.tight_layout()
    return fig

# 메인 애플리케이션
def main():
    # 헤더
    st.markdown("<h1 style='text-align: center; color: #0066cc;'>📈 주식 차트 패턴 예측기</h1>", 
                unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.1rem; color: #666;'>ResNet50 기반 딥러닝 모델로 주식 차트의 상승/하락 패턴을 예측합니다</p>", 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # 사이드바
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # 모델 선택
        model_options = {
            "앙상블 모델 (Fine-tuned)": "improved_model_final.h5",
            "ResNet 모델 (Transfer Learning)": "stage2_best.keras"
        }
        
        selected_model_name = st.selectbox(
            "모델 선택",
            options=list(model_options.keys()),
            index=0
        )
        
        model_path = model_options[selected_model_name]
        
        st.markdown("---")
        
        # 정보
        st.markdown("### 📊 모델 정보")
        st.markdown(f"""
        - **아키텍처**: ResNet50
        - **학습 방식**: Transfer Learning + Fine-tuning
        - **입력 크기**: 자동 감지
        - **클래스**: Up (상승) / Down (하락)
        """)
        
        st.markdown("---")
        
        # 사용 방법
        with st.expander("📖 사용 방법"):
            st.markdown("""
            1. 주식 차트 이미지를 업로드하세요
            2. 모델이 자동으로 패턴을 분석합니다
            3. 예측 결과 및 신뢰도를 확인하세요
            
            **지원 형식**: PNG, JPG, JPEG
            """)
        
        # 샘플 데이터 정보
        with st.expander("📁 샘플 데이터"):
            st.markdown("""
            `charts/up/` - 상승 패턴 샘플
            `charts/down/` - 하락 패턴 샘플
            
            또는 직접 차트 이미지를 업로드하세요.
            """)
    
    # 모델 로딩 (입력 크기 자동 감지)
    model, img_size = load_model(model_path)
    
    if model is None:
        st.error("⚠️ 모델을 로드할 수 없습니다. 먼저 모델을 학습시켜주세요.")
        st.info("💡 `stock_chart_ensenble.py`를 실행하여 모델을 학습시킬 수 있습니다.")
        st.code("python stock_chart_ensenble.py", language="bash")
        return
    
    st.success(f"✅ 모델 로드 완료: {selected_model_name}")
    st.info(f"📐 모델 입력 크기: {img_size[0]}x{img_size[1]} 픽셀")
    
    # 메인 컨텐츠
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 차트 이미지 업로드")
        
        uploaded_file = st.file_uploader(
            "주식 차트 이미지를 선택하세요",
            type=['png', 'jpg', 'jpeg'],
            help="PNG, JPG, JPEG 형식의 이미지를 업로드하세요"
        )
        
        if uploaded_file is not None:
            # 이미지 표시
            image = Image.open(uploaded_file)
            st.image(image, caption="업로드된 차트 이미지", use_container_width=True)
            
            # 이미지 정보
            st.markdown(f"""
            <div class="metric-card">
                <strong>이미지 정보</strong><br>
                크기: {image.size[0]} x {image.size[1]} 픽셀<br>
                형식: {image.format}<br>
                모드: {image.mode}
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("🎯 예측 결과")
        
        if uploaded_file is not None:
            with st.spinner("🔍 차트 패턴 분석 중..."):
                try:
                    # 예측 수행 (모델에 맞는 크기로 자동 전처리)
                    prediction_class, prediction_proba = predict_stock_chart(model, image, img_size)
                    
                    # 예측된 클래스에 대한 신뢰도 계산
                    if prediction_class == 1:  # UP 예측
                        confidence = prediction_proba  # UP 확률 그대로
                    else:  # DOWN 예측
                        confidence = 1 - prediction_proba  # DOWN 확률로 변환
                    
                except Exception as e:
                    st.error(f"⚠️ 예측 중 오류가 발생했습니다: {e}")
                    st.error("터미널(콘솔)에서 자세한 디버그 정보를 확인하세요.")
                    st.stop()
                
                # 결과 표시
                if prediction_class == 1:  # Up
                    st.markdown(f"""
                        📈 상승 (UP) 예측
                    </div>
                    """, unsafe_allow_html=True)
                    prediction_text = "상승"
                    advice = "💡 **투자 조언**: 매수 고려 시점일 수 있습니다."
                    confidence_color = "#667eea"
                else:  # Down
                    st.markdown(f"""
                    <div class="prediction-box down-prediction">
                        📉 하락 (DOWN) 예측
                    </div>
                    """, unsafe_allow_html=True)
                    prediction_text = "하락"
                    advice = "💡 **투자 조언**: 매도 또는 관망을 고려하세요."
                    confidence_color = "#f5576c"
                
                # 신뢰도 표시
                st.markdown("### 신뢰도 분석")
                
                # 게이지 차트
                fig = create_confidence_gauge(confidence, prediction_class)
                st.pyplot(fig)
                plt.close()
                
                # 상세 지표
                st.markdown(f"""
                <div class="metric-card">
                    <strong>상세 분석</strong><br>
                    예측 클래스: <strong>{prediction_text}</strong><br>
                    모델 원본 확률: <strong>{prediction_proba:.4f}</strong> (UP: {prediction_proba*100:.2f}%, DOWN: {(1-prediction_proba)*100:.2f}%)<br>
                    예측 신뢰도: <strong>{confidence:.4f}</strong> ({confidence*100:.2f}%)<br>
                    신뢰도 수준: <strong>{'높음' if confidence > 0.7 else '보통' if confidence > 0.55 else '낮음'}</strong>
                </div>
                """, unsafe_allow_html=True)
                
                st.info(advice)
                
                # 해석 가이드
                with st.expander("📊 결과 해석 가이드"):
                    st.markdown("""
                    **신뢰도 해석**:
                    - **70% 이상**: 높은 신뢰도 - 모델이 해당 방향에 대해 확신
                    - **55% ~ 70%**: 보통 신뢰도 - 어느 정도 확신하지만 불확실성 존재
                    - **55% 미만**: 낮은 신뢰도 - 불확실성 높음, 신중한 판단 필요
                    
                    **모델 원본 확률**:
                    - 모델이 출력하는 값은 **UP(상승) 확률**입니다
                    - DOWN 확률 = 100% - UP 확률
                    - 예: UP 26.7% → DOWN 73.3% → DOWN 예측 (신뢰도 73.3%)
                    
                    **주의사항**:
                    - 이 예측은 참고용이며, 실제 투자 결정은 종합적 판단이 필요합니다
                    - 과거 패턴이 미래를 보장하지 않습니다
                    - 다른 기술적/기본적 분석과 함께 활용하세요
                    """)
        
        else:
            st.info("👈 왼쪽에서 차트 이미지를 업로드하세요")
            
            # 예시 이미지 표시
            st.markdown("### 📋 예시")
            
            # 샘플 이미지가 있다면 표시
            if os.path.exists("charts/up") and os.path.exists("charts/down"):
                example_col1, example_col2 = st.columns(2)
                
                with example_col1:
                    st.markdown("**상승 패턴 예시**")
                    up_files = [f for f in os.listdir("charts/up") if f.endswith(('.png', '.jpg', '.jpeg'))]
                    if up_files:
                        sample_up = os.path.join("charts/up", up_files[0])
                        st.image(sample_up, use_container_width=True)
                
                with example_col2:
                    st.markdown("**하락 패턴 예시**")
                    down_files = [f for f in os.listdir("charts/down") if f.endswith(('.png', '.jpg', '.jpeg'))]
                    if down_files:
                        sample_down = os.path.join("charts/down", down_files[0])
                        st.image(sample_down, use_container_width=True)
    
    # 푸터
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 2rem;'>
            <p>📊 주식 차트 패턴 예측기 v1.0 | ResNet50 Transfer Learning</p>
            <p style='font-size: 0.9rem;'>⚠️ 본 예측 결과는 투자 조언이 아니며, 실제 투자 시 신중한 판단이 필요합니다.</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
