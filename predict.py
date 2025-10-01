"""
학습된 모델을 사용하여 새로운 차트 이미지 예측
"""

import os
import sys
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Windows 콘솔 인코딩 설정
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'ignore')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'ignore')

class StockChartPredictor:
    """주식 차트 예측 클래스"""
    
    def __init__(self, model_path='models/best_stock_chart_model.h5', img_size=(100, 100)):
        """
        Args:
            model_path: 학습된 모델 경로
            img_size: 이미지 크기 (width, height)
        """
        self.img_size = img_size
        self.model = self.load_model(model_path)
    
    def load_model(self, model_path):
        """모델 로드"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
        
        print(f"📦 모델 로딩 중: {model_path}")
        model = keras.models.load_model(model_path)
        print("✅ 모델 로드 완료")
        return model
    
    def preprocess_image(self, image_source):
        """이미지 전처리 (경로 또는 바이트 스트림을 허용)"""
        
        # PIL.Image.open()은 경로뿐만 아니라 바이트 스트림(BytesIO)도 처리 가능
        img = Image.open(image_source) 
        
        # RGB로 변환 (RGBA일 경우 대비)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # 리사이즈
        img = img.resize(self.img_size)
        
        # 배열로 변환 및 정규화
        img_array = np.array(img) / 255.0
        
        # 배치 차원 추가
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, img
    
    # 기존 predict 메서드를 수정하여 경로/바이트 처리를 통합
    def predict(self, image_source, verbose=True): # 'image_path'를 'image_source'로 변경
        """예측 수행"""
        # 이미지 전처리
        # image_source는 파일 경로 또는 BytesIO 객체가 될 수 있습니다.
        img_array, original_img = self.preprocess_image(image_source) 
        
        # 예측
        prediction = self.model.predict(img_array, verbose=0)[0][0]
        
        # 파일 이름은 image_source가 경로일 때만 추출 가능
        file_name = os.path.basename(image_source) if isinstance(image_source, str) else "Uploaded Chart"

        result = {
            'file': file_name, # 파일명 추출 방식 수정
            'prediction': 'Up (상승)' if prediction > 0.5 else 'Down (하락)',
            'confidence': prediction if prediction > 0.5 else 1 - prediction,
            'up_probability': prediction,
            'down_probability': 1 - prediction,
            'original_image': original_img
        }
        
        if verbose:
            print("\n" + "=" * 60)
            print(f"📊 예측 결과: {result['file']}")
            print("=" * 60)
            print(f"🎯 예측: {result['prediction']}")
            print(f"💯 신뢰도: {result['confidence']:.2%}")
            print(f"📈 상승 확률: {result['up_probability']:.2%}")
            print(f"📉 하락 확률: {result['down_probability']:.2%}")
            print("=" * 60)
        
        return result
    
    def predict_batch(self, image_paths):
        """여러 이미지 예측"""
        results = []
        
        print(f"\n🔄 {len(image_paths)}개 이미지 예측 중...")
        
        for i, image_path in enumerate(image_paths, 1):
            try:
                result = self.predict(image_path, verbose=False)
                results.append(result)
                
                if i % 10 == 0:
                    print(f"  진행률: {i}/{len(image_paths)} ({i/len(image_paths)*100:.1f}%)")
            
            except Exception as e:
                print(f"❌ 오류 발생 ({image_path}): {str(e)}")
                continue
        
        print(f"✅ 완료: {len(results)}/{len(image_paths)}개 성공")
        
        return results
    
    def visualize_prediction(self, result, save_path=None):
        """예측 결과 시각화"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # 이미지 표시
        ax.imshow(result['original_image'])
        ax.axis('off')
        
        # 예측 결과 텍스트
        prediction_text = f"예측: {result['prediction']}\n신뢰도: {result['confidence']:.2%}"
        
        # 배경색 설정 (상승: 빨강, 하락: 파랑)
        bg_color = 'red' if result['prediction'].startswith('Up') else 'blue'
        
        ax.text(0.5, -0.1, prediction_text,
               transform=ax.transAxes,
               fontsize=14,
               fontweight='bold',
               ha='center',
               bbox=dict(boxstyle='round', facecolor=bg_color, alpha=0.7, edgecolor='white', linewidth=2),
               color='white')
        
        plt.title(f"차트 분석: {result['file']}", fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 시각화 저장: {save_path}")
        
        plt.show()


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='주식 차트 이미지 예측')
    parser.add_argument('--image', type=str, help='예측할 이미지 경로')
    parser.add_argument('--model', type=str, default='models/best_stock_chart_model.h5',
                       help='모델 경로 (기본값: models/best_stock_chart_model.h5)')
    parser.add_argument('--batch', type=str, help='여러 이미지가 있는 디렉토리 경로')
    parser.add_argument('--visualize', action='store_true', help='결과 시각화')
    
    args = parser.parse_args()
    
    # Predictor 생성
    predictor = StockChartPredictor(model_path=args.model)
    
    if args.image:
        # 단일 이미지 예측
        result = predictor.predict(args.image)
        
        if args.visualize:
            predictor.visualize_prediction(result)
    
    elif args.batch:
        # 배치 예측
        image_files = [os.path.join(args.batch, f) for f in os.listdir(args.batch)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        results = predictor.predict_batch(image_files)
        
        # 통계 출력
        print("\n" + "=" * 60)
        print("📊 배치 예측 통계")
        print("=" * 60)
        up_count = sum(1 for r in results if r['prediction'].startswith('Up'))
        down_count = len(results) - up_count
        avg_confidence = np.mean([r['confidence'] for r in results])
        
        print(f"📈 상승 예측: {up_count}개 ({up_count/len(results)*100:.1f}%)")
        print(f"📉 하락 예측: {down_count}개 ({down_count/len(results)*100:.1f}%)")
        print(f"💯 평균 신뢰도: {avg_confidence:.2%}")
        print("=" * 60)
    
    else:
        print("❌ --image 또는 --batch 옵션을 지정해주세요.")
        print("예시:")
        print("  python predict.py --image dataset-2021/up/sample.jpg")
        print("  python predict.py --batch dataset-2021/up --model models/best_stock_chart_model.h5")


if __name__ == '__main__':
    main()
