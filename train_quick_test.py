"""
빠른 테스트를 위한 작은 서브셋 학습 스크립트
- 각 클래스당 1000개씩만 사용하여 빠른 프로토타입 테스트
"""

import os
import sys
import shutil
import random

# Windows 콘솔 인코딩 설정
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'ignore')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'ignore')

def create_subset_dataset(source_dir='dataset-2021', target_dir='dataset-subset', n_samples=1000):
    """서브셋 데이터셋 생성"""
    print(f"\n📦 서브셋 데이터셋 생성 중 (각 클래스당 {n_samples}개)...")
    
    # 기존 서브셋 디렉토리 삭제
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    
    # 디렉토리 생성
    os.makedirs(os.path.join(target_dir, 'up'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'down'), exist_ok=True)
    
    # Up 샘플 복사
    up_files = [f for f in os.listdir(os.path.join(source_dir, 'up')) if f.endswith('.jpg')]
    up_samples = random.sample(up_files, min(n_samples, len(up_files)))
    
    print(f"  📈 Up 이미지 복사 중...")
    for i, f in enumerate(up_samples):
        if (i + 1) % 200 == 0:
            print(f"     진행률: {i+1}/{n_samples}")
        src = os.path.join(source_dir, 'up', f)
        dst = os.path.join(target_dir, 'up', f)
        shutil.copy2(src, dst)
    
    # Down 샘플 복사
    down_files = [f for f in os.listdir(os.path.join(source_dir, 'down')) if f.endswith('.jpg')]
    down_samples = random.sample(down_files, min(n_samples, len(down_files)))
    
    print(f"  📉 Down 이미지 복사 중...")
    for i, f in enumerate(down_samples):
        if (i + 1) % 200 == 0:
            print(f"     진행률: {i+1}/{n_samples}")
        src = os.path.join(source_dir, 'down', f)
        dst = os.path.join(target_dir, 'down', f)
        shutil.copy2(src, dst)
    
    print(f"✅ 서브셋 데이터셋 생성 완료!")
    print(f"   위치: {target_dir}/")
    print(f"   총 {len(up_samples) + len(down_samples)}개 이미지")
    
    return target_dir


def train_quick_test():
    """빠른 테스트 학습"""
    from stock_chart_cnn import StockChartCNN
    
    print("\n" + "=" * 60)
    print("🚀 빠른 테스트 학습 시작")
    print("=" * 60)
    
    # 서브셋 생성 (각 클래스당 5000개)
    subset_dir = create_subset_dataset(n_samples=5000)
    
    # 모델 객체 생성
    stock_cnn = StockChartCNN(
        data_dir=subset_dir,
        img_size=(100, 100),  # 작은 이미지 크기로 빠른 학습
        batch_size=64
    )
    
    # 데이터 탐색
    stock_cnn.explore_data()
    
    # 데이터 제너레이터 생성
    stock_cnn.create_data_generators(validation_split=0.2)
    
    # 모델 구축
    stock_cnn.build_model()
    stock_cnn.model.summary()
    
    # 모델 구조 저장
    stock_cnn.save_model_summary(save_path='results/model_summary_test.txt')
    
    # 모델 학습 (적은 에포크)
    stock_cnn.train(epochs=20, save_path='models/test_stock_chart_model.h5')
    
    # 모델 평가
    accuracy, cm, y_true, y_pred = stock_cnn.evaluate()
    
    # 결과 시각화
    stock_cnn.plot_training_history(save_path='results/training_history_test.png')
    stock_cnn.plot_confusion_matrix(cm, save_path='results/confusion_matrix_test.png')
    
    # 샘플 예측 테스트
    print("\n" + "=" * 60)
    print("🔮 샘플 예측 테스트")
    print("=" * 60)
    
    # Up 샘플
    up_files = os.listdir(os.path.join(subset_dir, 'up'))
    up_sample = os.path.join(subset_dir, 'up', up_files[0])
    result = stock_cnn.predict_image(up_sample)
    print(f"\n📈 Up 샘플 예측:")
    print(f"   파일: {os.path.basename(up_sample)}")
    print(f"   예측: {result['prediction']}")
    print(f"   확률: {result['probability']:.2%}")
    print(f"   Up 확률: {result['up_prob']:.2%}, Down 확률: {result['down_prob']:.2%}")
    
    # Down 샘플
    down_files = os.listdir(os.path.join(subset_dir, 'down'))
    down_sample = os.path.join(subset_dir, 'down', down_files[0])
    result = stock_cnn.predict_image(down_sample)
    print(f"\n📉 Down 샘플 예측:")
    print(f"   파일: {os.path.basename(down_sample)}")
    print(f"   예측: {result['prediction']}")
    print(f"   확률: {result['probability']:.2%}")
    print(f"   Up 확률: {result['up_prob']:.2%}, Down 확률: {result['down_prob']:.2%}")
    
    print("\n" + "=" * 60)
    print("✅ 빠른 테스트 완료!")
    print(f"📊 최종 정확도: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("=" * 60)
    
    # 서브셋 디렉토리 제거 (선택사항)
    cleanup = input("\n서브셋 데이터를 삭제하시겠습니까? (y/n): ")
    if cleanup.lower() == 'y':
        shutil.rmtree(subset_dir)
        print(f"✅ 서브셋 데이터 삭제 완료: {subset_dir}")


if __name__ == '__main__':
    train_quick_test()
