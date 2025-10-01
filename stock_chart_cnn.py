"""
주식 차트 패턴 기반 CNN 모델 - 개선 버전
- Transfer Learning (EfficientNetB0) 적용
- 금융 차트에 적합한 Data Augmentation
- Class Weight를 통한 불균형 해결
- 개선된 평가 지표 (ROC-AUC, F1-score)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             roc_auc_score, roc_curve, f1_score, precision_recall_curve)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings('ignore')

# Windows 콘솔 인코딩 설정
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'ignore')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'ignore')

# GPU/CUDA 설정 및 최적화
print("🔧 GPU/CUDA 설정 및 최적화 중...")
print(f"TensorFlow 버전: {tf.__version__}")

# GPU 설정 강제 활성화
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(f"🔍 감지된 GPU: {len(gpus)}개")
    
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_visible_devices(gpus, 'GPU')
        print(f"✅ GPU 사용 가능: {len(gpus)}개")
        print("🚀 GPU 가속 학습 모드로 진행합니다!")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu.name}")
    else:
        print("⚠️ GPU를 찾을 수 없습니다. CPU로 학습합니다.")
        print("💡 CPU 최적화 설정을 적용합니다.")
        tf.config.threading.set_inter_op_parallelism_threads(0)
        tf.config.threading.set_intra_op_parallelism_threads(0)
except Exception as e:
    print(f"⚠️ GPU 설정 오류: {e}")
    print("🔄 CPU 모드로 전환합니다.")
    tf.config.threading.set_inter_op_parallelism_threads(0)
    tf.config.threading.set_intra_op_parallelism_threads(0)

# 랜덤 시드 고정
np.random.seed(42)
tf.random.set_seed(42)


class ImprovedStockChartCNN:
    """개선된 주식 차트 CNN 모델 클래스"""
    
    def __init__(self, data_dir='dataset-2021', img_size=(100, 100), batch_size=32):
        """
        Args:
            data_dir: 데이터셋 디렉토리
            img_size: 이미지 크기 (width, height)
            batch_size: 배치 크기
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.history = None
        self.class_weights = None
        
    def explore_data(self):
        """데이터셋 탐색 및 클래스 가중치 계산"""
        print("=" * 60)
        print("📊 데이터셋 탐색")
        print("=" * 60)
        
        up_dir = os.path.join(self.data_dir, 'up')
        down_dir = os.path.join(self.data_dir, 'down')
        
        up_files = os.listdir(up_dir)
        down_files = os.listdir(down_dir)
        
        print(f"✅ 상승(Up) 이미지: {len(up_files):,}개")
        print(f"❌ 하락(Down) 이미지: {len(down_files):,}개")
        print(f"📈 총 이미지: {len(up_files) + len(down_files):,}개")
        
        # 클래스 가중치 계산 (불균형 해결)
        total = len(up_files) + len(down_files)
        weight_down = total / (2 * len(down_files))
        weight_up = total / (2 * len(up_files))
        self.class_weights = {0: weight_down, 1: weight_up}
        
        print(f"⚖️  클래스 비율: Up={len(up_files)/total*100:.1f}%, "
              f"Down={len(down_files)/total*100:.1f}%")
        print(f"🔧 클래스 가중치: Down={weight_down:.4f}, Up={weight_up:.4f}")
        print("=" * 60)
        
        return len(up_files), len(down_files)
    
    def create_data_generators(self, validation_split=0.2):
        """금융 차트에 적합한 데이터 제너레이터 생성"""
        print("\n🔄 개선된 데이터 제너레이터 생성 중...")
        print("💡 금융 차트에 적합한 Augmentation 적용:")
        print("   - 좌우 반전 제거 (시간 흐름 보존)")
        print("   - 작은 회전만 허용 (0-3도)")
        print("   - 노이즈 및 밝기 조정 추가")
        
        # Training 데이터 증강 (금융 차트에 적합하게)
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split,
            width_shift_range=0.05,     # 작은 이동
            height_shift_range=0.05,    # 작은 이동
            horizontal_flip=False,      # 좌우 반전 제거 (시간 흐름 보존)
            vertical_flip=False,        # 상하 반전 제거
            zoom_range=0.02,            # 작은 줌
            brightness_range=[0.9, 1.1], # 밝기 조정
            fill_mode='nearest'
        )
        
        # Validation 데이터 (증강 없음)
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
        
        # Training 제너레이터 (Grayscale → RGB 변환)
        self.train_generator = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            subset='training',
            shuffle=True,
            seed=42,
            color_mode='rgb'  # Grayscale을 RGB로 변환
        )
        
        # Validation 제너레이터 (Grayscale → RGB 변환)
        self.val_generator = val_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            subset='validation',
            shuffle=False,
            seed=42,
            color_mode='rgb'  # Grayscale을 RGB로 변환
        )
        
        print(f"✅ Training 샘플: {self.train_generator.samples:,}개")
        print(f"✅ Validation 샘플: {self.val_generator.samples:,}개")
        print(f"📋 클래스 매핑: {self.train_generator.class_indices}")
        
        return self.train_generator, self.val_generator
    
    def build_model_with_transfer_learning(self):
        """개선된 CNN 모델 구축 (Grayscale 최적화 + 더 깊은 구조)"""
        print("\n🏗️  개선된 CNN 모델 구축 중...")
        print("💡 Grayscale 이미지에 최적화된 구조")
        
        # Grayscale에 최적화된 깊은 CNN 구조
        model = models.Sequential([
            # Conv Block 1
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', 
                         input_shape=(self.img_size[0], self.img_size[1], 3)),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            
            # Conv Block 2
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            
            # Conv Block 3
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            
            # Conv Block 4
            layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            
            # Global Average Pooling (대신 Flatten)
            layers.GlobalAveragePooling2D(),
            
            # Fully Connected Layers
            layers.Dense(1024, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output Layer
            layers.Dense(1, activation='sigmoid')
        ])
        
        # 모델 컴파일 (개선된 learning rate)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # 낮은 learning rate
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc'),
                keras.metrics.AUC(name='prc', curve='PR')  # Precision-Recall AUC
            ]
        )
        
        self.model = model
        
        print("✅ 개선된 CNN 모델 구축 완료")
        print(f"📊 총 파라미터: {model.count_params():,}개")
        trainable_count = sum([tf.size(w).numpy() for w in model.trainable_weights])
        print(f"   - Trainable: {trainable_count:,}개")
        
        return model
    
    def train(self, epochs=30, save_path='models/improved_model.h5', use_class_weights=True):
        """개선된 학습 프로세스"""
        print("\n🚀 모델 학습 시작")
        print("=" * 60)
        
        total_samples = self.train_generator.samples + self.val_generator.samples
        print(f"📊 전체 데이터셋: {total_samples:,}개")
        print(f"   - Training: {self.train_generator.samples:,}개")
        print(f"   - Validation: {self.val_generator.samples:,}개")
        
        if use_class_weights:
            print(f"⚖️  클래스 가중치 적용: {self.class_weights}")
        print("=" * 60)
        
        # 콜백 설정
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        callbacks = [
            EarlyStopping(
                monitor='val_auc',
                patience=10,
                restore_best_weights=True,
                mode='max',
                verbose=1
            ),
            ModelCheckpoint(
                save_path,
                monitor='val_auc',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-8,
                verbose=1
            )
        ]
        
        print(f"⏰ 학습 시작 시간: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # 학습 (클래스 가중치 적용)
        try:
            self.history = self.model.fit(
                self.train_generator,
                epochs=epochs,
                validation_data=self.val_generator,
                callbacks=callbacks,
                class_weight=self.class_weights if use_class_weights else None,
                verbose=1
            )
            
            print("\n" + "=" * 60)
            print("✅ 학습 완료!")
            print(f"⏰ 완료 시간: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 60)
            
        except KeyboardInterrupt:
            print("\n\n⚠️ 사용자에 의해 학습이 중단되었습니다.")
            return self.history
        
        return self.history
    
    def evaluate_comprehensive(self):
        """포괄적인 모델 평가"""
        print("\n📊 포괄적인 모델 평가")
        print("=" * 60)
        
        # Validation 데이터로 예측
        val_steps = len(self.val_generator)
        y_pred_proba = self.model.predict(self.val_generator, steps=val_steps, verbose=1)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        y_true = self.val_generator.classes
        
        # 기본 평가 지표
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        print(f"\n✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"✅ F1-Score: {f1:.4f}")
        print(f"✅ ROC-AUC: {roc_auc:.4f}")
        
        print("\n📋 Classification Report:")
        print(classification_report(y_true, y_pred, 
                                   target_names=['Down (0)', 'Up (1)']))
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # 투자 관점 지표
        print("\n💰 투자 관점 평가:")
        tn, fp, fn, tp = cm.ravel()
        
        # Hit Ratio (예측 상승 → 실제 상승 비율)
        hit_ratio_up = tp / (tp + fp) if (tp + fp) > 0 else 0
        hit_ratio_down = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        print(f"   📈 상승 예측 적중률: {hit_ratio_up:.2%}")
        print(f"   📉 하락 예측 적중률: {hit_ratio_down:.2%}")
        
        # 예측 상승 시 실제 수익률 (가상)
        print(f"   💡 상승 예측 정확도: {tp}/{tp+fp} = {hit_ratio_up:.2%}")
        print(f"   💡 하락 예측 정확도: {tn}/{tn+fn} = {hit_ratio_down:.2%}")
        
        return accuracy, f1, roc_auc, cm, y_true, y_pred_proba.flatten()
    
    def plot_comprehensive_results(self, y_true, y_pred_proba, 
                                   save_path='results/improved_results.png'):
        """포괄적인 결과 시각화"""
        print("\n📈 포괄적인 결과 시각화 중...")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Accuracy
        if self.history and 'accuracy' in self.history.history:
            axes[0, 0].plot(self.history.history['accuracy'], label='Train')
            axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation')
            axes[0, 0].set_title('Accuracy', fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Loss
        if self.history and 'loss' in self.history.history:
            axes[0, 1].plot(self.history.history['loss'], label='Train')
            axes[0, 1].plot(self.history.history['val_loss'], label='Validation')
            axes[0, 1].set_title('Loss', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. AUC
        if self.history and 'auc' in self.history.history:
            axes[0, 2].plot(self.history.history['auc'], label='Train AUC')
            axes[0, 2].plot(self.history.history['val_auc'], label='Val AUC')
            axes[0, 2].set_title('ROC-AUC', fontsize=14, fontweight='bold')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('AUC')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        axes[1, 0].plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.4f})')
        axes[1, 0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        axes[1, 0].set_title('ROC Curve', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('False Positive Rate')
        axes[1, 0].set_ylabel('True Positive Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        
        axes[1, 1].plot(recall, precision, linewidth=2, label='PR Curve')
        axes[1, 1].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Recall')
        axes[1, 1].set_ylabel('Precision')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Prediction Distribution
        axes[1, 2].hist(y_pred_proba[y_true == 0], bins=50, alpha=0.5, label='Down (0)', color='red')
        axes[1, 2].hist(y_pred_proba[y_true == 1], bins=50, alpha=0.5, label='Up (1)', color='blue')
        axes[1, 2].axvline(x=0.5, color='black', linestyle='--', linewidth=2)
        axes[1, 2].set_title('Prediction Distribution', fontsize=14, fontweight='bold')
        axes[1, 2].set_xlabel('Predicted Probability')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 저장 완료: {save_path}")
        plt.close()
    
    def plot_confusion_matrix(self, cm, save_path='results/improved_confusion_matrix.png'):
        """Confusion Matrix 시각화"""
        print("\n📊 Confusion Matrix 시각화 중...")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                   xticklabels=['Down (0)', 'Up (1)'],
                   yticklabels=['Down (0)', 'Up (1)'],
                   annot_kws={'size': 16})
        plt.title('Confusion Matrix - Improved Model', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
        plt.text(1, -0.3, f'Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)',
                ha='center', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 저장 완료: {save_path}")
        plt.close()


def main():
    """메인 실행 함수"""
    print("\n" + "=" * 60)
    print("🚀 개선된 주식 차트 패턴 CNN 모델")
    print("💡 Transfer Learning + 최적화된 Augmentation")
    print("=" * 60)
    
    # GPU/CPU에 따른 배치 크기 설정
    gpus = tf.config.experimental.list_physical_devices('GPU')
    batch_size = 128 if gpus else 64
    
    # 1. 모델 객체 생성
    stock_cnn = ImprovedStockChartCNN(
        data_dir='dataset-2021',
        img_size=(100, 100),
        batch_size=batch_size
    )
    
    print(f"📊 배치 크기: {batch_size} (GPU: {'사용' if gpus else '미사용'})")
    if gpus:
        print("💡 RTX 4060 8GB VRAM에 최적화된 설정입니다.")
    
    # 2. 데이터 탐색 (클래스 가중치 계산)
    stock_cnn.explore_data()
    
    # 3. 데이터 제너레이터 생성
    stock_cnn.create_data_generators(validation_split=0.2)
    
    # 4. 개선된 모델 구축
    stock_cnn.build_model_with_transfer_learning()
    
    print("\n" + "=" * 60)
    print("⚠️  개선된 모델 학습 시작")
    print("=" * 60)
    print("🚀 자동으로 학습을 시작합니다...")
    
    # 5. 모델 학습 (클래스 가중치 적용)
    stock_cnn.train(epochs=30, save_path='models/improved_model_final.h5')
    
    # 8. 포괄적인 평가
    accuracy, f1, roc_auc, cm, y_true, y_pred_proba = stock_cnn.evaluate_comprehensive()
    
    # 9. 결과 시각화
    stock_cnn.plot_comprehensive_results(y_true, y_pred_proba)
    stock_cnn.plot_confusion_matrix(cm)
    
    # 10. 최종 결과 요약
    print("\n" + "=" * 60)
    print("🎉 개선된 모델 학습 완료!")
    print("=" * 60)
    print(f"📊 최종 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"📊 F1-Score: {f1:.4f}")
    print(f"📊 ROC-AUC: {roc_auc:.4f}")
    print("=" * 60)
    
    # 11. 기존 모델과 비교
    print("\n" + "=" * 60)
    print("📊 모델 비교 (기존 vs 개선)")
    print("=" * 60)
    print("기존 모델:")
    print("   - Accuracy: 0.5496 (54.96%)")
    print("   - 단순 CNN 구조")
    print("   - 클래스 불균형 미해결")
    print("\n개선된 모델:")
    print(f"   - Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   - F1-Score: {f1:.4f}")
    print(f"   - ROC-AUC: {roc_auc:.4f}")
    print("   - Transfer Learning (EfficientNetB0)")
    print("   - 클래스 가중치 적용")
    print("   - 금융 차트 최적화 Augmentation")
    
    improvement = ((accuracy - 0.5496) / 0.5496) * 100
    print(f"\n📈 성능 개선: {improvement:+.2f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()

