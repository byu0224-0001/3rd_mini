"""
주식 차트 패턴 기반 CNN 모델 - Fine-tuning + 하이퍼파라미터 최적화 버전
- ResNet50 Transfer Learning + Fine-tuning
- 최적화된 하이퍼파라미터 (Learning rate, Batch size, Dropout)
- 개선된 Optimizer 및 Learning rate scheduler
- 5,000장 데이터셋으로 효율적 학습
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import shutil
import random
from pathlib import Path
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             roc_auc_score, roc_curve, f1_score, precision_recall_curve)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings('ignore')

# Windows 콘솔 인코딩 설정
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'ignore')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'ignore')

# GPU/CUDA 설정
print("🔧 GPU/CUDA 설정 및 최적화 중...")
print(f"TensorFlow 버전: {tf.__version__}")

try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(f"🔍 감지된 GPU: {len(gpus)}개")
    
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU 사용: {len(gpus)}개")
    else:
        print("⚠️ GPU를 찾을 수 없습니다. CPU로 학습합니다.")
        tf.config.threading.set_inter_op_parallelism_threads(0)
        tf.config.threading.set_intra_op_parallelism_threads(0)
except Exception as e:
    print(f"⚠️ GPU 설정 오류: {e}")

# 랜덤 시드 고정
np.random.seed(42)
tf.random.set_seed(42)


class FineTunedStockChartCNN:
    """Fine-tuning + 하이퍼파라미터 최적화 모델"""
    
    def __init__(self, data_dir, img_size=(128, 128), batch_size=32):
        """
        Args:
            data_dir: 데이터셋 디렉토리
            img_size: 이미지 크기 (개선: 100x100 → 128x128)
            batch_size: 배치 크기
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.base_model = None
        self.history_stage1 = None
        self.history_stage2 = None
        self.class_weights = None
        
    def explore_data(self):
        """데이터셋 탐색 및 클래스 가중치 계산"""
        print("="*60)
        print("📊 데이터셋 탐색")
        print("="*60)
        
        up_dir = os.path.join(self.data_dir, 'up')
        down_dir = os.path.join(self.data_dir, 'down')
        
        up_files = [f for f in os.listdir(up_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        down_files = [f for f in os.listdir(down_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"✅ 상승(Up) 이미지: {len(up_files):,}개")
        print(f"❌ 하락(Down) 이미지: {len(down_files):,}개")
        print(f"📈 총 이미지: {len(up_files) + len(down_files):,}개")
        
        # 클래스 가중치 계산
        total = len(up_files) + len(down_files)
        weight_down = total / (2 * len(down_files))
        weight_up = total / (2 * len(up_files))
        self.class_weights = {0: weight_down, 1: weight_up}
        
        print(f"⚖️  클래스 비율: Up={len(up_files)/total*100:.1f}%, Down={len(down_files)/total*100:.1f}%")
        print(f"🔧 클래스 가중치: Down={weight_down:.4f}, Up={weight_up:.4f}")
        print("="*60)
        
    def create_data_generators(self, validation_split=0.2):
        """최적화된 데이터 제너레이터 생성"""
        print("\n🔄 최적화된 데이터 제너레이터 생성 중...")
        print("💡 개선된 Augmentation:")
        print("   - 이미지 크기: 128x128 (100x100에서 증가)")
        print("   - 작은 회전: 0-5도")
        print("   - 약간의 shift, zoom, brightness 조정")
        
        # Training 데이터 증강
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split,
            rotation_range=5,           # 약간 증가
            width_shift_range=0.08,     # 약간 증가
            height_shift_range=0.08,    # 약간 증가
            zoom_range=0.05,            # 약간 증가
            brightness_range=[0.85, 1.15],  # 범위 증가
            horizontal_flip=False,
            vertical_flip=False,
            fill_mode='nearest'
        )
        
        # Validation 데이터
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
        
        # Training generator
        self.train_generator = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            subset='training',
            shuffle=True,
            color_mode='rgb',
            seed=42
        )
        
        # Validation generator
        self.val_generator = val_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            subset='validation',
            shuffle=False,
            color_mode='rgb',
            seed=42
        )
        
        print(f"✅ Training 샘플: {self.train_generator.samples:,}개")
        print(f"✅ Validation 샘플: {self.val_generator.samples:,}개")
        print(f"📋 클래스 매핑: {self.train_generator.class_indices}")
        
    def build_model_stage1(self, dropout_rate=0.4):
        """1단계: Transfer Learning 모델 구축 (Base frozen)"""
        print("\n🏗️  1단계: Transfer Learning 모델 구축 (ResNet50)...")
        print("💡 하이퍼파라미터 최적화 적용")
        
        # ResNet50 base model
        base_model = ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=(self.img_size[0], self.img_size[1], 3),
            pooling='avg'
        )
        
        base_model.trainable = False
        
        print(f"📊 Base Model: {len(base_model.layers)}개 레이어 (Frozen)")
        
        # Custom top layers (최적화된 구조)
        inputs = keras.Input(shape=(self.img_size[0], self.img_size[1], 3))
        x = base_model(inputs, training=False)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(512, activation='relu', 
                        kernel_regularizer=keras.regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate + 0.1)(x)
        x = layers.Dense(256, activation='relu',
                        kernel_regularizer=keras.regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate + 0.1)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = keras.Model(inputs, outputs)
        
        # 최적화된 Optimizer 설정
        optimizer = keras.optimizers.Adam(
            learning_rate=0.0001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc'),
                keras.metrics.AUC(name='prc', curve='PR')
            ]
        )
        
        self.model = model
        self.base_model = base_model
        
        print("✅ 1단계 모델 구축 완료")
        print(f"📊 총 파라미터: {model.count_params():,}개")
        print(f"   - Dropout: {dropout_rate}")
        print(f"   - L2 Regularization: 0.001")
        
        return model
        
    def build_model_stage2(self, unfreeze_layers=50, learning_rate=0.00001):
        """2단계: Fine-tuning 모델 설정"""
        print(f"\n🔓 2단계: Fine-tuning 준비...")
        print(f"💡 마지막 {unfreeze_layers}개 레이어 Unfreeze")
        print(f"💡 Learning rate: {learning_rate} (10배 감소)")
        
        # Base model unfreeze
        self.base_model.trainable = True
        frozen_layers = len(self.base_model.layers) - unfreeze_layers
        
        for i, layer in enumerate(self.base_model.layers):
            if i < frozen_layers:
                layer.trainable = False
            else:
                layer.trainable = True
        
        # Fine-tuning용 낮은 learning rate로 재컴파일
        optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc'),
                keras.metrics.AUC(name='prc', curve='PR')
            ]
        )
        
        trainable_count = sum([tf.size(w).numpy() for w in self.model.trainable_weights])
        print(f"✅ Fine-tuning 준비 완료")
        print(f"📊 Trainable 파라미터: {trainable_count:,}개")
        
    def train_stage(self, stage_name, epochs, save_path, patience=15):
        """단계별 학습"""
        print(f"\n{'='*60}")
        print(f"🚀 {stage_name} 학습 시작")
        print(f"{'='*60}")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 최적화된 Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_auc',
                patience=patience,
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
                patience=8,
                min_lr=1e-8,
                verbose=1
            )
        ]
        
        print(f"⏰ 시작 시간: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            history = self.model.fit(
                self.train_generator,
                epochs=epochs,
                validation_data=self.val_generator,
                callbacks=callbacks,
                class_weight=self.class_weights,
                verbose=1
            )
            
            print(f"\n✅ {stage_name} 완료!")
            print(f"⏰ 완료 시간: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            return history
            
        except KeyboardInterrupt:
            print(f"\n⚠️ {stage_name} 중단됨")
            return None
            
    def evaluate_comprehensive(self):
        """포괄적인 모델 평가"""
        print("\n"+"="*60)
        print("📊 최종 모델 평가")
        print("="*60)
        
        # 예측
        val_steps = len(self.val_generator)
        y_pred_proba = self.model.predict(self.val_generator, steps=val_steps, verbose=1)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        y_true = self.val_generator.classes
        
        # 평가 지표
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        print(f"\n✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"✅ F1-Score: {f1:.4f}")
        print(f"✅ ROC-AUC: {roc_auc:.4f}")
        
        print("\n📋 Classification Report:")
        print(classification_report(y_true, y_pred, target_names=['Down (0)', 'Up (1)']))
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # 투자 관점 평가
        print("\n💰 투자 관점 평가:")
        tn, fp, fn, tp = cm.ravel()
        hit_ratio_up = tp / (tp + fp) if (tp + fp) > 0 else 0
        hit_ratio_down = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        print(f"   📈 상승 예측 적중률: {hit_ratio_up:.2%}")
        print(f"   📉 하락 예측 적중률: {hit_ratio_down:.2%}")
        print(f"   💡 상승 예측: {tp+fp}건 중 {tp}건 적중")
        print(f"   💡 하락 예측: {tn+fn}건 중 {tn}건 적중")
        
        return accuracy, f1, roc_auc, cm, y_true, y_pred_proba.flatten()
        
    def plot_results(self, y_true, y_pred_proba):
        """결과 시각화"""
        print("\n📈 결과 시각화 중...")
        os.makedirs('results', exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Stage 1 Accuracy
        if self.history_stage1:
            axes[0, 0].plot(self.history_stage1.history['accuracy'], label='Train', linewidth=2)
            axes[0, 0].plot(self.history_stage1.history['val_accuracy'], label='Val', linewidth=2)
            axes[0, 0].set_title('Stage 1: Transfer Learning Accuracy', fontweight='bold')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].legend()
            axes[0, 0].grid(alpha=0.3)
        
        # 2. Stage 2 Accuracy
        if self.history_stage2:
            axes[0, 1].plot(self.history_stage2.history['accuracy'], label='Train', linewidth=2)
            axes[0, 1].plot(self.history_stage2.history['val_accuracy'], label='Val', linewidth=2)
            axes[0, 1].set_title('Stage 2: Fine-tuning Accuracy', fontweight='bold')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(alpha=0.3)
        
        # 3. Combined AUC
        if self.history_stage1 and self.history_stage2:
            epochs_s1 = len(self.history_stage1.history['auc'])
            epochs_s2 = len(self.history_stage2.history['auc'])
            
            axes[0, 2].plot(range(1, epochs_s1+1), self.history_stage1.history['auc'], 
                          label='Stage 1 Train', linewidth=2, color='blue')
            axes[0, 2].plot(range(1, epochs_s1+1), self.history_stage1.history['val_auc'], 
                          label='Stage 1 Val', linewidth=2, color='lightblue')
            axes[0, 2].plot(range(epochs_s1+1, epochs_s1+epochs_s2+1), self.history_stage2.history['auc'], 
                          label='Stage 2 Train', linewidth=2, color='red')
            axes[0, 2].plot(range(epochs_s1+1, epochs_s1+epochs_s2+1), self.history_stage2.history['val_auc'], 
                          label='Stage 2 Val', linewidth=2, color='salmon')
            axes[0, 2].axvline(x=epochs_s1, color='green', linestyle='--', linewidth=2, label='Fine-tuning 시작')
            axes[0, 2].set_title('ROC-AUC (2-Stage)', fontweight='bold')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('AUC')
            axes[0, 2].legend()
            axes[0, 2].grid(alpha=0.3)
        
        # 4. ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        axes[1, 0].plot(fpr, tpr, linewidth=3, label=f'ROC (AUC = {roc_auc:.4f})', color='blue')
        axes[1, 0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        axes[1, 0].set_title('ROC Curve', fontweight='bold')
        axes[1, 0].set_xlabel('False Positive Rate')
        axes[1, 0].set_ylabel('True Positive Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        axes[1, 0].fill_between(fpr, tpr, alpha=0.2)
        
        # 5. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        
        axes[1, 1].plot(recall, precision, linewidth=3, label='PR Curve', color='green')
        axes[1, 1].set_title('Precision-Recall Curve', fontweight='bold')
        axes[1, 1].set_xlabel('Recall')
        axes[1, 1].set_ylabel('Precision')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        axes[1, 1].fill_between(recall, precision, alpha=0.2)
        
        # 6. Prediction Distribution
        axes[1, 2].hist(y_pred_proba[y_true == 0], bins=50, alpha=0.6, label='Down (실제)', color='red', edgecolor='darkred')
        axes[1, 2].hist(y_pred_proba[y_true == 1], bins=50, alpha=0.6, label='Up (실제)', color='blue', edgecolor='darkblue')
        axes[1, 2].axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
        axes[1, 2].set_title('Prediction Distribution', fontweight='bold')
        axes[1, 2].set_xlabel('Predicted Probability')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].legend()
        axes[1, 2].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/finetuned_results.png', dpi=300, bbox_inches='tight')
        print("✅ 저장: results/finetuned_results.png")
        plt.close()
        
        # Confusion Matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', cbar=True,
                   xticklabels=['Down (0)', 'Up (1)'],
                   yticklabels=['Down (0)', 'Up (1)'],
                   annot_kws={'size': 16})
        plt.title('Fine-tuned Model - Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
        plt.text(1, -0.3, f'Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)',
                ha='center', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('results/finetuned_confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("✅ 저장: results/finetuned_confusion_matrix.png")
        plt.close()


def main():
    """메인 실행 함수"""
    print("\n"+"="*60)
    print("🚀 Fine-tuning + 하이퍼파라미터 최적화 모델")
    print("💡 2단계 학습으로 최고 성능 달성")
    print("="*60)
    
    # 설정
    gpus = tf.config.experimental.list_physical_devices('GPU')
    
    # 하이퍼파라미터 설정
    hyperparams = {
        'img_size': (128, 128),      # 이미지 크기 증가
        'batch_size': 64 if not gpus else 128,  # GPU 있으면 더 큰 배치
        'dropout_rate': 0.4,         # Dropout 비율
        'stage1_epochs': 50,         # 1단계 에포크
        'stage2_epochs': 50,         # 2단계 에포크
        'stage1_lr': 0.0001,         # 1단계 learning rate
        'stage2_lr': 0.00001,        # 2단계 learning rate (10배 낮게)
        'unfreeze_layers': 60,       # Unfreeze할 레이어 수
        'patience': 15               # Early stopping patience
    }
    
    print(f"\n📊 하이퍼파라미터 설정:")
    for key, value in hyperparams.items():
        print(f"   - {key}: {value}")
    print("="*60)
    
    # 데이터셋 확인
    data_dir = 'dataset-subset-5k'
    if not os.path.exists(data_dir):
        print(f"\n❌ 데이터셋을 찾을 수 없습니다: {data_dir}")
        print("💡 먼저 stock_chart_cnn_improved.py를 실행하여 서브셋을 생성하세요.")
        return
    
    # 모델 초기화
    model = FineTunedStockChartCNN(
        data_dir=data_dir,
        img_size=hyperparams['img_size'],
        batch_size=hyperparams['batch_size']
    )
    
    # 데이터 탐색
    model.explore_data()
    
    # 데이터 제너레이터 생성
    model.create_data_generators(validation_split=0.2)
    
    # ===== STAGE 1: Transfer Learning =====
    print("\n"+"#"*60)
    print("📌 STAGE 1: Transfer Learning (Base Frozen)")
    print("#"*60)
    
    model.build_model_stage1(dropout_rate=hyperparams['dropout_rate'])
    
    model.history_stage1 = model.train_stage(
        stage_name="Stage 1 - Transfer Learning",
        epochs=hyperparams['stage1_epochs'],
        save_path='models/finetuned/stage1_best.keras',
        patience=hyperparams['patience']
    )
    
    # ===== STAGE 2: Fine-tuning =====
    print("\n"+"#"*60)
    print("📌 STAGE 2: Fine-tuning (Partial Unfreeze)")
    print("#"*60)
    
    model.build_model_stage2(
        unfreeze_layers=hyperparams['unfreeze_layers'],
        learning_rate=hyperparams['stage2_lr']
    )
    
    model.history_stage2 = model.train_stage(
        stage_name="Stage 2 - Fine-tuning",
        epochs=hyperparams['stage2_epochs'],
        save_path='models/finetuned/stage2_best.keras',
        patience=hyperparams['patience']
    )
    
    # ===== 최종 평가 =====
    accuracy, f1, roc_auc, cm, y_true, y_pred_proba = model.evaluate_comprehensive()
    
    # 결과 시각화
    model.plot_results(y_true, y_pred_proba)
    
    # 최종 요약
    print("\n"+"="*60)
    print("🎉 Fine-tuning + 하이퍼파라미터 최적화 완료!")
    print("="*60)
    print(f"📊 최종 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"📊 F1-Score: {f1:.4f}")
    print(f"📊 ROC-AUC: {roc_auc:.4f}")
    print("="*60)
    
    # 성능 비교
    print("\n"+"="*60)
    print("📊 모델 성능 비교")
    print("="*60)
    print("기존 단순 CNN:")
    print("   - Accuracy: 54.96%")
    print("\n이전 Transfer Learning (1단계만):")
    print("   - Accuracy: 53.30%")
    print("\n현재 Fine-tuned + 최적화:")
    print(f"   - Accuracy: {accuracy*100:.2f}%")
    print(f"   - F1-Score: {f1:.4f}")
    print(f"   - ROC-AUC: {roc_auc:.4f}")
    print(f"   - 하이퍼파라미터 최적화 적용")
    print(f"   - 2단계 Fine-tuning 완료")
    
    improvement = ((accuracy - 0.5496) / 0.5496) * 100
    print(f"\n📈 기존 대비 성능 변화: {improvement:+.2f}%")
    print("="*60)
    
    # 하이퍼파라미터 요약
    print("\n"+"="*60)
    print("⚙️  최종 하이퍼파라미터")
    print("="*60)
    for key, value in hyperparams.items():
        print(f"   {key}: {value}")
    print("="*60)


if __name__ == '__main__':
    main()

