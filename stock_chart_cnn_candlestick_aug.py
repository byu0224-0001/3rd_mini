"""
주식 차트 패턴 기반 CNN 모델 - 캔들스틱 특화 데이터 증강 버전
- 논문 기반 캔들스틱 차트 전용 데이터 증강
- 캔들 무작위 이동 (100%, 50%, 10%)
- 캔들 이동 크기 (0.003, 0.002, 0.001, 0.00025)
- 가우시안 노이즈 추가 (평균 0, 분산 0.01)
- ResNet50 Transfer Learning + Fine-tuning
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

# GPU 설정
print("🔧 GPU/CUDA 설정 중...")
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

np.random.seed(42)
tf.random.set_seed(42)


class CandlestickAugmentation:
    """캔들스틱 차트 전용 데이터 증강"""
    
    @staticmethod
    def add_gaussian_noise(image, mean=0, var=0.01):
        """
        가우시안 노이즈 추가
        Args:
            image: 입력 이미지 (numpy array, 0-1 범위)
            mean: 노이즈 평균
            var: 노이즈 분산
        """
        sigma = var ** 0.5
        gaussian = np.random.normal(mean, sigma, image.shape)
        noisy_image = image + gaussian
        return np.clip(noisy_image, 0, 1)
    
    @staticmethod
    def shift_candles_vertically(image, shift_ratio=0.5, shift_amount=0.003):
        """
        캔들스틱을 수직으로 무작위 이동
        Args:
            image: 입력 이미지 (numpy array, 0-1 범위)
            shift_ratio: 이동할 캔들 비율 (1.0=100%, 0.5=50%, 0.1=10%)
            shift_amount: 이동 크기 (0.003, 0.002, 0.001, 0.00025)
        """
        # 이미지 복사
        shifted_image = image.copy()
        
        # 이미지 높이
        height = image.shape[0]
        
        # 픽셀 단위로 이동량 계산
        pixel_shift = int(height * shift_amount)
        
        if pixel_shift > 0:
            # 상하 랜덤 이동
            direction = np.random.choice([-1, 1])
            shift_pixels = direction * pixel_shift
            
            # 이미지를 수직으로 이동
            if shift_pixels > 0:
                shifted_image[shift_pixels:, :] = image[:-shift_pixels, :]
                shifted_image[:shift_pixels, :] = 0
            elif shift_pixels < 0:
                shifted_image[:shift_pixels, :] = image[-shift_pixels:, :]
                shifted_image[shift_pixels:, :] = 0
        
        return shifted_image


class CandlestickDataGenerator(keras.utils.Sequence):
    """캔들스틱 차트 전용 데이터 제너레이터"""
    
    def __init__(self, directory, class_mode='binary', batch_size=32, 
                 img_size=(128, 128), shuffle=True, augment=True, seed=42):
        self.directory = directory
        self.class_mode = class_mode
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.augment = augment
        self.seed = seed
        
        # 클래스 및 파일 로드
        self.classes = ['down', 'up']
        self.class_indices = {name: idx for idx, name in enumerate(self.classes)}
        
        self.image_paths = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(directory, class_name)
            if os.path.exists(class_dir):
                files = [f for f in os.listdir(class_dir) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                for f in files:
                    self.image_paths.append(os.path.join(class_dir, f))
                    self.labels.append(self.class_indices[class_name])
        
        self.samples = len(self.image_paths)
        self.indexes = np.arange(self.samples)
        
        if self.shuffle:
            np.random.seed(self.seed)
            np.random.shuffle(self.indexes)
        
        # 증강 설정
        self.aug_configs = [
            {'shift_ratio': 1.0, 'shift_amount': 0.003},   # 100% 캔들, 0.003 이동
            {'shift_ratio': 1.0, 'shift_amount': 0.002},   # 100% 캔들, 0.002 이동
            {'shift_ratio': 0.5, 'shift_amount': 0.003},   # 50% 캔들, 0.003 이동
            {'shift_ratio': 0.5, 'shift_amount': 0.001},   # 50% 캔들, 0.001 이동
            {'shift_ratio': 0.1, 'shift_amount': 0.00025}, # 10% 캔들, 0.00025 이동
        ]
        
    def __len__(self):
        return int(np.ceil(self.samples / self.batch_size))
    
    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        batch_images = []
        batch_labels = []
        
        for idx in batch_indexes:
            # 이미지 로드
            img_path = self.image_paths[idx]
            img = Image.open(img_path).convert('RGB')
            img = img.resize(self.img_size)
            img_array = np.array(img) / 255.0
            
            # 증강 적용 (Training 시에만)
            if self.augment and np.random.random() > 0.3:  # 70% 확률로 증강
                aug_choice = np.random.choice(['candle_shift', 'gaussian_noise', 'both'])
                
                if aug_choice in ['candle_shift', 'both']:
                    # 캔들 이동 증강
                    config = random.choice(self.aug_configs)
                    img_array = CandlestickAugmentation.shift_candles_vertically(
                        img_array, 
                        shift_ratio=config['shift_ratio'],
                        shift_amount=config['shift_amount']
                    )
                
                if aug_choice in ['gaussian_noise', 'both']:
                    # 가우시안 노이즈 증강
                    img_array = CandlestickAugmentation.add_gaussian_noise(
                        img_array, mean=0, var=0.01
                    )
            
            batch_images.append(img_array)
            batch_labels.append(self.labels[idx])
        
        return np.array(batch_images), np.array(batch_labels)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


class CandlestickStockChartCNN:
    """캔들스틱 차트 전용 CNN 모델"""
    
    def __init__(self, data_dir, img_size=(128, 128), batch_size=32):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.base_model = None
        self.history_stage1 = None
        self.history_stage2 = None
        self.class_weights = None
        
    def explore_data(self):
        """데이터셋 탐색"""
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
        
        # 클래스 가중치
        total = len(up_files) + len(down_files)
        weight_down = total / (2 * len(down_files))
        weight_up = total / (2 * len(up_files))
        self.class_weights = {0: weight_down, 1: weight_up}
        
        print(f"⚖️  클래스 비율: Up={len(up_files)/total*100:.1f}%, Down={len(down_files)/total*100:.1f}%")
        print(f"🔧 클래스 가중치: Down={weight_down:.4f}, Up={weight_up:.4f}")
        print("="*60)
        
    def create_data_generators(self):
        """캔들스틱 전용 데이터 제너레이터 생성"""
        print("\n🔄 캔들스틱 전용 데이터 제너레이터 생성 중...")
        print("💡 논문 기반 캔들스틱 차트 증강:")
        print("   ✅ 캔들 무작위 이동 (100%, 50%, 10%)")
        print("   ✅ 이동 크기 (0.003, 0.002, 0.001, 0.00025)")
        print("   ✅ 가우시안 노이즈 (평균 0, 분산 0.01)")
        print("   ❌ 회전, 반전, 크기 조절 제거 (의미 없음)")
        
        # Training generator (증강 적용)
        self.train_generator = CandlestickDataGenerator(
            directory=self.data_dir,
            batch_size=self.batch_size,
            img_size=self.img_size,
            shuffle=True,
            augment=True,
            seed=42
        )
        
        # Validation generator (증강 미적용)
        self.val_generator = CandlestickDataGenerator(
            directory=self.data_dir,
            batch_size=self.batch_size,
            img_size=self.img_size,
            shuffle=False,
            augment=False,
            seed=42
        )
        
        # Train/Val 분리 (80/20)
        total_samples = self.train_generator.samples
        train_size = int(total_samples * 0.8)
        
        self.train_generator.indexes = self.train_generator.indexes[:train_size]
        self.train_generator.samples = train_size
        
        self.val_generator.indexes = self.val_generator.indexes[train_size:]
        self.val_generator.samples = total_samples - train_size
        
        print(f"✅ Training 샘플: {self.train_generator.samples:,}개 (증강 적용)")
        print(f"✅ Validation 샘플: {self.val_generator.samples:,}개 (증강 미적용)")
        
    def build_model_stage1(self):
        """Stage 1: Transfer Learning 모델"""
        print("\n🏗️  Stage 1: Transfer Learning 모델 구축...")
        
        base_model = ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=(self.img_size[0], self.img_size[1], 3),
            pooling='avg'
        )
        
        base_model.trainable = False
        
        inputs = keras.Input(shape=(self.img_size[0], self.img_size[1], 3))
        x = base_model(inputs, training=False)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = keras.Model(inputs, outputs)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        self.model = model
        self.base_model = base_model
        
        print(f"✅ Stage 1 모델 구축 완료")
        print(f"📊 총 파라미터: {model.count_params():,}개")
        
    def build_model_stage2(self, unfreeze_layers=60):
        """Stage 2: Fine-tuning"""
        print(f"\n🔓 Stage 2: Fine-tuning (마지막 {unfreeze_layers}개 레이어 Unfreeze)...")
        
        self.base_model.trainable = True
        frozen_layers = len(self.base_model.layers) - unfreeze_layers
        
        for i, layer in enumerate(self.base_model.layers):
            layer.trainable = (i >= frozen_layers)
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.00001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
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
        
        print(f"⏰ 시작: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
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
            print(f"⏰ 완료: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            return history
            
        except KeyboardInterrupt:
            print(f"\n⚠️ {stage_name} 중단됨")
            return None
            
    def evaluate(self):
        """모델 평가"""
        print("\n"+"="*60)
        print("📊 최종 모델 평가")
        print("="*60)
        
        # 예측
        y_pred_proba_list = []
        y_true_list = []
        
        for i in range(len(self.val_generator)):
            x_batch, y_batch = self.val_generator[i]
            y_pred_batch = self.model.predict(x_batch, verbose=0)
            y_pred_proba_list.extend(y_pred_batch.flatten())
            y_true_list.extend(y_batch)
        
        y_pred_proba = np.array(y_pred_proba_list)
        y_true = np.array(y_true_list)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # 평가 지표
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        print(f"\n✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"✅ F1-Score: {f1:.4f}")
        print(f"✅ ROC-AUC: {roc_auc:.4f}")
        
        print("\n📋 Classification Report:")
        print(classification_report(y_true, y_pred, target_names=['Down (0)', 'Up (1)']))
        
        cm = confusion_matrix(y_true, y_pred)
        
        # 투자 관점
        print("\n💰 투자 관점 평가:")
        tn, fp, fn, tp = cm.ravel()
        hit_ratio_up = tp / (tp + fp) if (tp + fp) > 0 else 0
        hit_ratio_down = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        print(f"   📈 상승 예측 적중률: {hit_ratio_up:.2%}")
        print(f"   📉 하락 예측 적중률: {hit_ratio_down:.2%}")
        
        return accuracy, f1, roc_auc, cm, y_true, y_pred_proba
        
    def plot_results(self, y_true, y_pred_proba):
        """결과 시각화"""
        print("\n📈 결과 시각화 중...")
        os.makedirs('results', exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Accuracy
        if self.history_stage1 and self.history_stage2:
            epochs_s1 = len(self.history_stage1.history['accuracy'])
            epochs_s2 = len(self.history_stage2.history['accuracy'])
            
            all_train_acc = self.history_stage1.history['accuracy'] + self.history_stage2.history['accuracy']
            all_val_acc = self.history_stage1.history['val_accuracy'] + self.history_stage2.history['val_accuracy']
            
            axes[0, 0].plot(all_train_acc, label='Train', linewidth=2)
            axes[0, 0].plot(all_val_acc, label='Validation', linewidth=2)
            axes[0, 0].axvline(x=epochs_s1, color='red', linestyle='--', label='Fine-tuning 시작')
            axes[0, 0].set_title('Accuracy (2-Stage)', fontweight='bold', fontsize=14)
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].legend()
            axes[0, 0].grid(alpha=0.3)
        
        # 2. AUC
        if self.history_stage1 and self.history_stage2:
            all_train_auc = self.history_stage1.history['auc'] + self.history_stage2.history['auc']
            all_val_auc = self.history_stage1.history['val_auc'] + self.history_stage2.history['val_auc']
            
            axes[0, 1].plot(all_train_auc, label='Train', linewidth=2)
            axes[0, 1].plot(all_val_auc, label='Validation', linewidth=2)
            axes[0, 1].axvline(x=epochs_s1, color='red', linestyle='--', label='Fine-tuning 시작')
            axes[0, 1].set_title('ROC-AUC (2-Stage)', fontweight='bold', fontsize=14)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('AUC')
            axes[0, 1].legend()
            axes[0, 1].grid(alpha=0.3)
        
        # 3. ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        axes[1, 0].plot(fpr, tpr, linewidth=3, label=f'ROC (AUC = {roc_auc:.4f})', color='blue')
        axes[1, 0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        axes[1, 0].fill_between(fpr, tpr, alpha=0.2)
        axes[1, 0].set_title('ROC Curve', fontweight='bold', fontsize=14)
        axes[1, 0].set_xlabel('False Positive Rate')
        axes[1, 0].set_ylabel('True Positive Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # 4. Confusion Matrix
        cm = confusion_matrix(y_true, (y_pred_proba > 0.5).astype(int))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1],
                   xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
        axes[1, 1].set_title('Confusion Matrix', fontweight='bold', fontsize=14)
        axes[1, 1].set_ylabel('True Label')
        axes[1, 1].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig('results/candlestick_aug_results.png', dpi=300, bbox_inches='tight')
        print("✅ 저장: results/candlestick_aug_results.png")
        plt.close()


def create_subset_if_needed(source_dir, target_dir, num_samples_per_class=2500):
    """서브셋 데이터셋 생성 (필요시)"""
    if os.path.exists(target_dir):
        print(f"\n✅ 서브셋 데이터셋 존재: {target_dir}")
        return target_dir
    
    print("\n"+"="*60)
    print("📦 5,000장 서브셋 생성 중...")
    print("="*60)
    
    os.makedirs(target_dir, exist_ok=True)
    random.seed(42)
    
    for class_name in ['up', 'down']:
        print(f"\n📂 클래스: {class_name}")
        
        source_class_dir = os.path.join(source_dir, class_name)
        target_class_dir = os.path.join(target_dir, class_name)
        os.makedirs(target_class_dir, exist_ok=True)
        
        all_files = [f for f in os.listdir(source_class_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"   📊 전체: {len(all_files):,}개")
        
        sampled_files = random.sample(all_files, min(num_samples_per_class, len(all_files)))
        print(f"   🎯 샘플링: {len(sampled_files):,}개")
        
        for i, filename in enumerate(sampled_files, 1):
            shutil.copy2(
                os.path.join(source_class_dir, filename),
                os.path.join(target_class_dir, filename)
            )
            if i % 500 == 0:
                print(f"   ⏳ 복사 중: {i}/{len(sampled_files)}")
        
        print(f"   ✅ 완료: {len(sampled_files):,}개")
    
    print("\n"+"="*60)
    print("✅ 서브셋 생성 완료!")
    print("="*60)
    
    return target_dir


def main():
    """메인 실행"""
    print("\n"+"="*60)
    print("🚀 캔들스틱 차트 전용 데이터 증강 모델")
    print("💡 논문 기반 증강 + ResNet50 Transfer Learning")
    print("="*60)
    
    # GPU 설정
    gpus = tf.config.experimental.list_physical_devices('GPU')
    batch_size = 128 if gpus else 64
    
    print(f"\n📊 설정:")
    print(f"   - 배치 크기: {batch_size} (GPU: {'사용' if gpus else '미사용'})")
    print(f"   - 이미지 크기: 128×128")
    print(f"   - 데이터셋: 5,000장")
    
    # 서브셋 생성
    data_dir = create_subset_if_needed(
        source_dir='dataset-2021',
        target_dir='dataset-subset-5k',
        num_samples_per_class=2500
    )
    
    # 모델 초기화
    model = CandlestickStockChartCNN(
        data_dir=data_dir,
        img_size=(128, 128),
        batch_size=batch_size
    )
    
    # 데이터 탐색
    model.explore_data()
    
    # 데이터 제너레이터 생성
    model.create_data_generators()
    
    # Stage 1: Transfer Learning
    print("\n"+"#"*60)
    print("📌 STAGE 1: Transfer Learning")
    print("#"*60)
    
    model.build_model_stage1()
    model.history_stage1 = model.train_stage(
        stage_name="Stage 1 - Transfer Learning",
        epochs=50,
        save_path='models/candlestick_aug/stage1_best.keras',
        patience=15
    )
    
    # Stage 2: Fine-tuning
    print("\n"+"#"*60)
    print("📌 STAGE 2: Fine-tuning")
    print("#"*60)
    
    model.build_model_stage2(unfreeze_layers=60)
    model.history_stage2 = model.train_stage(
        stage_name="Stage 2 - Fine-tuning",
        epochs=50,
        save_path='models/candlestick_aug/stage2_best.keras',
        patience=15
    )
    
    # 최종 평가
    accuracy, f1, roc_auc, cm, y_true, y_pred_proba = model.evaluate()
    
    # 결과 시각화
    model.plot_results(y_true, y_pred_proba)
    
    # 최종 요약
    print("\n"+"="*60)
    print("🎉 캔들스틱 증강 모델 학습 완료!")
    print("="*60)
    print(f"📊 최종 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"📊 F1-Score: {f1:.4f}")
    print(f"📊 ROC-AUC: {roc_auc:.4f}")
    print("="*60)
    
    # 비교
    print("\n"+"="*60)
    print("📊 모델 성능 비교")
    print("="*60)
    print("기존 모델:")
    print("   - Accuracy: 54.96% (단순 CNN)")
    print("   - Accuracy: 53.30% (ResNet50, 일반 증강)")
    print("\n캔들스틱 특화 증강:")
    print(f"   - Accuracy: {accuracy*100:.2f}%")
    print(f"   - F1-Score: {f1:.4f}")
    print(f"   - ROC-AUC: {roc_auc:.4f}")
    print("   - 논문 기반 캔들스틱 증강 적용")
    print("   - 2단계 Fine-tuning 완료")
    print("="*60)


if __name__ == '__main__':
    main()

