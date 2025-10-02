"""
앙상블 모델 구축 V2
- ResNet50 + EfficientNet + DenseNet 조합
- 가중 평균 앙상블
- 성능 향상 기대
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, EfficientNetB0, DenseNet121
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# Windows 콘솔 인코딩 설정
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'ignore')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'ignore')

class EnsembleModelV2:
    """앙상블 모델 V2"""
    
    def __init__(self, data_dir='dataset-future-5day', img_size=(224, 224)):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = 32
        self.models = {}
        self.weights = {}
        
    def create_data_generators(self):
        """데이터 생성기"""
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            brightness_range=[0.95, 1.05],
            fill_mode='nearest'
        )
        
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            f'{self.data_dir}/train',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False,
            color_mode='rgb'
        )
        
        val_generator = val_datagen.flow_from_directory(
            f'{self.data_dir}/val',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False,
            color_mode='rgb'
        )
        
        return train_generator, val_generator
    
    def create_resnet50_model(self):
        """ResNet50 모델 생성"""
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'auc']
        )
        
        return model
    
    def create_efficientnet_model(self):
        """EfficientNet-B3 모델 생성 (채널 문제 해결)"""
        try:
            # EfficientNet-B0 사용 (더 안정적)
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.img_size, 3)
            )
            
            model = models.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dropout(0.5),
                layers.Dense(512, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer=Adam(0.0001),
                loss='binary_crossentropy',
                metrics=['accuracy', 'auc']
            )
            
            return model
        except Exception as e:
            print(f"EfficientNet 생성 실패: {e}")
            return None
    
    def create_densenet_model(self):
        """DenseNet-121 모델 생성"""
        try:
            base_model = DenseNet121(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.img_size, 3)
            )
            
            model = models.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dropout(0.5),
                layers.Dense(512, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer=Adam(0.0001),
                loss='binary_crossentropy',
                metrics=['accuracy', 'auc']
            )
            
            return model
        except Exception as e:
            print(f"DenseNet 생성 실패: {e}")
            return None
    
    def train_individual_models(self, train_generator, val_generator, epochs=20):
        """개별 모델 학습 (GPU 강제 사용)"""
        print("\n" + "="*60)
        print("개별 모델 학습 (GPU 강제 사용)")
        print("="*60)
        
        os.makedirs('models/ensemble_v2', exist_ok=True)
        
        # GPU 강제 사용
        with tf.device('/GPU:0'):
            # 모델 생성
            models_dict = {
                'resnet50': self.create_resnet50_model(),
                'efficientnet': self.create_efficientnet_model(),
                'densenet': self.create_densenet_model()
            }
        
        # None 제거
        models_dict = {k: v for k, v in models_dict.items() if v is not None}
        
        for name, model in models_dict.items():
            if model is None:
                continue
                
            print(f"\n{name} 모델 학습 중...")
            
            callbacks = [
                EarlyStopping(
                    monitor='val_auc',
                    patience=5,
                    restore_best_weights=True,
                    verbose=1
                ),
                ModelCheckpoint(
                    f'models/ensemble_v2/{name}_best.keras',
                    monitor='val_auc',
                    save_best_only=True,
                    verbose=1
                )
            ]
            
            try:
                # GPU 강제 사용
                with tf.device('/GPU:0'):
                    history = model.fit(
                        train_generator,
                        epochs=epochs,
                        validation_data=val_generator,
                        callbacks=callbacks,
                        verbose=1
                    )
                
                # 모델 성능 평가 (GPU 강제 사용)
                with tf.device('/GPU:0'):
                    val_loss, val_acc, val_auc = model.evaluate(val_generator, verbose=0)
                self.weights[name] = val_auc  # AUC를 가중치로 사용
                
                print(f"{name} 완료 - Val AUC: {val_auc:.4f}")
                
            except Exception as e:
                print(f"{name} 학습 실패: {e}")
                continue
        
        # 가중치 정규화
        if self.weights:
            total_weight = sum(self.weights.values())
            self.weights = {k: v/total_weight for k, v in self.weights.items()}
            print(f"\n모델 가중치: {self.weights}")
        
        self.models = models_dict
        return models_dict
    
    def ensemble_predict(self, val_generator):
        """앙상블 예측"""
        print("\n" + "="*60)
        print("앙상블 예측")
        print("="*60)
        
        if not self.models:
            print("학습된 모델이 없습니다.")
            return None, None
        
        predictions = {}
        
        # 각 모델의 예측 (GPU 강제 사용)
        for name, model in self.models.items():
            if model is None:
                continue
                
            try:
                with tf.device('/GPU:0'):
                    pred = model.predict(val_generator, verbose=0)
                predictions[name] = pred.flatten()
                print(f"{name} 예측 완료")
            except Exception as e:
                print(f"{name} 예측 실패: {e}")
                continue
        
        if not predictions:
            print("예측 가능한 모델이 없습니다.")
            return None, None
        
        # 가중 평균 앙상블
        ensemble_pred = np.zeros_like(list(predictions.values())[0])
        total_weight = 0
        
        for name, pred in predictions.items():
            weight = self.weights.get(name, 1.0)
            ensemble_pred += weight * pred
            total_weight += weight
        
        ensemble_pred /= total_weight
        
        # 실제 레이블
        y_true = val_generator.classes
        
        # 성능 평가
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(y_true, ensemble_pred)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        y_pred_binary = (ensemble_pred > optimal_threshold).astype(int)
        
        accuracy = accuracy_score(y_true, y_pred_binary)
        auc = roc_auc_score(y_true, ensemble_pred)
        f1 = f1_score(y_true, y_pred_binary)
        
        print(f"\n앙상블 성능:")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"AUC: {auc:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"최적 Threshold: {optimal_threshold:.4f}")
        
        # 개별 모델 성능 비교
        print(f"\n개별 모델 성능:")
        for name, pred in predictions.items():
            pred_binary = (pred > optimal_threshold).astype(int)
            acc = accuracy_score(y_true, pred_binary)
            auc_score = roc_auc_score(y_true, pred)
            f1_score_val = f1_score(y_true, pred_binary)
            print(f"{name}: Acc={acc:.4f}, AUC={auc_score:.4f}, F1={f1_score_val:.4f}")
        
        return ensemble_pred, y_true
    
    def run_ensemble(self):
        """앙상블 모델 실행"""
        print("\n" + "="*80)
        print("앙상블 모델 V2 실행")
        print("ResNet50 + EfficientNet + DenseNet")
        print("="*80)
        
        # 데이터 생성기
        train_generator, val_generator = self.create_data_generators()
        
        # 개별 모델 학습
        models = self.train_individual_models(train_generator, val_generator)
        
        # 앙상블 예측
        ensemble_pred, y_true = self.ensemble_predict(val_generator)
        
        if ensemble_pred is not None and y_true is not None:
            print("\n" + "="*80)
            print("앙상블 모델 V2 완료!")
            print("="*80)
            
            # 최종 성능
            from sklearn.metrics import roc_curve
            fpr, tpr, thresholds = roc_curve(y_true, ensemble_pred)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            
            y_pred_binary = (ensemble_pred > optimal_threshold).astype(int)
            final_accuracy = accuracy_score(y_true, y_pred_binary)
            
            print(f"최종 Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
            print(f"이전 모델 대비: {final_accuracy*100 - 54.05:+.2f}%p")
            
            if final_accuracy > 0.54:
                print("✅ 성능 향상!")
            else:
                print("❌ 성능 향상 없음")
        
        return ensemble_pred, y_true

def main():
    """메인 실행 함수"""
    ensemble = EnsembleModelV2()
    ensemble_pred, y_true = ensemble.run_ensemble()
    return ensemble_pred, y_true

if __name__ == "__main__":
    main()


