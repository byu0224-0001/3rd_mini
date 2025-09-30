"""

주식 차트 패턴 기반 CNN 모델

- 차트 이미지를 학습하여 다음날 상승/하락 예측

- Dataset: dataset-2021/up, dataset-2021/down

"""



import os

import sys

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from PIL import Image

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers, models

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

    # GPU 디바이스 확인

    gpus = tf.config.experimental.list_physical_devices('GPU')

    print(f"🔍 감지된 GPU: {len(gpus)}개")

    

    if gpus:

        # GPU 메모리 증가 설정

        for gpu in gpus:

            tf.config.experimental.set_memory_growth(gpu, True)

        

        # GPU 사용 설정

        tf.config.experimental.set_visible_devices(gpus, 'GPU')

        print(f"✅ GPU 사용 가능: {len(gpus)}개")

        print("✅ GPU 메모리 증가 설정 완료")

        print("🚀 GPU 가속 학습 모드로 진행합니다!")

        

        # GPU 정보 출력

        for i, gpu in enumerate(gpus):

            print(f"   GPU {i}: {gpu.name}")

    else:

        print("⚠️ GPU를 찾을 수 없습니다. CPU로 학습합니다.")

        print("💡 CPU 최적화 설정을 적용합니다.")

        

        # CPU 최적화 설정

        tf.config.threading.set_inter_op_parallelism_threads(0)

        tf.config.threading.set_intra_op_parallelism_threads(0)

        

except Exception as e:

    print(f"⚠️ GPU 설정 오류: {e}")

    print("🔄 CPU 모드로 전환합니다.")

    

    # CPU 최적화 설정

    tf.config.threading.set_inter_op_parallelism_threads(0)

    tf.config.threading.set_intra_op_parallelism_threads(0)



# 랜덤 시드 고정

np.random.seed(42)

tf.random.set_seed(42)



class StockChartCNN:

    """주식 차트 CNN 모델 클래스"""

    

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

        

    def explore_data(self):

        """데이터셋 탐색"""

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

        print(f"⚖️  클래스 비율: Up={len(up_files)/(len(up_files)+len(down_files))*100:.1f}%, "

              f"Down={len(down_files)/(len(up_files)+len(down_files))*100:.1f}%")

        

        # 샘플 이미지 크기 확인

        sample_img_path = os.path.join(up_dir, up_files[0])

        sample_img = Image.open(sample_img_path)

        print(f"📐 샘플 이미지 크기: {sample_img.size}")

        print(f"🎨 이미지 모드: {sample_img.mode}")

        print("=" * 60)

        

        return len(up_files), len(down_files)

    

    def create_data_generators(self, validation_split=0.2):

        """데이터 제너레이터 생성 (Data Augmentation 포함)"""

        print("\n🔄 데이터 제너레이터 생성 중...")

        

        # Training 데이터 증강

        train_datagen = ImageDataGenerator(

            rescale=1./255,

            validation_split=validation_split,

            rotation_range=10,

            width_shift_range=0.1,

            height_shift_range=0.1,

            horizontal_flip=True,

            zoom_range=0.1,

            fill_mode='nearest'

        )

        

        # Validation 데이터 (증강 없음)

        val_datagen = ImageDataGenerator(

            rescale=1./255,

            validation_split=validation_split

        )

        

        # Training 제너레이터

        self.train_generator = train_datagen.flow_from_directory(

            self.data_dir,

            target_size=self.img_size,

            batch_size=self.batch_size,

            class_mode='binary',

            subset='training',

            shuffle=True,

            seed=42

        )

        

        # Validation 제너레이터

        self.val_generator = val_datagen.flow_from_directory(

            self.data_dir,

            target_size=self.img_size,

            batch_size=self.batch_size,

            class_mode='binary',

            subset='validation',

            shuffle=False,

            seed=42

        )

        

        print(f"✅ Training 샘플: {self.train_generator.samples:,}개")

        print(f"✅ Validation 샘플: {self.val_generator.samples:,}개")

        print(f"📋 클래스 매핑: {self.train_generator.class_indices}")

        

        return self.train_generator, self.val_generator

    

    def build_model(self):

        """CNN 모델 구축"""

        print("\n🏗️  CNN 모델 구축 중...")

        

        model = models.Sequential([

            # 첫 번째 Conv Block

            layers.Conv2D(32, (3, 3), activation='relu', padding='same', 

                         input_shape=(self.img_size[0], self.img_size[1], 3)),

            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),

            layers.BatchNormalization(),

            layers.MaxPooling2D((2, 2)),

            layers.Dropout(0.25),

            

            # 두 번째 Conv Block

            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),

            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),

            layers.BatchNormalization(),

            layers.MaxPooling2D((2, 2)),

            layers.Dropout(0.25),

            

            # 세 번째 Conv Block

            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),

            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),

            layers.BatchNormalization(),

            layers.MaxPooling2D((2, 2)),

            layers.Dropout(0.25),

            

            # 네 번째 Conv Block

            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),

            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),

            layers.BatchNormalization(),

            layers.MaxPooling2D((2, 2)),

            layers.Dropout(0.25),

            

            # Fully Connected Layers

            layers.Flatten(),

            layers.Dense(512, activation='relu'),

            layers.BatchNormalization(),

            layers.Dropout(0.5),

            layers.Dense(256, activation='relu'),

            layers.BatchNormalization(),

            layers.Dropout(0.5),

            

            # Output Layer

            layers.Dense(1, activation='sigmoid')

        ])

        

        # 모델 컴파일

        model.compile(

            optimizer=keras.optimizers.Adam(learning_rate=0.001),

            loss='binary_crossentropy',

            metrics=['accuracy', 

                    keras.metrics.Precision(name='precision'),

                    keras.metrics.Recall(name='recall'),

                    keras.metrics.AUC(name='auc')]

        )

        

        self.model = model

        

        print("✅ 모델 구축 완료")

        print(f"📊 총 파라미터: {model.count_params():,}개")

        

        return model

    

    def train(self, epochs=50, save_path='models/best_model.h5'):

        """모델 학습"""

        print("\n🚀 모델 학습 시작")

        print("=" * 60)

        

        # 전체 데이터셋 크기 계산

        total_samples = self.train_generator.samples + self.val_generator.samples

        print(f"📊 전체 데이터셋: {total_samples:,}개")

        print(f"   - Training: {self.train_generator.samples:,}개")

        print(f"   - Validation: {self.val_generator.samples:,}개")

        print(f"   - 예상 학습 시간: {epochs} 에포크")

        print("=" * 60)

        

        # 콜백 설정

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        

        # 진행률 표시를 위한 커스텀 콜백

        class ProgressCallback(keras.callbacks.Callback):

            def __init__(self, total_epochs, total_samples):

                self.total_epochs = total_epochs

                self.total_samples = total_samples

                self.current_epoch = 0

                

            def on_epoch_begin(self, epoch, logs=None):

                self.current_epoch = epoch + 1

                progress = (self.current_epoch / self.total_epochs) * 100

                

                print(f"\n📈 Epoch {self.current_epoch}/{self.total_epochs} 시작")

                print(f"   진행률: {progress:.1f}% ({self.current_epoch}/{self.total_epochs})")

                

                # 25% 단위로 표시

                if progress >= 25 and progress < 50:

                    print("   🟡 25% 완료 - 학습이 진행 중입니다...")

                elif progress >= 50 and progress < 75:

                    print("   🟠 50% 완료 - 절반을 넘었습니다!")

                elif progress >= 75 and progress < 100:

                    print("   🔴 75% 완료 - 거의 다 왔습니다!")

                elif progress >= 100:

                    print("   🎉 100% 완료!")

                

            def on_epoch_end(self, epoch, logs=None):

                if logs:

                    train_acc = logs.get('accuracy', 0)

                    val_acc = logs.get('val_accuracy', 0)

                    train_loss = logs.get('loss', 0)

                    val_loss = logs.get('val_loss', 0)

                    

                    print(f"   📊 결과:")

                    print(f"      Training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

                    print(f"      Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

                    

                    # 성능 개선 표시

                    if epoch > 0:

                        prev_val_acc = getattr(self, 'prev_val_acc', 0)

                        if val_acc > prev_val_acc:

                            print(f"      📈 성능 개선! (+{val_acc - prev_val_acc:.4f})")

                        elif val_acc < prev_acc:

                            print(f"      📉 성능 하락 (-{prev_val_acc - val_acc:.4f})")

                    

                    self.prev_val_acc = val_acc

                

                print("   " + "-" * 50)

        

        callbacks = [

            ProgressCallback(epochs, total_samples),

            EarlyStopping(

                monitor='val_loss',

                patience=15,  # 더 긴 patience

                restore_best_weights=True,

                verbose=1

            ),

            ModelCheckpoint(

                save_path,

                monitor='val_accuracy',

                save_best_only=True,

                verbose=1

            ),

            ReduceLROnPlateau(

                monitor='val_loss',

                factor=0.5,

                patience=7,  # 더 긴 patience

                min_lr=1e-7,

                verbose=1

            )

        ]

        

        print(f"\n⏰ 학습 시작 시간: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

        print("💡 중간에 중단하려면 Ctrl+C를 누르세요")

        print("=" * 60)

        

        # 학습

        try:

            self.history = self.model.fit(

                self.train_generator,

                epochs=epochs,

                validation_data=self.val_generator,

                callbacks=callbacks,

                verbose=0  # 커스텀 콜백에서 출력

            )

            

            print("\n" + "=" * 60)

            print("✅ 학습 완료!")

            print(f"⏰ 완료 시간: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

            print("=" * 60)

            

        except KeyboardInterrupt:

            print("\n\n⚠️ 사용자에 의해 학습이 중단되었습니다.")

            print("💾 현재까지의 모델이 저장되었습니다.")

            return self.history

        

        return self.history

    

    def evaluate(self):

        """모델 평가"""

        print("\n📊 모델 평가")

        print("=" * 60)

        

        # Validation 데이터로 예측

        val_steps = len(self.val_generator)

        y_pred_proba = self.model.predict(self.val_generator, steps=val_steps)

        y_pred = (y_pred_proba > 0.5).astype(int).flatten()

        y_true = self.val_generator.classes

        

        # 평가 지표 계산

        accuracy = accuracy_score(y_true, y_pred)

        print(f"\n✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

        

        print("\n📋 Classification Report:")

        print(classification_report(y_true, y_pred, 

                                   target_names=['Down (0)', 'Up (1)']))

        

        # Confusion Matrix

        cm = confusion_matrix(y_true, y_pred)

        

        return accuracy, cm, y_true, y_pred

    

    def plot_training_history(self, save_path='results/training_history.png'):

        """학습 과정 시각화"""

        print("\n📈 학습 과정 시각화 중...")

        

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        

        # Accuracy

        axes[0, 0].plot(self.history.history['accuracy'], label='Train Accuracy')

        axes[0, 0].plot(self.history.history['val_accuracy'], label='Val Accuracy')

        axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')

        axes[0, 0].set_xlabel('Epoch')

        axes[0, 0].set_ylabel('Accuracy')

        axes[0, 0].legend()

        axes[0, 0].grid(True, alpha=0.3)

        

        # Loss

        axes[0, 1].plot(self.history.history['loss'], label='Train Loss')

        axes[0, 1].plot(self.history.history['val_loss'], label='Val Loss')

        axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')

        axes[0, 1].set_xlabel('Epoch')

        axes[0, 1].set_ylabel('Loss')

        axes[0, 1].legend()

        axes[0, 1].grid(True, alpha=0.3)

        

        # Precision

        axes[1, 0].plot(self.history.history['precision'], label='Train Precision')

        axes[1, 0].plot(self.history.history['val_precision'], label='Val Precision')

        axes[1, 0].set_title('Model Precision', fontsize=14, fontweight='bold')

        axes[1, 0].set_xlabel('Epoch')

        axes[1, 0].set_ylabel('Precision')

        axes[1, 0].legend()

        axes[1, 0].grid(True, alpha=0.3)

        

        # Recall

        axes[1, 1].plot(self.history.history['recall'], label='Train Recall')

        axes[1, 1].plot(self.history.history['val_recall'], label='Val Recall')

        axes[1, 1].set_title('Model Recall', fontsize=14, fontweight='bold')

        axes[1, 1].set_xlabel('Epoch')

        axes[1, 1].set_ylabel('Recall')

        axes[1, 1].legend()

        axes[1, 1].grid(True, alpha=0.3)

        

        plt.tight_layout()

        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        print(f"✅ 저장 완료: {save_path}")

        plt.close()

    

    def plot_confusion_matrix(self, cm, save_path='results/confusion_matrix.png'):

        """Confusion Matrix 시각화"""

        print("\n📊 Confusion Matrix 시각화 중...")

        

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        

        plt.figure(figsize=(10, 8))

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,

                   xticklabels=['Down (0)', 'Up (1)'],

                   yticklabels=['Down (0)', 'Up (1)'],

                   annot_kws={'size': 16})

        plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)

        plt.ylabel('True Label', fontsize=12)

        plt.xlabel('Predicted Label', fontsize=12)

        

        # 정확도 표시

        accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()

        plt.text(1, -0.3, f'Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)',

                ha='center', fontsize=14, fontweight='bold')

        

        plt.tight_layout()

        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        print(f"✅ 저장 완료: {save_path}")

        plt.close()

    

    def predict_image(self, image_path):

        """단일 이미지 예측"""

        # 이미지 로드 및 전처리

        img = Image.open(image_path)

        img = img.resize(self.img_size)

        img_array = np.array(img) / 255.0

        img_array = np.expand_dims(img_array, axis=0)

        

        # 예측

        prediction = self.model.predict(img_array, verbose=0)[0][0]

        

        result = {

            'prediction': 'Up (상승)' if prediction > 0.5 else 'Down (하락)',

            'probability': prediction if prediction > 0.5 else 1 - prediction,

            'up_prob': prediction,

            'down_prob': 1 - prediction

        }

        

        return result

    

    def save_model_summary(self, save_path='results/model_summary.txt'):

        """모델 구조 저장"""

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        

        with open(save_path, 'w', encoding='utf-8') as f:

            self.model.summary(print_fn=lambda x: f.write(x + '\n'))

        

        print(f"✅ 모델 구조 저장: {save_path}")





def main():

    """메인 실행 함수"""

    print("\n" + "=" * 60)

    print("🚀 주식 차트 패턴 기반 CNN 모델 프로젝트")

    print("🔥 전체 데이터셋 학습 모드 (1,015,729개 이미지)")

    print("=" * 60)

    

    # 1. 모델 객체 생성 (GPU 최적화된 배치 크기)

    # RTX 4060 8GB VRAM에 최적화된 배치 크기

    gpus = tf.config.experimental.list_physical_devices('GPU')

    batch_size = 128 if gpus else 64  # GPU: 128 (메모리 효율적), CPU: 64

    

    stock_cnn = StockChartCNN(

        data_dir='dataset-2021',

        img_size=(100, 100),

        batch_size=batch_size

    )

    

    print(f"📊 배치 크기: {batch_size} (GPU: {'사용' if gpus else '미사용'})")

    if gpus:

        print("💡 RTX 4060 8GB VRAM에 최적화된 설정입니다.")

    

    # 2. 데이터 탐색

    stock_cnn.explore_data()

    

    # 3. 데이터 제너레이터 생성

    stock_cnn.create_data_generators(validation_split=0.2)

    

    # 4. 모델 구축

    stock_cnn.build_model()

    stock_cnn.model.summary()

    

    # 5. 모델 구조 저장

    stock_cnn.save_model_summary()

    

    # 6. 학습 시작 전 확인

    print("\n" + "=" * 60)

    print("⚠️  전체 데이터셋 학습을 시작합니다!")

    print("=" * 60)

    print(f"📊 데이터셋 크기: {stock_cnn.train_generator.samples + stock_cnn.val_generator.samples:,}개")

    print(f"⏱️  예상 소요 시간: 5-10시간 (GPU) / 50-100시간 (CPU)")

    print(f"💾 메모리 사용량: 약 8-16GB RAM")

    print(f"🖥️  GPU 권장: NVIDIA GPU (CUDA)")

    print("=" * 60)

    

    print("\n🚀 자동으로 학습을 시작합니다...")

    

    # 7. 모델 학습 (에포크 수 단축)

    print("\n🚀 학습 시작!")

    stock_cnn.train(epochs=50, save_path='models/best_stock_chart_model.h5')

    

    # 8. 모델 평가

    accuracy, cm, y_true, y_pred = stock_cnn.evaluate()

    

    # 9. 결과 시각화

    stock_cnn.plot_training_history()

    stock_cnn.plot_confusion_matrix(cm)

    

    # 10. 샘플 예측 테스트

    print("\n" + "=" * 60)

    print("🔮 샘플 예측 테스트")

    print("=" * 60)

    

    # Up 샘플

    up_sample = os.path.join('dataset-2021/up', os.listdir('dataset-2021/up')[0])

    result = stock_cnn.predict_image(up_sample)

    print(f"\n📈 Up 샘플 예측:")

    print(f"   파일: {os.path.basename(up_sample)}")

    print(f"   예측: {result['prediction']}")

    print(f"   확률: {result['probability']:.2%}")

    print(f"   Up 확률: {result['up_prob']:.2%}, Down 확률: {result['down_prob']:.2%}")

    

    # Down 샘플

    down_sample = os.path.join('dataset-2021/down', os.listdir('dataset-2021/down')[0])

    result = stock_cnn.predict_image(down_sample)

    print(f"\n📉 Down 샘플 예측:")

    print(f"   파일: {os.path.basename(down_sample)}")

    print(f"   예측: {result['prediction']}")

    print(f"   확률: {result['probability']:.2%}")

    print(f"   Up 확률: {result['up_prob']:.2%}, Down 확률: {result['down_prob']:.2%}")

    

    print("\n" + "=" * 60)

    print("🎉 전체 학습 완료!")

    print(f"📊 최종 정확도: {accuracy:.4f} ({accuracy*100:.2f}%)")

    print("=" * 60)





if __name__ == '__main__':

    main()
