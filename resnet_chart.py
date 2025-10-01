import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler  # WeightedRandomSampler: 클래스 불균형 해결을 위한 샘플러
from torchvision import transforms, models

from PIL import Image
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, roc_curve
from collections import Counter  # 클래스별 샘플 수를 세기 위한 유틸리티
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class StockChartDataset(Dataset):
    def __init__(self, image_files, labels, transform=None):
        self.image_files = image_files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float)
# Define the directory containing the images
image_dir = 'C:/Users/Admin/workspace/project3/images/'
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]
labels = [int(f.split('-')[-1].split('.')[0]) for f in os.listdir(image_dir) if f.endswith('.jpg')]

# Split the dataset into training, validation, and testing sets
train_files, test_files, train_labels, test_labels = train_test_split(image_files, labels, test_size=0.2, random_state=42, stratify=labels)
train_files, val_files, train_labels, val_labels = train_test_split(train_files, train_labels, test_size=0.1, random_state=42, stratify=train_labels)

# Define transformations for data augmentation and normalization
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Create dataset instances
train_dataset = StockChartDataset(train_files, train_labels, transform=data_transforms['train'])
val_dataset = StockChartDataset(val_files, val_labels, transform=data_transforms['val'])
test_dataset = StockChartDataset(test_files, test_labels, transform=data_transforms['val'])

# Create DataLoader instances
batch_size = 32
# Note: DataLoaders will be created after computing class weights (see below)

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")
print(f"Number of testing samples: {len(test_dataset)}")

# ====================================================================
# 클래스 불균형 분석 및 대응
# ====================================================================
# 주식 데이터는 일반적으로 상승/하락 패턴의 분포가 불균형한 경우가 많습니다.
# 예: 상승 패턴 70%, 하락 패턴 30%와 같은 불균형이 존재할 수 있음
# 이러한 불균형을 해결하지 않으면 모델이 다수 클래스에 편향되어 학습됩니다.
# ====================================================================

print("\n=== Class Imbalance Analysis ===")

# Counter를 사용하여 각 클래스의 샘플 수를 계산
class_counts = Counter(train_labels)
print(f"Class 0 (Down): {class_counts[0]} samples")
print(f"Class 1 (Up): {class_counts[1]} samples")
total = sum(class_counts.values())
print(f"Class 0 ratio: {class_counts[0]/total*100:.2f}%")
print(f"Class 1 ratio: {class_counts[1]/total*100:.2f}%")

# ====================================================================
# pos_weight 계산 (BCEWithLogitsLoss용)
# ====================================================================
# pos_weight는 양성 클래스(Class 1)의 손실에 가중치를 부여합니다.
# 공식: pos_weight = (negative samples) / (positive samples)
# 예시: Class 0이 700개, Class 1이 300개라면 pos_weight = 700/300 = 2.33
# 이는 Class 1의 손실을 2.33배 더 중요하게 취급하여 불균형을 보정합니다.
# ====================================================================
pos_weight = torch.tensor([class_counts[0] / class_counts[1]], dtype=torch.float32).to(device)
print(f"\nCalculated pos_weight for BCEWithLogitsLoss: {pos_weight.item():.4f}")
print("→ Class 1(Up)의 손실이 {:.2f}배 더 중요하게 계산됩니다.".format(pos_weight.item()))

# ====================================================================
# WeightedRandomSampler를 위한 샘플 가중치 계산 (옵션)
# ====================================================================
# WeightedRandomSampler는 훈련 중 샘플링 확률을 조정하여 불균형을 해결합니다.
# 각 샘플의 가중치 = 1 / (해당 클래스의 샘플 수)
# 소수 클래스의 샘플이 더 자주 선택되도록 하여 균형을 맞춥니다.
# 
# 장점: 배치마다 클래스 비율이 균형있게 유지됨
# 단점: 소수 클래스 샘플이 중복 샘플링될 수 있음 (replacement=True)
# ====================================================================
class_weights = {0: 1.0 / class_counts[0], 1: 1.0 / class_counts[1]}
sample_weights = [class_weights[label] for label in train_labels]
sampler = WeightedRandomSampler(
    weights=sample_weights,           # 각 샘플의 가중치
    num_samples=len(sample_weights),  # 전체 샘플 수
    replacement=True                  # 중복 샘플링 허용 (균형을 위해 필요)
)
print(f"\nWeightedRandomSampler created with weights:")
print(f"  - Class 0 (Down): {class_weights[0]:.6f} (더 적은 클래스일수록 가중치가 높음)")
print(f"  - Class 1 (Up): {class_weights[1]:.6f}")
print("=" * 50)

# ====================================================================
# DataLoader 생성 - 클래스 불균형 대응 전략 선택
# ====================================================================
# 클래스 불균형 해결 방법 2가지:
# 
# 【옵션 1】 WeightedRandomSampler 사용
#   - 샘플링 단계에서 불균형 해결 (데이터 레벨)
#   - 소수 클래스를 더 자주 샘플링하여 배치 내 균형 유지
#   - sampler 사용 시 shuffle과 함께 사용 불가 (둘 중 하나만 선택)
#   - 사용법: train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
# 
# 【옵션 2】 pos_weight만 사용 (현재 설정)
#   - 손실 함수 레벨에서 불균형 해결
#   - 일반적인 shuffle 사용 가능
#   - 소수 클래스의 손실을 더 크게 계산하여 학습 시 더 중요하게 취급
#   - 더 간단하고 일반적으로 효과적
# ====================================================================

# 옵션 1: WeightedRandomSampler 사용 (아래 주석 해제)
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

# 옵션 2: 일반 shuffle 사용 + pos_weight 적용 (현재 설정 ✓)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("\n✓ DataLoader 설정: 일반 shuffle + pos_weight 방식 사용")
print("  (WeightedRandomSampler를 사용하려면 위의 옵션 1 주석을 해제하세요)")
print("  권장: pos_weight 방식이 더 간단하고 효과적입니다.")

# ====================================================================
# ResNet18 모델 로드 및 초기 설정
# ====================================================================
# 사전학습된 ResNet18을 로드하고 이진 분류를 위해 마지막 FC 레이어를 수정합니다.
# ====================================================================
model = models.resnet18(pretrained=True)

# 모든 파라미터 동결 (초기 상태)
for param in model.parameters():
    param.requires_grad = False

# 이진 분류를 위한 최종 FC 레이어 교체
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)

model = model.to(device)

print("\n" + "=" * 70)
print("📦 ResNet18 모델 로드 완료")
print("=" * 70)
print(f"✓ 사전학습된 가중치 사용: ImageNet")
print(f"✓ 최종 레이어: {num_ftrs} → 1 (Binary Classification)")
print(f"✓ 초기 상태: 모든 레이어 동결 (fc만 학습 가능)")
print("=" * 70)

# ====================================================================
# 손실 함수 정의 - pos_weight를 적용한 BCEWithLogitsLoss
# ====================================================================
# BCEWithLogitsLoss: 이진 분류를 위한 손실 함수
# - Sigmoid + Binary Cross Entropy를 결합하여 수치적으로 안정적
# - pos_weight 파라미터: 양성 클래스(Class 1)의 손실에 가중치 적용
#
# 작동 원리:
# - pos_weight가 없을 때: loss = -[y*log(p) + (1-y)*log(1-p)]
# - pos_weight 적용 시: loss = -[y*pos_weight*log(p) + (1-y)*log(1-p)]
# - Class 1의 손실이 pos_weight배 만큼 증폭되어 소수 클래스 학습 강화
#
# 예시: pos_weight=2.33이면 상승(Class 1) 예측 실패 시 손실이 2.33배로 계산
# ====================================================================
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
print(f"\n✓ BCEWithLogitsLoss initialized with pos_weight={pos_weight.item():.4f}")
print(f"  → 클래스 1(상승)의 오분류 패널티가 {pos_weight.item():.2f}배 증가합니다.")
print(f"  → 이를 통해 모델이 소수 클래스를 더 잘 학습하게 됩니다.")

# ====================================================================
# 단계적 파인튜닝을 위한 헬퍼 함수
# ====================================================================
# 전이 학습(Transfer Learning)에서 단계적 언프리즈는 매우 중요한 기법입니다.
# 
# 【왜 단계적으로 해야 하는가?】
# 1. 사전학습된 저수준 특징(edges, textures)은 대부분 유용함
# 2. 갑작스러운 전체 학습은 사전학습 가중치를 망칠 수 있음
# 3. 상위 레이어부터 점진적으로 해제하면 안정적인 학습 가능
# 
# 【차별적 학습률(Discriminative Learning Rate)】
# - 하위 레이어(layer1, layer2): 변경 최소화 (매우 작은 lr 또는 frozen)
# - 중간 레이어(layer3): 약간 조정 (작은 lr)
# - 상위 레이어(layer4): 도메인에 맞게 조정 (중간 lr)
# - FC 레이어: 완전히 새로 학습 (큰 lr)
# ====================================================================

def get_trainable_params_info(model):
    """
    현재 학습 가능한 파라미터 정보를 반환
    
    Returns:
        trainable_params: 학습 가능한 파라미터 수
        total_params: 전체 파라미터 수
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    return trainable_params, total_params

def unfreeze_layer(model, layer_name):
    """
    특정 레이어를 언프리즈하여 학습 가능하게 만듦
    
    Args:
        model: PyTorch 모델
        layer_name: 언프리즈할 레이어 이름 (예: 'layer4', 'layer3')
    """
    layer = getattr(model, layer_name)
    for param in layer.parameters():
        param.requires_grad = True
    print(f"   ✓ {layer_name} 언프리즈 완료")

# ====================================================================
# 임계값 튜닝 함수
# ====================================================================
# F1-score를 최대화하는 최적의 임계값을 찾는 함수
# 기본 0.5 대신 데이터에 맞는 최적 임계값을 찾아 성능을 개선합니다.
# ====================================================================
def find_optimal_threshold(model, dataloader, device):
    """
    검증 데이터에서 F1-score를 최대화하는 최적 임계값을 찾습니다.
    
    Args:
        model: 학습된 PyTorch 모델
        dataloader: 검증 데이터로더
        device: 연산 장치 (CPU/GPU)
    
    Returns:
        best_threshold: F1-score가 최대인 임계값
        best_f1: 최대 F1-score 값
    """
    model.eval()
    all_probs = []
    all_labels = []
    
    print("\n" + "=" * 70)
    print("🔍 최적 임계값(Threshold) 탐색 중...")
    print("=" * 70)
    
    # 검증 데이터에서 예측 확률 수집
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs).squeeze()
            probs = torch.sigmoid(outputs)  # logits를 확률로 변환 (0~1 사이)
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Precision-Recall 곡선에서 다양한 임계값과 그에 따른 precision, recall 계산
    precisions, recalls, thresholds = precision_recall_curve(all_labels, all_probs)
    
    # 각 임계값에 대한 F1-score 계산
    # F1 = 2 * (precision * recall) / (precision + recall)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)  # 0으로 나누기 방지
    
    # F1-score가 최대인 임계값 찾기
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]
    best_precision = precisions[best_idx]
    best_recall = recalls[best_idx]
    
    # 기본 0.5 임계값과 비교
    default_preds = (all_probs > 0.5).astype(int)
    default_f1 = f1_score(all_labels, default_preds)
    
    print(f"\n📊 임계값 탐색 결과:")
    print(f"   기본 임계값 (0.5000):")
    print(f"      - F1-score: {default_f1:.4f}")
    print(f"\n   ✨ 최적 임계값 ({best_threshold:.4f}):")
    print(f"      - F1-score: {best_f1:.4f} (↑ {(best_f1-default_f1)*100:+.2f}%p)")
    print(f"      - Precision: {best_precision:.4f}")
    print(f"      - Recall: {best_recall:.4f}")
    print(f"\n💡 최적 임계값을 사용하면 F1-score가 개선됩니다!")
    print("=" * 70)
    
    # 임계값 vs F1-score 그래프 저장
    plot_threshold_analysis(thresholds, f1_scores, precisions, recalls, best_threshold, best_f1)
    
    return best_threshold, best_f1

def plot_threshold_analysis(thresholds, f1_scores, precisions, recalls, best_threshold, best_f1):
    """임계값 분석 결과를 시각화"""
    plt.figure(figsize=(12, 5))
    
    # F1-score vs Threshold
    plt.subplot(1, 2, 1)
    plt.plot(thresholds, f1_scores[:-1], 'b-', linewidth=2, label='F1-score')
    plt.axvline(x=best_threshold, color='r', linestyle='--', linewidth=2, 
                label=f'Optimal: {best_threshold:.4f}')
    plt.axvline(x=0.5, color='gray', linestyle=':', linewidth=1.5, label='Default: 0.5')
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('F1-score', fontsize=12)
    plt.title('F1-score vs Threshold', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Precision & Recall vs Threshold
    plt.subplot(1, 2, 2)
    plt.plot(thresholds, precisions[:-1], 'g-', linewidth=2, label='Precision')
    plt.plot(thresholds, recalls[:-1], 'orange', linewidth=2, label='Recall')
    plt.axvline(x=best_threshold, color='r', linestyle='--', linewidth=2, 
                label=f'Optimal: {best_threshold:.4f}')
    plt.axvline(x=0.5, color='gray', linestyle=':', linewidth=1.5, label='Default: 0.5')
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Precision & Recall vs Threshold', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('threshold_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n📈 임계값 분석 그래프 저장: threshold_analysis.png")
    plt.close()

def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, 
                num_epochs=25, patience=5, min_delta=0.001):
    """
    EarlyStopping을 포함한 학습 함수
    
    Args:
        model: PyTorch 모델
        train_dataloader: 훈련 데이터로더
        val_dataloader: 검증 데이터로더
        criterion: 손실 함수
        optimizer: 옵티마이저
        scheduler: 학습률 스케줄러
        num_epochs: 최대 에포크 수
        patience: EarlyStopping patience (몇 epoch 동안 개선 없으면 중단)
        min_delta: 개선으로 간주할 최소 변화량
    
    Returns:
        model: 학습된 모델
        best_val_loss: 최고 검증 손실
        best_val_acc: 최고 검증 정확도
    """
    best_model_wts = model.state_dict()
    best_val_acc = 0.0
    best_val_loss = float('inf')
    
    # EarlyStopping을 위한 변수
    patience_counter = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_dataloader
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = val_dataloader

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs).squeeze()  # 모델 출력 (logits)
                    loss = criterion(outputs, labels)   # pos_weight가 적용된 손실 계산
                    
                    # 예측: logits를 sigmoid로 확률로 변환 후 0.5 기준으로 이진 분류
                    # BCEWithLogitsLoss는 내부적으로 sigmoid를 적용하므로 여기서 다시 적용
                    preds = torch.sigmoid(outputs) > 0.5

                    # Backward pass + optimize only in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Validation phase에서 best model 체크 및 EarlyStopping
            if phase == 'val':
                # 개선 체크 (loss 기준)
                if epoch_loss < (best_val_loss - min_delta):
                    print(f'   → 검증 손실 개선: {best_val_loss:.4f} → {epoch_loss:.4f}')
                    best_val_loss = epoch_loss
                    best_val_acc = epoch_acc
                    best_model_wts = model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    print(f'   → EarlyStopping 카운터: {patience_counter}/{patience}')
                
                # EarlyStopping 체크
                if patience_counter >= patience:
                    print(f'\n⚠️  EarlyStopping 발동: {patience} epoch 동안 개선 없음')
                    print(f'   최고 검증 손실: {best_val_loss:.4f}, 정확도: {best_val_acc:.4f}')
                    model.load_state_dict(best_model_wts)
                    return model, best_val_loss, best_val_acc

        # Step the scheduler after each epoch
        if scheduler is not None:
            scheduler.step()
        print()

    print(f'✓ 전체 {num_epochs} epoch 완료')
    print(f'  최고 검증 손실: {best_val_loss:.4f}, 정확도: {best_val_acc:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_val_loss, best_val_acc


# ====================================================================
# 단계적 파인튜닝 전략 (Progressive Unfreezing with Adaptive Training)
# ====================================================================
# 【전략 개요】
# Transfer Learning에서 사전학습된 가중치를 효과적으로 활용하기 위해
# 레이어를 단계적으로 언프리즈하면서 학습합니다.
#
# 【개선된 3단계 전략 - EarlyStopping & 조건부 실행】
# Step 1: FC 레이어만 학습 (base frozen)
#   - 목적: 새로운 태스크에 맞는 출력 레이어 초기화
#   - 학습률: 1e-3 (상대적으로 큰 값)
#   - Max Epochs: 20, EarlyStopping patience=5
#   - 항상 실행 (필수 단계)
#
# Step 2: Layer4 언프리즈 + Fine-tuning
#   - 목적: 상위 특징을 도메인에 맞게 미세 조정
#   - 학습률: layer4=1e-5 (작음), fc=1e-4 (중간)
#   - Max Epochs: 15, EarlyStopping patience=5
#   - 항상 실행
#   - Step 1 대비 성능 향상 체크 → Step 3 진행 여부 결정
#
# Step 3: Layer3까지 언프리즈 + Deep Fine-tuning (조건부)
#   - 목적: 중간 레벨 특징까지 도메인 특화
#   - 학습률: layer3=5e-6 (매우 작음), layer4=1e-5, fc=1e-4
#   - Max Epochs: 10, EarlyStopping patience=5
#   - 조건: Step 2에서 0.5% 이상 성능 향상 시에만 실행
#   - 향상 없으면 건너뛰기 (과적합 방지)
#
# 【학습률 설정 원칙】
# - 하위 레이어일수록 작은 학습률 (사전학습 지식 보존)
# - 상위 레이어일수록 큰 학습률 (태스크 특화)
# - FC 레이어는 가장 큰 학습률 (완전히 새로 학습)
#
# 【EarlyStopping 장점】
# - 과적합 방지: validation loss 모니터링
# - 효율성: 불필요한 epoch 생략 → 학습 시간 단축
# - 자동화: 각 단계마다 최적 epoch 자동 결정
#
# 【조건부 실행 장점】
# - 성능 향상 없으면 다음 단계 건너뛰기
# - 데이터셋 특성에 맞는 유연한 학습
# - 과적합 위험 최소화
# ====================================================================

print("\n" + "=" * 70)
print("🎯 단계적 파인튜닝 전략 시작")
print("=" * 70)
print("💡 전략: Progressive Unfreezing with Adaptive Training")
print("   - 사전학습 지식 보존 + 도메인 특화 학습")
print("   - EarlyStopping: 각 단계마다 최적 epoch 자동 결정")
print("   - 조건부 실행: 성능 향상 없으면 다음 단계 건너뛰기")
print("=" * 70)

# ====================================================================
# Step 1: FC 레이어만 학습
# ====================================================================
# 먼저 FC 레이어만 학습하여 새로운 분류 태스크에 적응시킵니다.
# Base network는 frozen 상태로 유지하여 사전학습된 특징 추출기를 보존합니다.
# ====================================================================
print("\n【Step 1】 FC 레이어만 학습 (Base Frozen)")
print("-" * 70)

trainable, total = get_trainable_params_info(model)
print(f"📊 학습 가능 파라미터: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
print(f"   → FC 레이어만 학습 가능 (나머지는 frozen)")

# FC 레이어만 학습하는 optimizer
# 학습률 1e-3: FC 레이어는 랜덤 초기화되었으므로 큰 학습률 사용
optimizer_step1 = optim.Adam(model.fc.parameters(), lr=1e-3)
scheduler_step1 = optim.lr_scheduler.StepLR(optimizer_step1, step_size=7, gamma=0.1)

print(f"⚙️  Optimizer: Adam (lr=1e-3)")
print(f"   → FC 레이어 초기화를 위해 상대적으로 큰 학습률 사용")
print(f"📅 Max Epochs: 20 (EarlyStopping patience=5)")
print("-" * 70)

# Step 1 학습 (patience=5로 조기 종료 가능)
trained_model, step1_loss, step1_acc = train_model(
    model, train_dataloader, val_dataloader, criterion, 
    optimizer_step1, scheduler_step1, num_epochs=20, patience=5
)

print(f"\n✓ Step 1 완료 - Val Loss: {step1_loss:.4f}, Val Acc: {step1_acc:.4f}")

# ====================================================================
# Step 2: Layer4 언프리즈 + Fine-tuning (성능 향상 시에만 진행)
# ====================================================================
# Step 1에서 FC 레이어가 초기화되었으므로, 이제 상위 레이어(layer4)를
# 언프리즈하여 도메인에 맞게 미세 조정합니다.
# 
# ResNet18 구조: conv1 → layer1 → layer2 → layer3 → layer4 → fc
#                (low-level)                          (high-level)
# 
# Layer4는 고수준 특징(high-level features)을 추출하므로,
# 주식 차트 도메인에 맞게 조정하면 성능 향상이 기대됩니다.
# 
# 차별적 학습률 사용:
# - layer4: 1e-5 (사전학습 가중치 크게 변경하지 않음)
# - fc: 1e-4 (계속 학습, 하지만 Step 1보다 작게)
# ====================================================================
print("\n" + "=" * 70)
print("【Step 2】 Layer4 언프리즈 + Fine-tuning")
print("-" * 70)

# Layer4 언프리즈
unfreeze_layer(trained_model, 'layer4')

trainable, total = get_trainable_params_info(trained_model)
print(f"📊 학습 가능 파라미터: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
print(f"   → layer4 + fc 학습 가능")

# 차별적 학습률 적용 (Discriminative Learning Rate)
# - layer4: 작은 learning rate (1e-5) - 사전학습 가중치 보존
# - fc: 중간 learning rate (1e-4) - 계속 학습
optimizer_step2 = optim.Adam([
    {'params': trained_model.layer4.parameters(), 'lr': 1e-5},
    {'params': trained_model.fc.parameters(), 'lr': 1e-4}
])
scheduler_step2 = optim.lr_scheduler.StepLR(optimizer_step2, step_size=5, gamma=0.5)

print(f"⚙️  Optimizer: Adam (Discriminative LR)")
print(f"   - layer4: lr=1e-5 (사전학습 가중치 미세 조정)")
print(f"   - fc: lr=1e-4 (Step 1의 1/10, 안정적 학습)")
print(f"📅 Max Epochs: 15 (EarlyStopping patience=5)")
print(f"📉 Scheduler: StepLR (step=5, gamma=0.5)")
print("-" * 70)

# Step 2 학습
trained_model, step2_loss, step2_acc = train_model(
    trained_model, train_dataloader, val_dataloader, criterion,
    optimizer_step2, scheduler_step2, num_epochs=15, patience=5
)

print(f"\n✓ Step 2 완료 - Val Loss: {step2_loss:.4f}, Val Acc: {step2_acc:.4f}")

# ====================================================================
# 성능 향상 체크 - Step 3 진행 여부 결정
# ====================================================================
# Step 2에서 layer4를 언프리즈한 결과를 평가합니다.
# 개선이 충분하지 않으면 추가 언프리즈는 과적합만 유발할 수 있으므로 건너뜁니다.
# 
# 개선 기준:
# - improvement_threshold = 0.5% (조정 가능)
# - 이 기준은 데이터셋 크기와 복잡도에 따라 달라질 수 있음
# - 작은 데이터셋: 더 보수적 (예: 1%)
# - 큰 데이터셋: 더 공격적 (예: 0.1%)
# ====================================================================
improvement_threshold = 0.005  # 0.5% 이상 개선 필요
loss_improvement = (step1_loss - step2_loss) / step1_loss

if loss_improvement < improvement_threshold:
    print(f"\n⚠️  Step 2 성능 향상 미미: {loss_improvement*100:.2f}% < {improvement_threshold*100:.1f}%")
    print(f"   → Step 3 (layer3 언프리즈)를 건너뜁니다.")
    print(f"   → 추가 언프리즈는 과적합만 유발할 가능성이 높습니다.")
    skip_step3 = True
else:
    print(f"\n✓ Step 2 성능 향상 확인: Loss {loss_improvement*100:.2f}% 개선")
    print(f"   → Step 3 진행합니다.")
    skip_step3 = False

# ====================================================================
# Step 3: Layer3까지 언프리즈 + Deep Fine-tuning (조건부 실행)
# ====================================================================
# 중간 레벨 특징(layer3)까지 도메인에 맞게 조정합니다.
# 
# 주의사항:
# 1. Layer3는 중간 레벨 특징을 추출 (shapes, patterns)
# 2. 너무 많이 학습하면 과적합 위험 → 짧은 epoch 사용
# 3. 매우 작은 학습률 사용 (5e-6) → 사전학습 지식 최대한 보존
# 
# 학습률 계층 구조:
# - layer3: 5e-6 (가장 작음, 최소 변경)
# - layer4: 1e-5 (작음, 미세 조정)
# - fc: 1e-4 (상대적으로 큼, 태스크 특화)
# 
# 하위 레이어(layer1, layer2)는 frozen 유지:
# - 일반적인 저수준 특징(edges, colors)은 대부분 유용
# - 과적합 방지 및 학습 시간 단축
# ====================================================================

if not skip_step3:
    print("\n" + "=" * 70)
    print("【Step 3】 Layer3까지 언프리즈 + Deep Fine-tuning")
    print("-" * 70)

    # Layer3 언프리즈
    unfreeze_layer(trained_model, 'layer3')

    trainable, total = get_trainable_params_info(trained_model)
    print(f"📊 학습 가능 파라미터: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
    print(f"   → layer3 + layer4 + fc 학습 가능")
    print(f"   → layer1, layer2는 frozen 유지 (저수준 특징 보존)")

    # 차별적 학습률 적용 (3단계 계층 구조)
    # - layer3: 매우 작은 learning rate (5e-6) - 최소 변경
    # - layer4: 작은 learning rate (1e-5) - 미세 조정
    # - fc: 중간 learning rate (1e-4) - 태스크 특화
    optimizer_step3 = optim.Adam([
        {'params': trained_model.layer3.parameters(), 'lr': 5e-6},
        {'params': trained_model.layer4.parameters(), 'lr': 1e-5},
        {'params': trained_model.fc.parameters(), 'lr': 1e-4}
    ])
    scheduler_step3 = optim.lr_scheduler.StepLR(optimizer_step3, step_size=3, gamma=0.5)

    print(f"⚙️  Optimizer: Adam (3-tier Discriminative LR)")
    print(f"   - layer3: lr=5e-6 (사전학습 가중치 최대 보존)")
    print(f"   - layer4: lr=1e-5 (미세 조정 지속)")
    print(f"   - fc: lr=1e-4 (태스크 특화 학습)")
    print(f"📅 Max Epochs: 10 (EarlyStopping patience=5)")
    print(f"📉 Scheduler: StepLR (step=3, gamma=0.5)")
    print(f"⚠️  주의: 너무 많이 학습하면 과적합 가능성 증가")
    print("-" * 70)

    # Step 3 학습
    trained_model, step3_loss, step3_acc = train_model(
        trained_model, train_dataloader, val_dataloader, criterion,
        optimizer_step3, scheduler_step3, num_epochs=10, patience=5
    )

    print(f"\n✓ Step 3 완료 - Val Loss: {step3_loss:.4f}, Val Acc: {step3_acc:.4f}")
    
    # 성능 체크
    step3_improvement = (step2_loss - step3_loss) / step2_loss
    if step3_improvement < improvement_threshold:
        print(f"\n⚠️  Step 3 성능 향상 미미: {step3_improvement*100:.2f}%")
        print(f"   → Step 2 모델이 더 좋을 수 있습니다.")
    else:
        print(f"\n✓ Step 3 성능 향상 확인: Loss {step3_improvement*100:.2f}% 개선")
    
    final_loss = step3_loss
    final_acc = step3_acc
else:
    print("\n" + "=" * 70)
    print("⏭️  Step 3 건너뜀 (Step 2 성능 향상 미미)")
    print("=" * 70)
    final_loss = step2_loss
    final_acc = step2_acc

print("\n" + "=" * 70)
print("✅ 단계적 파인튜닝 완료!")
print("=" * 70)
print("📊 최종 학습 결과:")
print(f"   - 최종 Val Loss: {final_loss:.4f}")
print(f"   - 최종 Val Acc: {final_acc:.4f}")
trainable, total = get_trainable_params_info(trained_model)
print(f"   - 학습된 파라미터: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
print(f"   - Frozen 파라미터: {total - trainable:,} ({(total-trainable)/total*100:.2f}%)")
print("=" * 70)

# ====================================================================
# 최적 임계값 탐색 (검증 세트 사용)
# ====================================================================
# 훈련이 완료된 후, 검증 세트에서 F1-score를 최대화하는 임계값을 찾습니다.
# 이 임계값은 테스트 세트 평가에 사용됩니다.
# ====================================================================
optimal_threshold, optimal_val_f1 = find_optimal_threshold(trained_model, val_dataloader, device)

# ====================================================================
# 테스트 세트 평가 (최적 임계값 사용)
# ====================================================================
# 클래스 불균형이 해결된 모델의 성능을 테스트 데이터로 평가합니다.
# F1-Score는 정밀도(Precision)와 재현율(Recall)의 조화평균으로,
# 불균형 데이터에서 Accuracy보다 더 신뢰할 수 있는 지표입니다.
# 
# 여기서는 검증 세트에서 찾은 최적 임계값을 사용합니다.
# ====================================================================
print("\n=== Evaluating on Test Set ===")
trained_model.eval()
all_probs = []  # 확률 저장 (임계값 적용을 위해)
all_labels = []

with torch.no_grad():
    for inputs, labels in test_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = trained_model(inputs).squeeze()
        probs = torch.sigmoid(outputs)  # logits를 확률로 변환
        
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_probs = np.array(all_probs)
all_labels = np.array(all_labels)

# ====================================================================
# 평가 지표 계산 - 기본 임계값 0.5 vs 최적 임계값
# ====================================================================
# 1. 기본 임계값 0.5 사용
default_preds = (all_probs > 0.5).astype(int)
default_accuracy = accuracy_score(all_labels, default_preds)
default_f1 = f1_score(all_labels, default_preds)

# 2. 최적 임계값 사용 (검증 세트에서 찾은 값)
optimal_preds = (all_probs > optimal_threshold).astype(int)
optimal_accuracy = accuracy_score(all_labels, optimal_preds)
optimal_f1 = f1_score(all_labels, optimal_preds)

print(f"\n" + "=" * 70)
print("📊 테스트 결과 비교 (클래스 불균형 대응 + 임계값 튜닝)")
print("=" * 70)

print(f"\n【기본 임계값 0.5 사용】")
print(f"   - Test Accuracy: {default_accuracy:.4f} ({default_accuracy*100:.2f}%)")
print(f"   - Test F1-Score: {default_f1:.4f}")

print(f"\n【최적 임계값 {optimal_threshold:.4f} 사용】 ✨")
print(f"   - Test Accuracy: {optimal_accuracy:.4f} ({optimal_accuracy*100:.2f}%)")
print(f"   - Test F1-Score: {optimal_f1:.4f}")

print(f"\n📈 성능 개선:")
print(f"   - Accuracy: {(optimal_accuracy - default_accuracy)*100:+.2f}%p")
print(f"   - F1-Score: {(optimal_f1 - default_f1)*100:+.2f}%p")

print(f"\n💡 최적 임계값을 사용하면 F1-score가 개선됩니다!")
print("=" * 70)

# 최종적으로 사용할 지표 (최적 임계값 기준)
test_accuracy = optimal_accuracy
test_f1 = optimal_f1

# ====================================================================
# 모델 및 최적 임계값 저장
# ====================================================================
# 모델 가중치와 함께 최적 임계값도 저장하여 추론 시 동일한 임계값을 사용합니다.
# ====================================================================
# 모델 가중치 저장
torch.save(trained_model.state_dict(), 'resnet18_stock_chart_improved.pth')

# 최적 임계값 저장 (추론 시 사용)
threshold_info = {
    'optimal_threshold': optimal_threshold,
    'validation_f1': optimal_val_f1,
    'test_accuracy': test_accuracy,
    'test_f1': test_f1
}
torch.save(threshold_info, 'optimal_threshold.pth')

print("\n✅ 모델 및 설정 저장 완료:")
print(f"   - 모델 가중치: resnet18_stock_chart_improved.pth")
print(f"   - 최적 임계값: optimal_threshold.pth")
print(f"   - 저장된 임계값: {optimal_threshold:.4f}")
print(f"\n💡 추론 시 이 임계값을 로드하여 사용하세요:")
print(f"   threshold_info = torch.load('optimal_threshold.pth')")
print(f"   threshold = threshold_info['optimal_threshold']")

# ====================================================================
# [참고] 추론 시 모델 및 임계값 로드 예제 코드
# ====================================================================
"""
# 모델 로드
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)
model.load_state_dict(torch.load('resnet18_stock_chart_improved.pth'))
model.to(device)
model.eval()

# 최적 임계값 로드
threshold_info = torch.load('optimal_threshold.pth')
optimal_threshold = threshold_info['optimal_threshold']

# 추론
with torch.no_grad():
    outputs = model(input_tensor).squeeze()
    probs = torch.sigmoid(outputs)
    predictions = (probs > optimal_threshold).int()  # 최적 임계값 사용
    
    # 0: 하락(Down), 1: 상승(Up)
    print(f"예측: {'상승' if predictions.item() == 1 else '하락'}")
    print(f"확률: {probs.item():.4f}")
    print(f"임계값: {optimal_threshold:.4f}")
"""
print("\n" + "=" * 70)
print("🎉 모델 훈련 및 임계값 튜닝 완료!")
print("=" * 70)
