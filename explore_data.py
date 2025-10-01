"""
데이터셋 탐색 및 샘플 시각화
"""

import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter

# Windows 콘솔 인코딩 설정
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'ignore')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'ignore')

def explore_dataset(data_dir='dataset-2021'):
    """데이터셋 탐색"""
    print("\n" + "=" * 60)
    print("📊 데이터셋 탐색 및 분석")
    print("=" * 60)
    
    up_dir = os.path.join(data_dir, 'up')
    down_dir = os.path.join(data_dir, 'down')
    
    up_files = [f for f in os.listdir(up_dir) if f.endswith('.jpg')]
    down_files = [f for f in os.listdir(down_dir) if f.endswith('.jpg')]
    
    print(f"\n✅ 상승(Up) 이미지: {len(up_files):,}개")
    print(f"❌ 하락(Down) 이미지: {len(down_files):,}개")
    print(f"📈 총 이미지: {len(up_files) + len(down_files):,}개")
    
    total = len(up_files) + len(down_files)
    print(f"\n⚖️  클래스 비율:")
    print(f"   Up:   {len(up_files)/total*100:.2f}% ({len(up_files):,}개)")
    print(f"   Down: {len(down_files)/total*100:.2f}% ({len(down_files):,}개)")
    
    # 이미지 크기 분석
    print(f"\n📐 이미지 크기 분석 (샘플 100개)...")
    sample_files = random.sample(up_files[:100], min(50, len(up_files))) + \
                   random.sample(down_files[:100], min(50, len(down_files)))
    
    sizes = []
    for f in sample_files:
        if f in up_files:
            path = os.path.join(up_dir, f)
        else:
            path = os.path.join(down_dir, f)
        
        try:
            img = Image.open(path)
            sizes.append(img.size)
        except:
            continue
    
    size_counter = Counter(sizes)
    print(f"   가장 흔한 크기: {size_counter.most_common(3)}")
    
    # 파일명 패턴 분석
    print(f"\n📝 파일명 패턴:")
    sample_up = up_files[0]
    sample_down = down_files[0]
    print(f"   Up 샘플:   {sample_up}")
    print(f"   Down 샘플: {sample_down}")
    
    # 종목코드 추출
    stock_codes_up = set([f.split('-')[0] for f in up_files[:1000]])
    stock_codes_down = set([f.split('-')[0] for f in down_files[:1000]])
    
    print(f"\n🏢 종목 분석 (샘플 1000개):")
    print(f"   Up 종목 수: {len(stock_codes_up)}개")
    print(f"   Down 종목 수: {len(stock_codes_down)}개")
    print(f"   공통 종목: {len(stock_codes_up & stock_codes_down)}개")
    
    print("\n" + "=" * 60)
    
    return up_files, down_files


def visualize_samples(data_dir='dataset-2021', n_samples=8, save_path='results/sample_charts.png'):
    """샘플 이미지 시각화"""
    print(f"\n🎨 샘플 이미지 시각화 ({n_samples}개)...")
    
    up_dir = os.path.join(data_dir, 'up')
    down_dir = os.path.join(data_dir, 'down')
    
    up_files = [f for f in os.listdir(up_dir) if f.endswith('.jpg')]
    down_files = [f for f in os.listdir(down_dir) if f.endswith('.jpg')]
    
    # 랜덤 샘플 선택
    up_samples = random.sample(up_files, n_samples // 2)
    down_samples = random.sample(down_files, n_samples // 2)
    
    # 시각화
    fig, axes = plt.subplots(2, n_samples // 2, figsize=(20, 8))
    
    # Up 샘플
    for i, filename in enumerate(up_samples):
        img_path = os.path.join(up_dir, filename)
        img = Image.open(img_path)
        axes[0, i].imshow(img)
        axes[0, i].axis('off')
        axes[0, i].set_title(f'Up\n{filename[:20]}...', 
                            fontsize=10, color='red', fontweight='bold')
    
    # Down 샘플
    for i, filename in enumerate(down_samples):
        img_path = os.path.join(down_dir, filename)
        img = Image.open(img_path)
        axes[1, i].imshow(img)
        axes[1, i].axis('off')
        axes[1, i].set_title(f'Down\n{filename[:20]}...', 
                            fontsize=10, color='blue', fontweight='bold')
    
    plt.suptitle('주식 차트 샘플 (상승 vs 하락)', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # 저장
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 샘플 시각화 저장: {save_path}")
    
    plt.show()
    plt.close()


def analyze_image_statistics(data_dir='dataset-2021', n_samples=100):
    """이미지 통계 분석 (평균, 표준편차 등)"""
    print(f"\n📊 이미지 통계 분석 ({n_samples}개 샘플)...")
    
    up_dir = os.path.join(data_dir, 'up')
    down_dir = os.path.join(data_dir, 'down')
    
    up_files = [f for f in os.listdir(up_dir) if f.endswith('.jpg')]
    down_files = [f for f in os.listdir(down_dir) if f.endswith('.jpg')]
    
    # 샘플 선택
    sample_up = random.sample(up_files, min(n_samples // 2, len(up_files)))
    sample_down = random.sample(down_files, min(n_samples // 2, len(down_files)))
    
    def get_stats(files, dir_path):
        means = []
        stds = []
        
        for f in files:
            try:
                img = Image.open(os.path.join(dir_path, f))
                img_array = np.array(img) / 255.0
                means.append(img_array.mean())
                stds.append(img_array.std())
            except:
                continue
        
        return np.mean(means), np.mean(stds)
    
    up_mean, up_std = get_stats(sample_up, up_dir)
    down_mean, down_std = get_stats(sample_down, down_dir)
    
    print(f"\n📈 Up 이미지 통계:")
    print(f"   평균 픽셀 값: {up_mean:.4f}")
    print(f"   표준편차: {up_std:.4f}")
    
    print(f"\n📉 Down 이미지 통계:")
    print(f"   평균 픽셀 값: {down_mean:.4f}")
    print(f"   표준편차: {down_std:.4f}")
    
    print(f"\n💡 차이:")
    print(f"   평균 차이: {abs(up_mean - down_mean):.4f}")
    print(f"   표준편차 차이: {abs(up_std - down_std):.4f}")


def create_data_distribution_plot(data_dir='dataset-2021', save_path='results/data_distribution.png'):
    """데이터 분포 시각화"""
    print(f"\n📊 데이터 분포 시각화...")
    
    up_dir = os.path.join(data_dir, 'up')
    down_dir = os.path.join(data_dir, 'down')
    
    up_count = len([f for f in os.listdir(up_dir) if f.endswith('.jpg')])
    down_count = len([f for f in os.listdir(down_dir) if f.endswith('.jpg')])
    
    # 바 차트
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 개수 비교
    categories = ['Up (상승)', 'Down (하락)']
    counts = [up_count, down_count]
    colors = ['red', 'blue']
    
    bars = ax1.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax1.set_ylabel('이미지 개수', fontsize=12, fontweight='bold')
    ax1.set_title('클래스별 이미지 개수', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # 값 표시
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}개\n({count/(up_count+down_count)*100:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 비율 파이 차트
    ax2.pie(counts, labels=categories, colors=colors, autopct='%1.1f%%',
           startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'},
           wedgeprops={'edgecolor': 'black', 'linewidth': 2})
    ax2.set_title('클래스 비율', fontsize=14, fontweight='bold')
    
    plt.suptitle(f'데이터셋 분포 (총 {up_count+down_count:,}개)', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # 저장
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 분포 시각화 저장: {save_path}")
    
    plt.show()
    plt.close()


def main():
    """메인 실행 함수"""
    print("\n" + "=" * 60)
    print("🔍 주식 차트 데이터셋 탐색 및 분석")
    print("=" * 60)
    
    # 1. 데이터셋 탐색
    up_files, down_files = explore_dataset()
    
    # 2. 샘플 이미지 시각화
    visualize_samples(n_samples=8)
    
    # 3. 이미지 통계 분석
    analyze_image_statistics(n_samples=100)
    
    # 4. 데이터 분포 시각화
    create_data_distribution_plot()
    
    print("\n" + "=" * 60)
    print("✅ 데이터 탐색 완료!")
    print("=" * 60)


if __name__ == '__main__':
    main()
