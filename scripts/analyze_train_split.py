"""
📊 Comprehensive Train Split Analysis Script
Tạo biểu đồ và thống kê cho report

Output:
- figures/train_class_distribution.png  - Phân bố số video mỗi lớp
- figures/train_frame_histogram.png     - Histogram số frame
- figures/train_val_split.png           - Pie chart train/val split
- figures/train_top_bottom_classes.png  - Top/Bottom classes
- figures/train_frame_boxplot.png       - Boxplot frame per class
- figures/train_analysis_summary.json   - Thống kê tổng hợp
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
import json
from tqdm import tqdm
import pickle
from collections import Counter

# =============================================================================
# CẤU HÌNH
# =============================================================================
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

DATASET_PATH = Path("dataset/dataset/train")
LABEL_MAPPING_PATH = Path("dataset/dataset/label_mapping.pkl")
FIGURES_PATH = Path("figures")
FIGURES_PATH.mkdir(exist_ok=True)

# Cùng seed với notebook
TRAIN_SPLIT_RATIO = 0.8
RANDOM_SEED = 42
NUM_CLASSES = 100

# Matplotlib config
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def count_video_frames(video_path):
    """Đếm số lượng khung hình trong video"""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            frame_count = 0
            while True:
                ret, _ = cap.read()
                if not ret:
                    break
                frame_count += 1
        cap.release()
        return frame_count
    except Exception as e:
        return None


def get_video_duration(video_path):
    """Lấy duration (seconds) của video"""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if fps > 0:
            return frame_count / fps
        return None
    except:
        return None


def collect_all_data(dataset_path):
    """Thu thập tất cả video cùng với metadata"""
    data = []
    
    for label_folder in sorted(os.listdir(dataset_path)):
        path = dataset_path / label_folder
        if path.is_dir():
            for video_file in os.listdir(path):
                if video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_path = path / video_file
                    data.append({
                        'path': video_path,
                        'label': label_folder,
                        'filename': video_file
                    })
    
    return data


def create_train_val_split(data, train_ratio=0.8, seed=42):
    """Chia train/val giống notebook"""
    np.random.seed(seed)
    total = len(data)
    train_size = int(train_ratio * total)
    
    indices = list(range(total))
    np.random.shuffle(indices)
    
    train_indices = set(indices[:train_size])
    
    for i, item in enumerate(data):
        item['split'] = 'train' if i in train_indices else 'val'
    
    return data


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================
def analyze_class_distribution(data):
    """Phân tích phân bố class"""
    train_data = [d for d in data if d['split'] == 'train']
    val_data = [d for d in data if d['split'] == 'val']
    
    train_counter = Counter([d['label'] for d in train_data])
    val_counter = Counter([d['label'] for d in val_data])
    
    return train_counter, val_counter


def analyze_frames(data, desc="Analyzing"):
    """Phân tích frame counts"""
    frame_data = []
    
    for item in tqdm(data, desc=desc):
        frame_count = count_video_frames(item['path'])
        if frame_count and frame_count > 0:
            item['frame_count'] = frame_count
            frame_data.append(item)
    
    return frame_data


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================
def plot_class_distribution(train_counter, val_counter, output_path):
    """Biểu đồ phân bố class - stacked bar"""
    # Sắp xếp theo tổng số videos
    all_labels = sorted(train_counter.keys(), 
                        key=lambda x: train_counter[x] + val_counter.get(x, 0), 
                        reverse=True)
    
    fig, ax = plt.subplots(figsize=(20, 8))
    
    x = np.arange(len(all_labels))
    train_counts = [train_counter[l] for l in all_labels]
    val_counts = [val_counter.get(l, 0) for l in all_labels]
    
    width = 0.8
    bars1 = ax.bar(x, train_counts, width, label='Train (80%)', color='#5B9BD5', edgecolor='white')
    bars2 = ax.bar(x, val_counts, width, bottom=train_counts, label='Val (20%)', color='#FF6B6B', edgecolor='white')
    
    ax.set_xlabel('Lớp', fontsize=12, fontweight='bold')
    ax.set_ylabel('Số lượng video', fontsize=12, fontweight='bold')
    ax.set_title('Phân bố số lượng video trên mỗi lớp (Train/Val Split)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, rotation=90, fontsize=6)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved: {output_path}")
    plt.close()
    
    return all_labels


def plot_frame_histogram(train_data, output_path):
    """Histogram số frame"""
    frame_counts = [d['frame_count'] for d in train_data if 'frame_count' in d]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    n, bins, patches = ax.hist(frame_counts, bins=30, edgecolor='black', 
                               color='#66BB6A', alpha=0.8)
    
    # Stats
    mean_val = np.mean(frame_counts)
    median_val = np.median(frame_counts)
    
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
    ax.axvline(median_val, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_val:.1f}')
    
    ax.set_xlabel('Số lượng khung hình', fontsize=12, fontweight='bold')
    ax.set_ylabel('Số lượng video', fontsize=12, fontweight='bold')
    ax.set_title('Phân bố số lượng khung hình - Tập Train (80%)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Info box
    info = f"N = {len(frame_counts)}\n"
    info += f"Min = {min(frame_counts)}\n"
    info += f"Max = {max(frame_counts)}\n"
    info += f"Std = {np.std(frame_counts):.1f}"
    ax.text(0.98, 0.78, info, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved: {output_path}")
    plt.close()


def plot_train_val_pie(train_count, val_count, output_path):
    """Pie chart train/val split"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    sizes = [train_count, val_count]
    labels = [f'Train\n{train_count} videos\n({train_count/(train_count+val_count)*100:.1f}%)',
              f'Val\n{val_count} videos\n({val_count/(train_count+val_count)*100:.1f}%)']
    colors = ['#5B9BD5', '#FF6B6B']
    explode = (0.02, 0.02)
    
    ax.pie(sizes, labels=labels, colors=colors, explode=explode,
           autopct='', shadow=True, startangle=90,
           textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax.set_title('Tỷ lệ Train/Val Split', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved: {output_path}")
    plt.close()


def plot_top_bottom_classes(train_counter, output_path, top_n=10):
    """Top và Bottom classes"""
    sorted_classes = sorted(train_counter.items(), key=lambda x: x[1], reverse=True)
    
    top_classes = sorted_classes[:top_n]
    bottom_classes = sorted_classes[-top_n:][::-1]  # Reverse để hiển thị từ thấp lên
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Top classes
    ax1 = axes[0]
    labels1 = [c[0] for c in top_classes]
    values1 = [c[1] for c in top_classes]
    bars1 = ax1.barh(labels1, values1, color='#5B9BD5', edgecolor='black')
    ax1.set_xlabel('Số lượng video')
    ax1.set_title(f'Top {top_n} lớp nhiều video nhất', fontweight='bold')
    ax1.invert_yaxis()
    for bar, val in zip(bars1, values1):
        ax1.text(val + 0.5, bar.get_y() + bar.get_height()/2, str(val),
                va='center', fontsize=9)
    
    # Bottom classes
    ax2 = axes[1]
    labels2 = [c[0] for c in bottom_classes]
    values2 = [c[1] for c in bottom_classes]
    bars2 = ax2.barh(labels2, values2, color='#FF6B6B', edgecolor='black')
    ax2.set_xlabel('Số lượng video')
    ax2.set_title(f'Bottom {top_n} lớp ít video nhất', fontweight='bold')
    ax2.invert_yaxis()
    for bar, val in zip(bars2, values2):
        ax2.text(val + 0.5, bar.get_y() + bar.get_height()/2, str(val),
                va='center', fontsize=9)
    
    plt.suptitle('So sánh số lượng video giữa các lớp (Train Set)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved: {output_path}")
    plt.close()


def plot_frame_boxplot_by_class(data, output_path, top_n=20):
    """Boxplot frame per class (top N classes)"""
    # Group by class
    class_frames = {}
    for item in data:
        if 'frame_count' in item:
            label = item['label']
            if label not in class_frames:
                class_frames[label] = []
            class_frames[label].append(item['frame_count'])
    
    # Sort by mean frame count
    sorted_classes = sorted(class_frames.items(), 
                           key=lambda x: np.mean(x[1]), reverse=True)[:top_n]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    labels = [c[0] for c in sorted_classes]
    data_to_plot = [c[1] for c in sorted_classes]
    
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(data_to_plot)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Lớp', fontsize=12, fontweight='bold')
    ax.set_ylabel('Số lượng khung hình', fontsize=12, fontweight='bold')
    ax.set_title(f'Boxplot số khung hình theo lớp (Top {top_n} classes)', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved: {output_path}")
    plt.close()


def save_summary_json(data, train_counter, val_counter, output_path):
    """Lưu thống kê tổng hợp"""
    train_data = [d for d in data if d['split'] == 'train' and 'frame_count' in d]
    val_data = [d for d in data if d['split'] == 'val' and 'frame_count' in d]
    
    train_frames = [d['frame_count'] for d in train_data]
    val_frames = [d['frame_count'] for d in val_data]
    
    summary = {
        'dataset_info': {
            'total_videos': len(data),
            'total_classes': len(set(d['label'] for d in data)),
            'train_ratio': TRAIN_SPLIT_RATIO,
            'random_seed': RANDOM_SEED
        },
        'train_split': {
            'num_videos': len(train_data),
            'num_classes': len(train_counter),
            'videos_per_class': {
                'min': min(train_counter.values()),
                'max': max(train_counter.values()),
                'mean': np.mean(list(train_counter.values())),
                'std': np.std(list(train_counter.values()))
            },
            'frame_statistics': {
                'min': int(min(train_frames)) if train_frames else 0,
                'max': int(max(train_frames)) if train_frames else 0,
                'mean': float(np.mean(train_frames)) if train_frames else 0,
                'median': float(np.median(train_frames)) if train_frames else 0,
                'std': float(np.std(train_frames)) if train_frames else 0,
                'q1': float(np.percentile(train_frames, 25)) if train_frames else 0,
                'q3': float(np.percentile(train_frames, 75)) if train_frames else 0
            }
        },
        'val_split': {
            'num_videos': len(val_data),
            'num_classes': len(val_counter),
            'frame_statistics': {
                'min': int(min(val_frames)) if val_frames else 0,
                'max': int(max(val_frames)) if val_frames else 0,
                'mean': float(np.mean(val_frames)) if val_frames else 0,
                'median': float(np.median(val_frames)) if val_frames else 0,
            }
        },
        'class_distribution': {
            label: {'train': train_counter.get(label, 0), 'val': val_counter.get(label, 0)}
            for label in sorted(set(train_counter.keys()) | set(val_counter.keys()))
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Saved: {output_path}")
    return summary


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("📊 COMPREHENSIVE TRAIN SPLIT ANALYSIS")
    print("=" * 70)
    
    if not DATASET_PATH.exists():
        print(f"❌ Dataset not found: {DATASET_PATH}")
        return
    
    # 1. Collect data
    print("\n📂 Collecting videos...")
    data = collect_all_data(DATASET_PATH)
    print(f"Found {len(data)} videos")
    
    # 2. Create split
    print(f"\n✂️ Creating {TRAIN_SPLIT_RATIO*100:.0f}/{(1-TRAIN_SPLIT_RATIO)*100:.0f} split (seed={RANDOM_SEED})...")
    data = create_train_val_split(data, TRAIN_SPLIT_RATIO, RANDOM_SEED)
    
    train_count = sum(1 for d in data if d['split'] == 'train')
    val_count = sum(1 for d in data if d['split'] == 'val')
    print(f"Train: {train_count} | Val: {val_count}")
    
    # 3. Analyze class distribution
    print("\n📊 Analyzing class distribution...")
    train_counter, val_counter = analyze_class_distribution(data)
    
    # 4. Analyze frames (train only)
    print("\n🎬 Analyzing frame counts (train set)...")
    train_data = [d for d in data if d['split'] == 'train']
    train_data = analyze_frames(train_data, desc="Counting frames")
    
    # Update main data với frame counts
    train_frame_dict = {str(d['path']): d['frame_count'] for d in train_data if 'frame_count' in d}
    for d in data:
        if str(d['path']) in train_frame_dict:
            d['frame_count'] = train_frame_dict[str(d['path'])]
    
    # 5. Create visualizations
    print("\n📈 Creating visualizations...")
    
    plot_class_distribution(
        train_counter, val_counter,
        FIGURES_PATH / "train_class_distribution.png"
    )
    
    plot_frame_histogram(
        train_data,
        FIGURES_PATH / "train_frame_histogram.png"
    )
    
    plot_train_val_pie(
        train_count, val_count,
        FIGURES_PATH / "train_val_split.png"
    )
    
    plot_top_bottom_classes(
        train_counter,
        FIGURES_PATH / "train_top_bottom_classes.png"
    )
    
    plot_frame_boxplot_by_class(
        train_data,
        FIGURES_PATH / "train_frame_boxplot.png"
    )
    
    # 6. Save summary
    print("\n📝 Saving summary...")
    summary = save_summary_json(
        data, train_counter, val_counter,
        FIGURES_PATH / "train_analysis_summary.json"
    )
    
    # 7. Print summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print(f"Total videos: {summary['dataset_info']['total_videos']}")
    print(f"Total classes: {summary['dataset_info']['total_classes']}")
    print(f"\nTrain Set ({TRAIN_SPLIT_RATIO*100:.0f}%):")
    print(f"  - Videos: {summary['train_split']['num_videos']}")
    print(f"  - Videos/class: min={summary['train_split']['videos_per_class']['min']}, "
          f"max={summary['train_split']['videos_per_class']['max']}, "
          f"mean={summary['train_split']['videos_per_class']['mean']:.1f}")
    print(f"  - Frames/video: min={summary['train_split']['frame_statistics']['min']}, "
          f"max={summary['train_split']['frame_statistics']['max']}, "
          f"mean={summary['train_split']['frame_statistics']['mean']:.1f}")
    print(f"\nVal Set ({(1-TRAIN_SPLIT_RATIO)*100:.0f}%):")
    print(f"  - Videos: {summary['val_split']['num_videos']}")
    
    print("\n" + "=" * 70)
    print("✅ Analysis completed! Check 'figures/' folder for outputs.")
    print("=" * 70)


if __name__ == "__main__":
    main()
