"""
Script phân tích frame distribution của tập TRAIN (80% split)
Chỉ phân tích số lượng khung hình trong video
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm

# Cấu hình
DATASET_PATH = Path("dataset/dataset/train")
FIGURES_PATH = Path("figures")
FIGURES_PATH.mkdir(exist_ok=True)

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

TRAIN_SPLIT_RATIO = 0.8
RANDOM_SEED = 42


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
        print(f"Error: {e}")
        return None


def collect_all_videos(dataset_path):
    """Thu thập tất cả video từ dataset"""
    all_videos = []
    
    for label_folder in sorted(os.listdir(dataset_path)):
        path = dataset_path / label_folder
        if path.is_dir():
            for video_file in os.listdir(path):
                if video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    all_videos.append(path / video_file)
    
    return all_videos


def create_train_split(all_videos, train_ratio=0.8, seed=42):
    """Tạo train split 80-20"""
    np.random.seed(seed)
    
    total_size = len(all_videos)
    train_size = int(train_ratio * total_size)
    
    indices = list(range(total_size))
    np.random.shuffle(indices)
    
    train_indices = indices[:train_size]
    train_videos = [all_videos[i] for i in train_indices]
    
    print(f"Total videos: {total_size}")
    print(f"Train set (80%): {len(train_videos)} videos")
    
    return train_videos


def analyze_frame_distribution(train_videos):
    """Phân tích frame count của train set"""
    print("\n🎬 Counting frames in train set...")
    
    frame_counts = []
    for video_path in tqdm(train_videos, desc="Processing"):
        frame_count = count_video_frames(video_path)
        if frame_count and frame_count > 0:
            frame_counts.append(frame_count)
    
    return np.array(frame_counts)


def create_visualization(frame_counts, output_dir):
    """Tạo biểu đồ frame distribution"""
    
    # Calculate statistics
    stats = {
        'total_videos': len(frame_counts),
        'min_frames': int(np.min(frame_counts)),
        'max_frames': int(np.max(frame_counts)),
        'mean_frames': float(np.mean(frame_counts)),
        'median_frames': float(np.median(frame_counts)),
        'std_frames': float(np.std(frame_counts)),
        'quartiles': {
            'q1': float(np.percentile(frame_counts, 25)),
            'q2': float(np.percentile(frame_counts, 50)),
            'q3': float(np.percentile(frame_counts, 75))
        }
    }
    
    # Save JSON
    json_path = output_dir / 'train_split_frame_stats.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Saved: {json_path}")
    
    # Print stats
    print("\n" + "="*70)
    print("📊 TRAIN SET (80%) - FRAME DISTRIBUTION")
    print("="*70)
    print(f"Total videos: {stats['total_videos']}")
    print(f"Min frames: {stats['min_frames']}")
    print(f"Max frames: {stats['max_frames']}")
    print(f"Mean: {stats['mean_frames']:.2f}")
    print(f"Median: {stats['median_frames']:.2f}")
    print(f"Std: {stats['std_frames']:.2f}")
    print(f"Q1: {stats['quartiles']['q1']:.2f}")
    print(f"Q3: {stats['quartiles']['q3']:.2f}")
    print("="*70)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bins = range(
        int(stats['min_frames']), 
        int(stats['max_frames']) + 2, 
        max(1, (int(stats['max_frames']) - int(stats['min_frames'])) // 20)
    )
    ax.hist(frame_counts, bins=bins, edgecolor='black', color='skyblue', alpha=0.8)
    
    ax.set_xlabel('Số lượng khung hình', fontsize=14, fontweight='bold')
    ax.set_ylabel('Số lượng video', fontsize=14, fontweight='bold')
    ax.set_title('Biểu đồ phân bố số lượng khung hình - Train Set (80%)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Info text
    info_text = f'Tổng số video: {stats["total_videos"]}\n'
    info_text += f'Trung bình: {stats["mean_frames"]:.1f} frames\n'
    info_text += f'Trung vị: {stats["median_frames"]:.1f} frames\n'
    info_text += f'Độ lệch chuẩn: {stats["std_frames"]:.1f}'
    
    ax.text(0.02, 0.97, info_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    output_path = output_dir / 'train_split_frame_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved: {output_path}")
    plt.close()


def main():
    print("=" * 70)
    print("📊 PHÂN TÍCH FRAME DISTRIBUTION - TẬP TRAIN (80%)")
    print("=" * 70)
    
    if not DATASET_PATH.exists():
        print(f"❌ Dataset not found: {DATASET_PATH}")
        exit(1)
    
    # Collect videos
    print("\n📂 Collecting videos...")
    all_videos = collect_all_videos(DATASET_PATH)
    
    # Create train split
    train_videos = create_train_split(all_videos, TRAIN_SPLIT_RATIO, RANDOM_SEED)
    
    # Analyze
    frame_counts = analyze_frame_distribution(train_videos)
    
    # Visualize
    create_visualization(frame_counts, FIGURES_PATH)
    
    print("\n" + "=" * 70)
    print("✅ Completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
