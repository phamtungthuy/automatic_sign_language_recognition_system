"""
Script to analyze frame count distribution in training videos
Phân tích phân bố số lượng khung hình trong các video huấn luyện
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
import json
from tqdm import tqdm

# Set Vietnamese font for matplotlib
plt.rcParams['font.family'] = 'DejaVu Sans'

def count_video_frames(video_path):
    """
    Count the number of frames in a video
    
    Args:
        video_path: Path to video file
        
    Returns:
        Number of frames in the video, or None if error
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        
        # Method 1: Try to get frame count directly
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Method 2: If frame count is 0 or unreliable, count manually
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
        print(f"Error processing {video_path}: {e}")
        return None


def analyze_frame_distribution(dataset_path, output_dir):
    """
    Analyze frame count distribution across all training videos
    
    Args:
        dataset_path: Path to dataset/dataset/train folder
        output_dir: Output directory for figures
    """
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔍 Scanning dataset...")
    
    # Collect all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
    all_videos = []
    
    for label_folder in sorted(dataset_path.iterdir()):
        if label_folder.is_dir():
            for video_file in label_folder.iterdir():
                if video_file.suffix.lower() in video_extensions:
                    all_videos.append({
                        'path': video_file,
                        'label': label_folder.name
                    })
    
    print(f"📊 Found {len(all_videos)} videos across {len(list(dataset_path.iterdir()))} labels")
    
    # Count frames for each video
    frame_counts = []
    video_info = []
    
    print("\n📹 Analyzing frame counts...")
    for video in tqdm(all_videos, desc="Processing videos"):
        frame_count = count_video_frames(video['path'])
        if frame_count is not None and frame_count > 0:
            frame_counts.append(frame_count)
            video_info.append({
                'label': video['label'],
                'filename': video['path'].name,
                'frame_count': frame_count
            })
    
    if not frame_counts:
        print("❌ No valid videos found!")
        return
    
    frame_counts = np.array(frame_counts)
    
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
    
    print("\n📈 Frame Count Statistics:")
    print(f"  Total videos: {stats['total_videos']}")
    print(f"  Min frames: {stats['min_frames']}")
    print(f"  Max frames: {stats['max_frames']}")
    print(f"  Mean frames: {stats['mean_frames']:.2f}")
    print(f"  Median frames: {stats['median_frames']:.2f}")
    print(f"  Std deviation: {stats['std_frames']:.2f}")
    print(f"  Q1: {stats['quartiles']['q1']:.2f}")
    print(f"  Q2: {stats['quartiles']['q2']:.2f}")
    print(f"  Q3: {stats['quartiles']['q3']:.2f}")
    
    # Save detailed statistics to JSON
    stats_output = {
        'statistics': stats,
        'video_details': video_info
    }
    
    json_path = output_dir / 'frame_distribution_stats.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(stats_output, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Saved detailed statistics to {json_path}")
    
    # Create visualizations
    create_frame_distribution_plots(frame_counts, stats, output_dir)
    
    return stats, frame_counts


def create_frame_distribution_plots(frame_counts, stats, output_dir):
    """
    Create comprehensive visualization of frame distribution
    
    Args:
        frame_counts: Array of frame counts
        stats: Statistics dictionary
        output_dir: Output directory for figures
    """
    
    # Plot 1: Histogram of frame distribution
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Histogram
    ax = axes[0, 0]
    counts, bins, patches = ax.hist(frame_counts, bins=30, edgecolor='black', 
                                      color='skyblue', alpha=0.7)
    ax.axvline(stats['mean_frames'], color='red', linestyle='--', 
               linewidth=2, label=f"Mean: {stats['mean_frames']:.1f}")
    ax.axvline(stats['median_frames'], color='green', linestyle='--', 
               linewidth=2, label=f"Median: {stats['median_frames']:.1f}")
    ax.set_xlabel('Số lượng khung hình (frames)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Số lượng video', fontsize=12, fontweight='bold')
    ax.set_title('Biểu đồ phân bố số lượng khung hình trong video', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Box plot
    ax = axes[0, 1]
    bp = ax.boxplot(frame_counts, vert=True, patch_artist=True,
                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5))
    ax.set_ylabel('Số lượng khung hình (frames)', fontsize=12, fontweight='bold')
    ax.set_title('Box Plot - Phân bố khung hình', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics text
    stats_text = f"""
    Min: {stats['min_frames']}
    Q1: {stats['quartiles']['q1']:.1f}
    Median: {stats['median_frames']:.1f}
    Q3: {stats['quartiles']['q3']:.1f}
    Max: {stats['max_frames']}
    Mean: {stats['mean_frames']:.1f}
    Std: {stats['std_frames']:.1f}
    """
    ax.text(1.15, 0.5, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Cumulative distribution
    ax = axes[1, 0]
    sorted_frames = np.sort(frame_counts)
    cumulative = np.arange(1, len(sorted_frames) + 1) / len(sorted_frames) * 100
    ax.plot(sorted_frames, cumulative, linewidth=2, color='navy')
    ax.axhline(50, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(stats['median_frames'], color='red', linestyle='--', 
               linewidth=1, alpha=0.5, label=f"Median: {stats['median_frames']:.1f}")
    ax.set_xlabel('Số lượng khung hình (frames)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Phần trăm tích lũy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Phân bố tích lũy số lượng khung hình', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Frame count frequency (grouped)
    ax = axes[1, 1]
    # Group by bins
    bin_edges = np.arange(stats['min_frames'], stats['max_frames'] + 5, 5)
    hist, edges = np.histogram(frame_counts, bins=bin_edges)
    bin_centers = (edges[:-1] + edges[1:]) / 2
    
    bars = ax.bar(bin_centers, hist, width=4, edgecolor='black', 
                   color='lightcoral', alpha=0.7)
    ax.set_xlabel('Số lượng khung hình (frames)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Tần suất', fontsize=12, fontweight='bold')
    ax.set_title('Tần suất xuất hiện các khoảng khung hình', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / 'frame_distribution_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved frame distribution plot to {output_path}")
    plt.close()
    
    # Plot 2: Simple histogram like in the reference image
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create histogram with specific bins
    counts, bins, patches = ax.hist(frame_counts, bins=range(
        int(stats['min_frames']), 
        int(stats['max_frames']) + 2, 
        max(1, (int(stats['max_frames']) - int(stats['min_frames'])) // 20)
    ), edgecolor='black', color='skyblue', alpha=0.8)
    
    ax.set_xlabel('Số lượng khung hình', fontsize=14, fontweight='bold')
    ax.set_ylabel('Số lượng video', fontsize=14, fontweight='bold')
    ax.set_title('Biểu đồ phân bố số lượng khung hình trong video', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add text with statistics
    info_text = f'Tổng số video: {stats["total_videos"]}\n'
    info_text += f'Trung bình: {stats["mean_frames"]:.1f} frames\n'
    info_text += f'Trung vị: {stats["median_frames"]:.1f} frames\n'
    info_text += f'Độ lệch chuẩn: {stats["std_frames"]:.1f}'
    
    ax.text(0.02, 0.97, info_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    output_path = output_dir / 'frame_distribution_simple.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved simple frame distribution plot to {output_path}")
    plt.close()


if __name__ == "__main__":
    # Configuration
    DATASET_PATH = "dataset/dataset/train"
    OUTPUT_DIR = "figures"
    
    print("=" * 60)
    print("📊 FRAME DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset path not found: {DATASET_PATH}")
        print("Please update DATASET_PATH in the script")
        exit(1)
    
    # Run analysis
    stats, frame_counts = analyze_frame_distribution(DATASET_PATH, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("✅ Analysis completed successfully!")
    print("=" * 60)
