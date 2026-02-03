"""
Script phân tích đặc điểm không gian và thời gian của video dataset
Phân tích: độ phân giải, số frame, fps, duration của các video
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json
from collections import defaultdict

# Cấu hình đường dẫn
DATASET_PATH = Path("dataset/dataset/train")
FIGURES_PATH = Path("figures")
FIGURES_PATH.mkdir(exist_ok=True)

# Font tiếng Việt cho matplotlib
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class VideoAnalyzer:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.video_stats = []
        self.label_stats = defaultdict(list)
        
    def analyze_video(self, video_path):
        """Phân tích một video và trả về thông tin không gian và thời gian"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                return None
            
            # Đặc điểm không gian
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Đặc điểm thời gian
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            return {
                'path': str(video_path),
                'width': width,
                'height': height,
                'aspect_ratio': width / height if height > 0 else 0,
                'fps': fps,
                'frame_count': frame_count,
                'duration': duration
            }
        except Exception as e:
            print(f"Lỗi khi phân tích {video_path}: {e}")
            return None
    
    def scan_dataset(self):
        """Quét toàn bộ dataset và phân tích tất cả video"""
        print("Đang quét dataset...")
        
        # Lấy danh sách tất cả các folder nhãn
        label_folders = [f for f in self.dataset_path.iterdir() if f.is_dir()]
        print(f"Tìm thấy {len(label_folders)} nhãn")
        
        total_videos = 0
        for label_folder in label_folders:
            video_files = list(label_folder.glob("*.mp4")) + \
                         list(label_folder.glob("*.avi")) + \
                         list(label_folder.glob("*.mov"))
            total_videos += len(video_files)
        
        print(f"Tổng số video: {total_videos}")
        
        # Phân tích từng video
        with tqdm(total=total_videos, desc="Phân tích video") as pbar:
            for label_folder in label_folders:
                label = label_folder.name
                
                # Lấy tất cả video trong folder
                video_files = list(label_folder.glob("*.mp4")) + \
                             list(label_folder.glob("*.avi")) + \
                             list(label_folder.glob("*.mov"))
                
                for video_file in video_files:
                    stats = self.analyze_video(video_file)
                    if stats:
                        stats['label'] = label
                        self.video_stats.append(stats)
                        self.label_stats[label].append(stats)
                    pbar.update(1)
        
        print(f"Đã phân tích thành công {len(self.video_stats)} video")
        return self.video_stats
    
    def plot_spatial_analysis(self):
        """Vẽ biểu đồ phân tích đặc điểm không gian"""
        if not self.video_stats:
            print("Chưa có dữ liệu để vẽ")
            return
        
        widths = [v['width'] for v in self.video_stats]
        heights = [v['height'] for v in self.video_stats]
        
        # 1. Biểu đồ phân bố độ phân giải (scatter plot)
        plt.figure(figsize=(12, 8))
        plt.scatter(widths, heights, alpha=0.5, s=20)
        
        # Vẽ đường tỷ lệ khung hình vuông (1:1)
        max_dim = max(max(widths), max(heights))
        plt.plot([0, max_dim], [0, max_dim], 'r--', 
                label=f'Tỷ lệ khung hình vuông (1:1)', linewidth=2)
        
        plt.xlabel('Chiều rộng (pixel)', fontsize=12)
        plt.ylabel('Chiều cao (pixel)', fontsize=12)
        plt.title('Biểu đồ phân bố độ phân giải', 
                 fontsize=14, pad=20)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIGURES_PATH / 'resolution_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Đã lưu: {FIGURES_PATH / 'resolution_distribution.png'}")
        
        # 2. Histogram độ phân giải
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].hist(widths, bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[0].set_xlabel('Chiều rộng (pixel)', fontsize=11)
        axes[0].set_ylabel('Số lượng video', fontsize=11)
        axes[0].set_title('Phân bố chiều rộng video', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        axes[1].hist(heights, bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[1].set_xlabel('Chiều cao (pixel)', fontsize=11)
        axes[1].set_ylabel('Số lượng video', fontsize=11)
        axes[1].set_title('Phân bố chiều cao video', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(FIGURES_PATH / 'resolution_histogram.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Đã lưu: {FIGURES_PATH / 'resolution_histogram.png'}")
        
        # 3. Biểu đồ tỷ lệ khung hình
        aspect_ratios = [v['aspect_ratio'] for v in self.video_stats]
        plt.figure(figsize=(12, 6))
        plt.hist(aspect_ratios, bins=50, alpha=0.7, color='purple', edgecolor='black')
        plt.xlabel('Tỷ lệ khung hình (width/height)', fontsize=11)
        plt.ylabel('Số lượng video', fontsize=11)
        plt.title('Phân bố tỷ lệ khung hình', fontsize=12)
        plt.axvline(x=1.0, color='r', linestyle='--', label='Tỷ lệ 1:1 (vuông)', linewidth=2)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIGURES_PATH / 'aspect_ratio_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Đã lưu: {FIGURES_PATH / 'aspect_ratio_distribution.png'}")
    
    def plot_temporal_analysis(self):
        """Vẽ biểu đồ phân tích đặc điểm thời gian"""
        if not self.video_stats:
            print("Chưa có dữ liệu để vẽ")
            return
        
        fps_list = [v['fps'] for v in self.video_stats]
        frame_counts = [v['frame_count'] for v in self.video_stats]
        durations = [v['duration'] for v in self.video_stats]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. FPS distribution
        axes[0, 0].hist(fps_list, bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[0, 0].set_xlabel('FPS (frames per second)', fontsize=11)
        axes[0, 0].set_ylabel('Số lượng video', fontsize=11)
        axes[0, 0].set_title('Phân bố FPS', fontsize=12)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Frame count distribution
        axes[0, 1].hist(frame_counts, bins=30, alpha=0.7, color='red', edgecolor='black')
        axes[0, 1].set_xlabel('Số lượng frame', fontsize=11)
        axes[0, 1].set_ylabel('Số lượng video', fontsize=11)
        axes[0, 1].set_title('Phân bố số lượng frame', fontsize=12)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Duration distribution
        axes[1, 0].hist(durations, bins=30, alpha=0.7, color='teal', edgecolor='black')
        axes[1, 0].set_xlabel('Thời lượng (giây)', fontsize=11)
        axes[1, 0].set_ylabel('Số lượng video', fontsize=11)
        axes[1, 0].set_title('Phân bố thời lượng video', fontsize=12)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. FPS vs Duration scatter plot
        axes[1, 1].scatter(durations, fps_list, alpha=0.5, s=20, color='brown')
        axes[1, 1].set_xlabel('Thời lượng (giây)', fontsize=11)
        axes[1, 1].set_ylabel('FPS', fontsize=11)
        axes[1, 1].set_title('Mối quan hệ FPS và thời lượng', fontsize=12)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(FIGURES_PATH / 'temporal_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Đã lưu: {FIGURES_PATH / 'temporal_analysis.png'}")
    
    def plot_label_statistics(self):
        """Vẽ biểu đồ thống kê theo nhãn"""
        if not self.label_stats:
            print("Chưa có dữ liệu để vẽ")
            return
        
        # Số lượng video theo nhãn
        labels = list(self.label_stats.keys())
        counts = [len(videos) for videos in self.label_stats.values()]
        
        plt.figure(figsize=(16, 8))
        bars = plt.bar(range(len(labels)), counts, alpha=0.7, color='steelblue', edgecolor='black')
        plt.xlabel('Nhãn', fontsize=11)
        plt.ylabel('Số lượng video', fontsize=11)
        plt.title(f'Phân bố số lượng video theo nhãn (Tổng: {len(labels)} nhãn)', fontsize=12)
        plt.xticks(range(len(labels)), labels, rotation=90, fontsize=8)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Thêm giá trị lên đầu cột
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=7)
        
        plt.tight_layout()
        plt.savefig(FIGURES_PATH / 'videos_per_label.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Đã lưu: {FIGURES_PATH / 'videos_per_label.png'}")
    
    def generate_summary_stats(self):
        """Tạo thống kê tổng quan"""
        if not self.video_stats:
            print("Chưa có dữ liệu")
            return
        
        widths = [v['width'] for v in self.video_stats]
        heights = [v['height'] for v in self.video_stats]
        fps_list = [v['fps'] for v in self.video_stats]
        frame_counts = [v['frame_count'] for v in self.video_stats]
        durations = [v['duration'] for v in self.video_stats]
        
        summary = {
            'total_videos': len(self.video_stats),
            'total_labels': len(self.label_stats),
            'spatial_stats': {
                'width': {
                    'min': int(np.min(widths)),
                    'max': int(np.max(widths)),
                    'mean': float(np.mean(widths)),
                    'std': float(np.std(widths))
                },
                'height': {
                    'min': int(np.min(heights)),
                    'max': int(np.max(heights)),
                    'mean': float(np.mean(heights)),
                    'std': float(np.std(heights))
                }
            },
            'temporal_stats': {
                'fps': {
                    'min': float(np.min(fps_list)),
                    'max': float(np.max(fps_list)),
                    'mean': float(np.mean(fps_list)),
                    'std': float(np.std(fps_list))
                },
                'frame_count': {
                    'min': int(np.min(frame_counts)),
                    'max': int(np.max(frame_counts)),
                    'mean': float(np.mean(frame_counts)),
                    'std': float(np.std(frame_counts))
                },
                'duration': {
                    'min': float(np.min(durations)),
                    'max': float(np.max(durations)),
                    'mean': float(np.mean(durations)),
                    'std': float(np.std(durations))
                }
            }
        }
        
        # Lưu thống kê ra file JSON
        with open(FIGURES_PATH / 'dataset_statistics.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
        print(f"✓ Đã lưu: {FIGURES_PATH / 'dataset_statistics.json'}")
        
        # In thống kê ra console
        print("\n" + "="*60)
        print("THỐNG KÊ TỔNG QUAN DATASET")
        print("="*60)
        print(f"Tổng số video: {summary['total_videos']}")
        print(f"Tổng số nhãn: {summary['total_labels']}")
        print("\nĐẶC ĐIỂM KHÔNG GIAN:")
        print(f"  Chiều rộng: {summary['spatial_stats']['width']['min']}-{summary['spatial_stats']['width']['max']} pixel (TB: {summary['spatial_stats']['width']['mean']:.1f})")
        print(f"  Chiều cao: {summary['spatial_stats']['height']['min']}-{summary['spatial_stats']['height']['max']} pixel (TB: {summary['spatial_stats']['height']['mean']:.1f})")
        print("\nĐẶC ĐIỂM THỜI GIAN:")
        print(f"  FPS: {summary['temporal_stats']['fps']['min']:.1f}-{summary['temporal_stats']['fps']['max']:.1f} (TB: {summary['temporal_stats']['fps']['mean']:.1f})")
        print(f"  Số frame: {summary['temporal_stats']['frame_count']['min']}-{summary['temporal_stats']['frame_count']['max']} (TB: {summary['temporal_stats']['frame_count']['mean']:.1f})")
        print(f"  Thời lượng: {summary['temporal_stats']['duration']['min']:.2f}-{summary['temporal_stats']['duration']['max']:.2f}s (TB: {summary['temporal_stats']['duration']['mean']:.2f}s)")
        print("="*60)
        
        return summary


def main():
    print("Bắt đầu phân tích dataset video...")
    print(f"Dataset path: {DATASET_PATH.absolute()}")
    print(f"Figures path: {FIGURES_PATH.absolute()}\n")
    
    # Kiểm tra dataset có tồn tại không
    if not DATASET_PATH.exists():
        print(f"❌ Không tìm thấy dataset tại {DATASET_PATH}")
        return
    
    # Khởi tạo analyzer
    analyzer = VideoAnalyzer(DATASET_PATH)
    
    # Quét và phân tích dataset
    analyzer.scan_dataset()
    
    if not analyzer.video_stats:
        print("❌ Không tìm thấy video nào để phân tích")
        return
    
    print("\nĐang tạo biểu đồ phân tích...")
    
    # Tạo các biểu đồ
    analyzer.plot_spatial_analysis()
    # analyzer.plot_temporal_analysis()
    # analyzer.plot_label_statistics()
    
    # Tạo thống kê tổng quan
    # analyzer.generate_summary_stats()
    
    print(f"\n✅ Hoàn thành! Tất cả biểu đồ đã được lưu trong folder '{FIGURES_PATH}'")


if __name__ == "__main__":
    main()
