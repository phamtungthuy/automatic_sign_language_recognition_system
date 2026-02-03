"""
Video Anomaly Detection Script
Phat hien video loi: khung hinh toan mau (den, trang, vang, xanh...)

Output:
- figures/video_anomaly_clean_vs_glitchy.png  - Pie chart Clean vs Glitchy
- figures/video_anomaly_glitch_types.png      - Bar chart loai loi
- figures/video_anomaly_severity.png          - Histogram % bad frames
- figures/video_anomaly_report.json           - Bao cao chi tiet
"""

import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm
from collections import defaultdict

# =============================================================================
# CAU HINH
# =============================================================================
DATASET_PATH = Path("dataset/dataset/train")
FIGURES_PATH = Path("figures")
FIGURES_PATH.mkdir(exist_ok=True)

# Nguong phat hien frame loi
STD_THRESHOLD = 10  # Nguong do lech chuan de phat hien solid color
SOLID_COLOR_THRESHOLD = 5  # Bien thien mau qua thap = solid color

# Cac mau solid color can phat hien
SOLID_COLORS = {
    'Solid Black': {'range': ((0, 0, 0), (30, 30, 30))},
    'Solid White': {'range': ((225, 225, 225), (255, 255, 255))},
    'Solid Yellow': {'range': ((200, 200, 0), (255, 255, 100))},
    'Solid Blue': {'range': ((0, 0, 150), (100, 100, 255))},
    'Solid Green': {'range': ((0, 150, 0), (100, 255, 100))},
    'Solid Red': {'range': ((150, 0, 0), (255, 100, 100))}
}

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


# =============================================================================
# DETECTION FUNCTIONS
# =============================================================================
def is_solid_color_frame(frame, std_threshold=STD_THRESHOLD):
    """Kiem tra frame co phai solid color khong"""
    # Tinh do lech chuan cua tung kenh mau
    std_per_channel = np.std(frame, axis=(0, 1))
    
    # Neu tat ca kenh co std qua thap -> solid color
    if np.all(std_per_channel < std_threshold):
        return True, get_solid_color_type(frame)
    
    return False, None


def get_solid_color_type(frame):
    """Xac dinh loai solid color"""
    mean_color = np.mean(frame, axis=(0, 1))
    
    # Kiem tra solid black
    if np.all(mean_color < 30):
        return 'Solid Black'
    
    # Kiem tra solid white
    if np.all(mean_color > 225):
        return 'Solid White'
    
    # Kiem tra solid yellow (R, G cao, B thap)
    if mean_color[0] > 200 and mean_color[1] > 200 and mean_color[2] < 100:
        return 'Solid Yellow'
    
    # Kiem tra solid blue (B cao, R, G thap)
    if mean_color[2] > 150 and mean_color[0] < 100 and mean_color[1] < 100:
        return 'Solid Blue'
    
    # Kiem tra solid green (G cao, R, B thap)
    if mean_color[1] > 150 and mean_color[0] < 100 and mean_color[2] < 100:
        return 'Solid Green'
    
    # Kiem tra solid red (R cao, G, B thap)
    if mean_color[0] > 150 and mean_color[1] < 100 and mean_color[2] < 100:
        return 'Solid Red'
    
    return 'Solid Color'


def analyze_video(video_path):
    """Phan tich video tim frame loi"""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        
        total_frames = 0
        bad_frames = 0
        glitch_types = defaultdict(int)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            total_frames += 1
            
            # Convert BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Kiem tra solid color
            is_solid, color_type = is_solid_color_frame(frame_rgb)
            if is_solid:
                bad_frames += 1
                glitch_types[color_type] += 1
        
        cap.release()
        
        if total_frames == 0:
            return None
        
        bad_percent = (bad_frames / total_frames) * 100
        
        return {
            'total_frames': total_frames,
            'bad_frames': bad_frames,
            'bad_percent': bad_percent,
            'glitch_types': dict(glitch_types),
            'is_clean': bad_frames == 0
        }
        
    except Exception as e:
        return None


def collect_all_videos(dataset_path):
    """Thu thap tat ca video"""
    videos = []
    for label_folder in sorted(os.listdir(dataset_path)):
        path = dataset_path / label_folder
        if path.is_dir():
            for video_file in os.listdir(path):
                if video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    videos.append({
                        'path': path / video_file,
                        'label': label_folder,
                        'filename': video_file
                    })
    return videos


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================
def plot_clean_vs_glitchy(clean_count, glitchy_count, output_path):
    """Pie chart Clean vs Glitchy Videos"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    total = clean_count + glitchy_count
    clean_pct = clean_count / total * 100
    glitchy_pct = glitchy_count / total * 100
    
    sizes = [clean_count, glitchy_count]
    labels = ['Clean', 'Has Issues']
    colors = ['#5B9BD5', '#FF6B6B']
    explode = (0.02, 0.02)
    
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, explode=explode,
        autopct=lambda pct: f'{pct:.1f}%',
        shadow=False, startangle=90,
        textprops={'fontsize': 12, 'fontweight': 'bold'}
    )
    
    # Style
    for autotext in autotexts:
        autotext.set_fontsize(14)
        autotext.set_fontweight('bold')
    
    ax.set_title('Clean vs. Glitchy Videos', fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[SAVED] {output_path}")
    plt.close()


def plot_glitch_types(glitch_counts, output_path):
    """Bar chart Top Glitch Types"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by count
    sorted_types = sorted(glitch_counts.items(), key=lambda x: x[1], reverse=True)
    labels = [t[0] for t in sorted_types]
    counts = [t[1] for t in sorted_types]
    
    # Colors giong trong anh
    colors = ['#4472C4', '#5B9BD5', '#70AD47', '#FFC000', '#5B9BD5', '#ED7D31']
    
    bars = ax.barh(range(len(labels)), counts, color=colors[:len(labels)], edgecolor='black')
    
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel('count', fontsize=12)
    ax.set_title('Top Glitch Types', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    # Add count labels
    for bar, count in zip(bars, counts):
        ax.text(count + max(counts)*0.01, bar.get_y() + bar.get_height()/2,
                str(count), va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[SAVED] {output_path}")
    plt.close()


def plot_severity_distribution(bad_percents, output_path):
    """Histogram Severity Distribution (% Bad Frames per Video)"""
    # Chi lay video co loi
    bad_percents_filtered = [p for p in bad_percents if p > 0]
    
    if not bad_percents_filtered:
        print("[SKIP] No glitchy videos to plot severity")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram voi bins tu 0-100
    bins = np.arange(0, 105, 5)
    n, bins_out, patches = ax.hist(bad_percents_filtered, bins=bins, 
                                    color='#FF6B6B', edgecolor='black', alpha=0.7)
    
    # Fit normal curve (optional - giong anh)
    try:
        from scipy import stats
        x = np.linspace(0, 100, 100)
        if len(bad_percents_filtered) > 5:
            mu, sigma = stats.norm.fit(bad_percents_filtered)
            pdf = stats.norm.pdf(x, mu, sigma) * len(bad_percents_filtered) * 5  # scale
            ax.plot(x, pdf, 'r-', linewidth=2, label=f'Normal fit (mu={mu:.1f}, sigma={sigma:.1f})')
    except:
        pass
    
    ax.set_xlabel('Percent of Video Corrupted', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Severity Distribution (% Bad Frames per Video)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[SAVED] {output_path}")
    plt.close()


def plot_combined_report(clean_count, glitchy_count, glitch_counts, bad_percents, output_path):
    """Tao report 3 bieu do giong anh goc"""
    fig = plt.figure(figsize=(16, 5))
    
    # 1. Pie chart
    ax1 = fig.add_subplot(131)
    total = clean_count + glitchy_count
    sizes = [clean_count, glitchy_count]
    labels = ['Clean', 'Has Issues']
    colors = ['#5B9BD5', '#FF6B6B']
    
    wedges, texts, autotexts = ax1.pie(
        sizes, labels=labels, colors=colors,
        autopct=lambda pct: f'{pct:.1f}%',
        startangle=90, textprops={'fontsize': 10}
    )
    ax1.set_title('Clean vs. Glitchy Videos', fontsize=12, fontweight='bold')
    
    # 2. Bar chart
    ax2 = fig.add_subplot(132)
    sorted_types = sorted(glitch_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    if sorted_types:
        labels2 = [t[0] for t in sorted_types]
        counts2 = [t[1] for t in sorted_types]
        bar_colors = ['#4472C4', '#5B9BD5', '#70AD47', '#FFC000', '#5B9BD5']
        
        bars = ax2.barh(range(len(labels2)), counts2, color=bar_colors[:len(labels2)])
        ax2.set_yticks(range(len(labels2)))
        ax2.set_yticklabels(labels2, fontsize=9)
        ax2.set_xlabel('count')
        ax2.set_title('Top Glitch Types', fontsize=12, fontweight='bold')
        ax2.invert_yaxis()
    
    # 3. Histogram
    ax3 = fig.add_subplot(133)
    bad_percents_filtered = [p for p in bad_percents if p > 0]
    if bad_percents_filtered:
        bins = np.arange(0, 105, 5)
        ax3.hist(bad_percents_filtered, bins=bins, color='#FF6B6B', edgecolor='black', alpha=0.7)
        
        # Fit curve
        try:
            from scipy import stats
            x = np.linspace(0, 100, 100)
            if len(bad_percents_filtered) > 5:
                mu, sigma = stats.norm.fit(bad_percents_filtered)
                pdf = stats.norm.pdf(x, mu, sigma) * len(bad_percents_filtered) * 5
                ax3.plot(x, pdf, 'r-', linewidth=2)
        except:
            pass
    
    ax3.set_xlabel('Percent of Video Corrupted')
    ax3.set_ylabel('Count')
    ax3.set_title('Severity Distribution (% Bad Frames per Video)', fontsize=12, fontweight='bold')
    ax3.set_xlim(0, 100)
    
    plt.suptitle('Anomaly Detection Report', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[SAVED] {output_path}")
    plt.close()


def save_report(results, output_path):
    """Luu report JSON"""
    clean_videos = [r for r in results if r['analysis']['is_clean']]
    glitchy_videos = [r for r in results if not r['analysis']['is_clean']]
    
    # Tong hop glitch types
    all_glitch_types = defaultdict(int)
    for r in glitchy_videos:
        for gtype, count in r['analysis']['glitch_types'].items():
            all_glitch_types[gtype] += count
    
    report = {
        'summary': {
            'total_videos': len(results),
            'clean_videos': len(clean_videos),
            'glitchy_videos': len(glitchy_videos),
            'clean_percent': len(clean_videos) / len(results) * 100 if results else 0,
            'glitchy_percent': len(glitchy_videos) / len(results) * 100 if results else 0
        },
        'glitch_types': dict(sorted(all_glitch_types.items(), key=lambda x: x[1], reverse=True)),
        'severity_stats': {
            'mean_bad_percent': np.mean([r['analysis']['bad_percent'] for r in glitchy_videos]) if glitchy_videos else 0,
            'max_bad_percent': max([r['analysis']['bad_percent'] for r in glitchy_videos]) if glitchy_videos else 0,
            'videos_over_50_percent': len([r for r in glitchy_videos if r['analysis']['bad_percent'] > 50])
        },
        'glitchy_video_list': [
            {
                'path': str(r['path']),
                'label': r['label'],
                'bad_percent': r['analysis']['bad_percent'],
                'bad_frames': r['analysis']['bad_frames'],
                'total_frames': r['analysis']['total_frames'],
                'glitch_types': r['analysis']['glitch_types']
            }
            for r in sorted(glitchy_videos, key=lambda x: x['analysis']['bad_percent'], reverse=True)[:50]
        ]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"[SAVED] {output_path}")
    return report


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("[ANOMALY DETECTION] Video Quality Analysis")
    print("=" * 70)
    
    if not DATASET_PATH.exists():
        print(f"[ERROR] Dataset not found: {DATASET_PATH}")
        return
    
    # 1. Collect videos
    print("\n[1/4] Collecting videos...")
    videos = collect_all_videos(DATASET_PATH)
    print(f"Found {len(videos)} videos")
    
    # 2. Analyze each video
    print("\n[2/4] Analyzing videos for anomalies...")
    results = []
    for v in tqdm(videos, desc="Analyzing"):
        analysis = analyze_video(v['path'])
        if analysis:
            results.append({
                'path': v['path'],
                'label': v['label'],
                'filename': v['filename'],
                'analysis': analysis
            })
    
    print(f"Successfully analyzed {len(results)} videos")
    
    # 3. Generate statistics
    print("\n[3/4] Generating statistics...")
    clean_count = sum(1 for r in results if r['analysis']['is_clean'])
    glitchy_count = len(results) - clean_count
    
    all_glitch_types = defaultdict(int)
    bad_percents = []
    for r in results:
        bad_percents.append(r['analysis']['bad_percent'])
        for gtype, count in r['analysis']['glitch_types'].items():
            all_glitch_types[gtype] += count
    
    print(f"Clean: {clean_count} ({clean_count/len(results)*100:.1f}%)")
    print(f"Glitchy: {glitchy_count} ({glitchy_count/len(results)*100:.1f}%)")
    
    # 4. Create visualizations
    print("\n[4/4] Creating visualizations...")
    
    plot_clean_vs_glitchy(
        clean_count, glitchy_count,
        FIGURES_PATH / "video_anomaly_clean_vs_glitchy.png"
    )
    
    if all_glitch_types:
        plot_glitch_types(
            dict(all_glitch_types),
            FIGURES_PATH / "video_anomaly_glitch_types.png"
        )
    
    plot_severity_distribution(
        bad_percents,
        FIGURES_PATH / "video_anomaly_severity.png"
    )
    
    plot_combined_report(
        clean_count, glitchy_count, dict(all_glitch_types), bad_percents,
        FIGURES_PATH / "video_anomaly_report.png"
    )
    
    report = save_report(results, FIGURES_PATH / "video_anomaly_report.json")
    
    # Print summary
    print("\n" + "=" * 70)
    print("[SUMMARY]")
    print("=" * 70)
    print(f"Total videos: {len(results)}")
    print(f"Clean: {clean_count} ({clean_count/len(results)*100:.1f}%)")
    print(f"Has Issues: {glitchy_count} ({glitchy_count/len(results)*100:.1f}%)")
    print(f"\nTop glitch types:")
    for gtype, count in sorted(all_glitch_types.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  - {gtype}: {count}")
    
    print("\n" + "=" * 70)
    print("[DONE] Check 'figures/' folder for outputs")
    print("=" * 70)


if __name__ == "__main__":
    main()
