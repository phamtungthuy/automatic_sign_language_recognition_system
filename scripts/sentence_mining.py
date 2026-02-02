"""
Sign Language Sentence Mining & Video Concatenation

This script:
1. Analyzes available sign labels
2. Generates meaningful Vietnamese sentences from available labels
3. Concatenates videos to create sentence demonstrations
"""

import os
import pickle
import random
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np


def convert_to_browser_compatible(input_path: Path, output_path: Optional[Path] = None) -> Path:
    """
    Convert video to browser-compatible format using ffmpeg.
    If ffmpeg not available, returns original path.
    """
    if output_path is None:
        output_path = input_path.with_suffix('.webm')
    
    # Check if ffmpeg is available
    if shutil.which('ffmpeg') is None:
        print("  ⚠️ ffmpeg not found, video may not play in browser")
        return input_path
    
    try:
        # Convert to WebM (VP9) which is universally supported
        cmd = [
            'ffmpeg', '-y', '-i', str(input_path),
            '-c:v', 'libvpx-vp9', '-crf', '30', '-b:v', '0',
            '-an',  # No audio
            str(output_path)
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        
        # If successful, optionally remove original
        if output_path.exists() and output_path.stat().st_size > 0:
            return output_path
    except subprocess.CalledProcessError:
        pass
    except Exception:
        pass
    
    # Fallback: try H.264 mp4
    h264_path = input_path.with_name(input_path.stem + '_h264.mp4')
    try:
        cmd = [
            'ffmpeg', '-y', '-i', str(input_path),
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
            '-pix_fmt', 'yuv420p',  # Required for browser compatibility
            '-an',
            str(h264_path)
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        
        if h264_path.exists() and h264_path.stat().st_size > 0:
            # Replace original with H.264 version
            h264_path.replace(input_path)
            return input_path
    except Exception:
        pass
    
    return input_path


# ============== CONFIGURATION ==============
ROOT_PATH = Path(__file__).parent.parent
TRAIN_DIR = ROOT_PATH / "dataset" / "dataset" / "train"
LABEL_MAPPING_PATH = ROOT_PATH / "models" / "slr" / "label_mapping.pkl"
OUTPUT_DIR = ROOT_PATH / "output" / "sentence_videos"


# ============== PREDEFINED MEANINGFUL SENTENCES ==============
# Các câu có nghĩa ghép từ labels có sẵn
MEANINGFUL_SENTENCES = [
    # Chào hỏi cơ bản
    ["Chào"],
    ["Cảm ơn"],
    ["Xin lỗi"],
    
    # Câu đơn giản
    ["Tôi", "Ăn"],
    ["Tôi", "Uống"],
    ["Tôi", "Đi"],
    ["Tôi", "Chạy"],
    ["Tôi", "Nghe"],
    ["Tôi", "Nhìn"],
    ["Tôi", "Nói"],
    ["Tôi", "Biết"],
    ["Tôi", "Thích"],
    ["Tôi", "Ghét"],
    ["Tôi", "Nhớ"],
    ["Tôi", "Cần"],
    ["Tôi", "Khóc"],
    
    # Câu ghép với chủ ngữ
    ["Chúng ta", "Ăn"],
    ["Chúng ta", "Đi"],
    ["Chúng ta", "Chạy"],
    ["Họ", "Ăn"],
    ["Họ", "Đi"],
    
    # Câu về sức khỏe (COVID theme)
    ["Tôi", "Sốt"],
    ["Tôi", "Ho"],
    ["Tôi", "Nôn ói"],
    ["Bệnh nhân", "Sốt"],
    ["Bệnh nhân", "Ho"],
    ["Bệnh nhân", "Phục hồi"],
    ["Bệnh nhân", "Xuất viện"],
    
    # Câu về địa điểm
    ["Đi", "Bệnh viện"],
    ["Đi", "Trường học"],
    ["Đi", "Nhà"],
    ["Tôi", "Đi", "Bệnh viện"],
    ["Tôi", "Đi", "Trường học"],
    ["Tôi", "Đi", "Nhà"],
    
    # Câu về hướng dẫn
    ["Rẽ phải"],
    ["Rẽ trái"],
    ["Chậm lại"],
    ["Mời vào"],
    
    # Câu về thời gian
    ["Hôm nay"],
    ["Ban ngày"],
    ["Ban đêm"],
    ["Chiều"],
    ["Tối"],
    ["Trưa"],
    ["Thức dậy"],
    
    # Câu phức tạp hơn
    ["Tôi", "Cần", "Giúp"],
    ["Tôi", "Cần", "Nghỉ ngơi"],
    ["Tôi", "Cần", "Thức ăn"],
    ["Tôi", "Thích", "Ăn"],
    ["Tôi", "Thích", "Tập luyện"],
    
    # Câu về xe cộ
    ["Tôi", "Đi", "Ô tô"],
    ["Tôi", "Đi", "Xe máy"],
    ["Tôi", "Đi", "Xe đạp"],
    
    # Câu về học sinh
    ["Học sinh", "Đi", "Trường học"],
    ["Học sinh", "Tập luyện"],
    ["Dạy dỗ", "Học sinh"],
    
    # Câu về cảm xúc
    ["Tôi", "Lo lắng"],
    ["Tôi", "Băn khoăn"],
    ["Tôi", "Xúc động"],
    ["Tôi", "Ăn mừng"],
    
    # Câu về đồng ý/từ chối
    ["Đồng ý"],
    ["Chấp nhận"],
    ["Xin phép"],
    ["Vâng lời"],
    
    # Câu về COVID
    ["Sử dụng", "Khẩu trang"],
    ["Cách ly"],
    ["Khai báo"],
    ["Lây bệnh"],
    ["Khu cách ly"],
    ["Bộ y tế"],
    
    # Câu về thăm hỏi
    ["Thăm", "Bệnh nhân"],
    ["Thăm", "Bạn thân"],
    ["An ủi", "Bệnh nhân"],
    
    # Câu 3 từ
    ["Tôi", "Thích", "Cá"],
    ["Tôi", "Thích", "Rau"],
    ["Tôi", "Ăn", "Cá"],
    ["Tôi", "Ăn", "Rau"],
    ["Chúng ta", "Cần", "Giúp"],
    ["Bạn thân", "Giúp", "Tôi"],
    
    # Câu 4 từ
    ["Hôm nay", "Tôi", "Đi", "Bệnh viện"],
    ["Hôm nay", "Tôi", "Đi", "Trường học"],
    ["Bệnh nhân", "Cần", "Nghỉ ngơi"],
    ["Học sinh", "Cần", "Tập luyện"],
    
    # Câu về biếu tặng
    ["Tôi", "Biếu tặng"],
    ["San sẻ"],
    ["Ủng hộ"],
    ["Hâm mộ"],
    
    # Câu về cơ thể
    ["Cơ thể"],
    ["Bàn tay"],
    ["Ngón tay"],
    ["Chân"],
    ["Đầu"],
    
    # Câu có thể/dễ
    ["Có thể"],
    ["Dễ"],
    ["Nặng"],
    
    # Câu hỏi địa điểm
    ["Đâu"],
    ["Phía sau"],
    ["Xa"],
]


def load_label_mapping() -> Dict[str, int]:
    """Load label mapping from pickle file"""
    with open(LABEL_MAPPING_PATH, 'rb') as f:
        return pickle.load(f)


def get_available_labels() -> List[str]:
    """Get list of labels that have videos in train directory"""
    labels = []
    if TRAIN_DIR.exists():
        for folder in TRAIN_DIR.iterdir():
            if folder.is_dir():
                labels.append(folder.name)
    return labels


def validate_sentences(sentences: List[List[str]], available_labels: List[str]) -> List[Tuple[List[str], bool]]:
    """
    Validate which sentences can be formed from available labels
    Returns: List of (sentence, is_valid)
    """
    results = []
    for sentence in sentences:
        is_valid = all(word in available_labels for word in sentence)
        results.append((sentence, is_valid))
    return results


def get_random_video_for_label(label: str) -> Optional[Path]:
    """Get a random video file for a label"""
    label_dir = TRAIN_DIR / label
    if not label_dir.exists():
        return None
    
    videos = list(label_dir.glob("*.mp4")) + list(label_dir.glob("*.avi"))
    if not videos:
        return None
    
    return random.choice(videos)


def concatenate_videos(video_paths: List[Path], output_path: Path, fps: int = 30) -> bool:
    """
    Concatenate multiple videos into one
    
    Args:
        video_paths: List of video file paths
        output_path: Output video path
        fps: Output video FPS
    
    Returns:
        True if successful
    """
    if not video_paths:
        return False
    
    # Read all frames from all videos
    all_frames = []
    target_size = None
    
    for video_path in video_paths:
        cap = cv2.VideoCapture(str(video_path))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Set target size from first video
            if target_size is None:
                target_size = (frame.shape[1], frame.shape[0])
            
            # Resize if needed
            if (frame.shape[1], frame.shape[0]) != target_size:
                frame = cv2.resize(frame, target_size)
            
            all_frames.append(frame)
        cap.release()
    
    if not all_frames:
        return False
    
    # Write output video with browser-compatible codec
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Try multiple codecs for compatibility
    # H.264 codecs that work across platforms
    codecs_to_try = ['avc1', 'H264', 'X264', 'mp4v']
    
    for codec in codecs_to_try:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(str(output_path), fourcc, fps, target_size)
            
            if out.isOpened():
                for frame in all_frames:
                    out.write(frame)
                out.release()
                
                # Verify the output file exists and has content
                if output_path.exists() and output_path.stat().st_size > 0:
                    return True
        except Exception:
            continue
    
    # Fallback: try with .avi extension
    avi_path = output_path.with_suffix('.avi')
    try:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(str(avi_path), fourcc, fps, target_size)
        if out.isOpened():
            for frame in all_frames:
                out.write(frame)
            out.release()
            return True
    except Exception:
        pass
    
    return False


def create_sentence_video(sentence: List[str], output_name: Optional[str] = None) -> Optional[Path]:
    """
    Create a concatenated video for a sentence
    
    Args:
        sentence: List of labels forming the sentence
        output_name: Optional custom output filename
    
    Returns:
        Path to output video or None if failed
    """
    # Get random video for each word
    video_paths = []
    for word in sentence:
        video_path = get_random_video_for_label(word)
        if video_path is None:
            print(f"  ⚠️ No video found for '{word}'")
            return None
        video_paths.append(video_path)
    
    # Create output filename
    if output_name is None:
        output_name = "_".join(sentence).replace(" ", "-")
    
    output_path = OUTPUT_DIR / f"{output_name}.mp4"
    
    # Concatenate
    if concatenate_videos(video_paths, output_path):
        # Convert to browser-compatible format
        final_path = convert_to_browser_compatible(output_path)
        return final_path
    return None


def mine_and_create_sentences():
    """Main function to mine sentences and create videos"""
    print("=" * 60)
    print("🔍 SIGN LANGUAGE SENTENCE MINING & VIDEO CONCATENATION")
    print("=" * 60)
    
    # Get available labels
    available_labels = get_available_labels()
    print(f"\n📁 Found {len(available_labels)} labels in {TRAIN_DIR}")
    
    # Validate sentences
    print(f"\n📝 Checking {len(MEANINGFUL_SENTENCES)} predefined sentences...")
    
    valid_sentences = []
    invalid_sentences = []
    
    for sentence, is_valid in validate_sentences(MEANINGFUL_SENTENCES, available_labels):
        if is_valid:
            valid_sentences.append(sentence)
        else:
            invalid_sentences.append(sentence)
    
    print(f"   ✅ Valid sentences: {len(valid_sentences)}")
    print(f"   ❌ Invalid sentences: {len(invalid_sentences)}")
    
    # Print some valid sentences
    print("\n📋 Valid sentences (first 20):")
    for i, sentence in enumerate(valid_sentences[:20]):
        print(f"   {i+1:2d}. {' + '.join(sentence)}")
    
    if len(valid_sentences) > 20:
        print(f"   ... and {len(valid_sentences) - 20} more")
    
    # Ask user which sentences to create
    print("\n" + "=" * 60)
    print("🎬 CREATING SENTENCE VIDEOS")
    print("=" * 60)
    
    # Create videos for all valid sentences
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    created = 0
    for i, sentence in enumerate(valid_sentences):
        sentence_text = " → ".join(sentence)
        print(f"\n[{i+1}/{len(valid_sentences)}] Creating: {sentence_text}")
        
        output_path = create_sentence_video(sentence)
        if output_path:
            print(f"   ✅ Saved to: {output_path}")
            created += 1
        else:
            print(f"   ❌ Failed")
    
    print("\n" + "=" * 60)
    print(f"✅ Created {created}/{len(valid_sentences)} sentence videos")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print("=" * 60)


def list_valid_sentences():
    """Just list valid sentences without creating videos"""
    available_labels = get_available_labels()
    
    print("📋 Valid sentences that can be formed:\n")
    
    for sentence, is_valid in validate_sentences(MEANINGFUL_SENTENCES, available_labels):
        if is_valid:
            print(f"  ✅ {' → '.join(sentence)}")
    
    print(f"\n📁 Available labels ({len(available_labels)}):")
    print(", ".join(sorted(available_labels)))


# ============== CLI ==============
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Sign Language Sentence Mining")
    parser.add_argument("--list", action="store_true", help="Just list valid sentences")
    parser.add_argument("--create", action="store_true", help="Create sentence videos")
    parser.add_argument("--sentence", nargs="+", help="Create specific sentence, e.g., --sentence Tôi Ăn Cá")
    parser.add_argument("--output", type=str, help="Output filename for --sentence")
    
    args = parser.parse_args()
    
    if args.list:
        list_valid_sentences()
    elif args.sentence:
        print(f"Creating video for: {' → '.join(args.sentence)}")
        output_path = create_sentence_video(args.sentence, args.output)
        if output_path:
            print(f"✅ Created: {output_path}")
        else:
            print("❌ Failed to create video")
    elif args.create:
        mine_and_create_sentences()
    else:
        # Default: list valid sentences
        list_valid_sentences()
        print("\n💡 Use --create to generate all sentence videos")
        print("💡 Use --sentence Tôi Ăn to create specific sentence")
