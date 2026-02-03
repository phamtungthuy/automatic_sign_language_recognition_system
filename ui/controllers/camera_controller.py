import cv2
import numpy as np
import time
import shutil
import os
from typing import List, Tuple, Optional


class CameraController:
    def __init__(self, debug_dir: Optional[str] = None):
        self.sliding_window_buffer: List[np.ndarray] = []
        self.is_recording: bool = False
        self.window_start_time: float = 0
        self.last_send_time: float = 0
        self.frame_count: int = 0
        
        if debug_dir:
            self.debug_dir = debug_dir
        else:
            self.debug_dir = os.path.join(os.path.dirname(__file__), "..", "debug_videos")
        os.makedirs(self.debug_dir, exist_ok=True)
    
    
    def start_recording(self) -> str:
        self.is_recording = True
        self.sliding_window_buffer.clear()
        self.last_send_time = time.time()
        self.window_start_time = time.time()
        self.frame_count = 0
        return "Dang recording..."
    
    def stop_recording(self) -> str:
        """Dừng recording"""
        self.is_recording = False
        frame_count = len(self.sliding_window_buffer)
        return f"Da dung. Da thu {frame_count} frames."
    
    def clear_buffer(self) -> None:
        """Xóa buffer frames"""
        self.sliding_window_buffer.clear()
        self.frame_count = 0
    
    # ============== FRAME MANAGEMENT ==============
    
    def add_frame(self, frame: np.ndarray, max_buffer_size: int = 100) -> None:
        """
        Thêm frame vào buffer
        
        Args:
            frame: numpy array của frame (RGB)
            max_buffer_size: giới hạn tối đa frames trong buffer
        """
        self.sliding_window_buffer.append(frame.copy())
        self.frame_count += 1
        
        # Giới hạn buffer size
        if len(self.sliding_window_buffer) > max_buffer_size:
            self.sliding_window_buffer = self.sliding_window_buffer[-max_buffer_size:]
    
    def get_elapsed_time(self) -> float:
        """Thời gian đã trôi qua từ lúc bắt đầu window hiện tại"""
        if self.window_start_time == 0:
            return 0
        return time.time() - self.window_start_time
    
    def should_send(self, collect_seconds: float, min_frames: int = 5) -> bool:
        """
        Kiểm tra đã đủ điều kiện để gửi predict chưa
        
        Args:
            collect_seconds: thời gian thu thập tối thiểu (giây)
            min_frames: số frames tối thiểu
        """
        elapsed = self.get_elapsed_time()
        return elapsed >= collect_seconds and len(self.sliding_window_buffer) >= min_frames
    
    def pop_frames_for_prediction(self) -> Tuple[List[np.ndarray], float]:
        """
        Lấy frames ra để predict và reset buffer
        
        Returns:
            (frames, real_fps): list frames và fps thực tế
        """
        frames = list(self.sliding_window_buffer)
        elapsed_time = self.get_elapsed_time()
        
        # Tính FPS thực: frames / thời gian
        real_fps = len(frames) / elapsed_time if elapsed_time > 0 else 0
        
        # Reset buffer và timer
        self.sliding_window_buffer.clear()
        self.window_start_time = time.time()
        self.last_send_time = time.time()
        
        return frames, real_fps
    
    def get_buffer_info(self) -> str:
        """Lấy thông tin buffer để hiển thị"""
        return f"Buffer: {len(self.sliding_window_buffer)} frames"
    
    # ============== VIDEO CONVERSION ==============
    
    def frames_to_video(self, frames: List[np.ndarray], output_path: str, fps: float = 0.0) -> bool:
        """
        Convert frames thành video file
        
        Args:
            frames: list các numpy array (RGB)
            output_path: đường dẫn file video output
            fps: fps của video (nếu <= 0 sẽ tự tính)
        
        Returns:
            True nếu thành công
        """
        if not frames:
            return False
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Nếu fps không được cung cấp, tính default
        if fps <= 0:
            actual_fps = max(5, len(frames) / 2.5)
        else:
            actual_fps = fps
        
        out = cv2.VideoWriter(output_path, fourcc, actual_fps, (width, height))
        
        for frame in frames:
            # Frame là RGB (từ Gradio) -> convert sang BGR cho OpenCV
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            out.write(frame_bgr)
        
        out.release()
        return True
    
    def save_debug_video(self, source_path: str, num_frames: int, fps: float) -> str:
        timestamp = time.strftime("%H%M%S")
        debug_path = os.path.join(
            self.debug_dir, 
            f"webcam_{timestamp}_{num_frames}frames_{fps:.1f}fps.mp4"
        )
        shutil.copy(source_path, debug_path)
        print(f"Debug video: {debug_path} (frames={num_frames}, fps={fps:.1f})")
        return debug_path
