"""
OpenCV Camera Capture - High FPS camera capture (30 FPS)
"""
import cv2
import numpy as np
import threading
import time
import tempfile
import os
from typing import List, Optional, Tuple


class CameraCapture:
    """OpenCV camera capture với 30 FPS"""
    
    def __init__(self, camera_index: int = 0, target_fps: int = 30):
        self.camera_index = camera_index
        self.target_fps = target_fps
        self.cap: Optional[cv2.VideoCapture] = None
        
        # State
        self.is_running = False
        self.is_recording = False
        self.current_frame: Optional[np.ndarray] = None
        self.frame_buffer: List[np.ndarray] = []
        self.record_start_time: float = 0
        
        # Thread
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Debug
        self.debug_dir = os.path.join(os.path.dirname(__file__), "debug_videos")
        os.makedirs(self.debug_dir, exist_ok=True)
    
    def start(self) -> bool:
        """Bắt đầu capture camera"""
        if self.is_running:
            return True
        
        # Dùng DirectShow backend thay vì MSMF (fix lỗi Windows)
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            # Thử lại với backend mặc định
            print(f"DirectShow failed, trying default backend...")
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                print(f"Cannot open camera {self.camera_index}")
                return False
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Giảm buffer để giảm lag
        
        self.is_running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        
        # Warmup: đợi camera sẵn sàng
        print("Waiting for camera warmup...")
        time.sleep(1.0)  # Cho camera khởi động
        
        # Test read
        for i in range(10):
            ret, test_frame = self.cap.read()
            if ret and test_frame is not None:
                print(f"Camera ready! Frame shape: {test_frame.shape}")
                with self._lock:
                    self.current_frame = cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB)
                break
            time.sleep(0.1)
        else:
            print("WARNING: Camera not returning frames!")
        
        print(f"Camera started: {self.get_actual_fps():.1f} FPS")
        return True
    
    def stop(self):
        """Dừng capture camera"""
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def _capture_loop(self):
        """Loop capture frames liên tục"""
        while self.is_running and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            with self._lock:
                self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                if self.is_recording:
                    self.frame_buffer.append(frame.copy())
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Lấy frame hiện tại (RGB)"""
        with self._lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
        return None
    
    def get_actual_fps(self) -> float:
        """Lấy FPS thực tế của camera"""
        if self.cap:
            return self.cap.get(cv2.CAP_PROP_FPS)
        return 0
    
    def start_recording(self):
        """Bắt đầu recording frames"""
        with self._lock:
            self.frame_buffer.clear()
            self.is_recording = True
            self.record_start_time = time.time()
    
    def stop_recording(self) -> Tuple[List[np.ndarray], float]:
        """Dừng recording và trả về frames"""
        with self._lock:
            self.is_recording = False
            frames = list(self.frame_buffer)
            elapsed = time.time() - self.record_start_time
            actual_fps = len(frames) / elapsed if elapsed > 0 else 0
            self.frame_buffer.clear()
        return frames, actual_fps
    
    def record_segment(self, duration: float) -> Tuple[List[np.ndarray], float]:
        """Record một segment với duration nhất định"""
        self.start_recording()
        time.sleep(duration)
        return self.stop_recording()
    
    def save_video(self, frames: List[np.ndarray], output_path: str, fps: float = 30.0) -> bool:
        """Lưu frames thành video file"""
        if not frames:
            return False
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
        return True
    
    def save_debug_video(self, frames: List[np.ndarray], fps: float) -> str:
        """Lưu debug video với timestamp"""
        timestamp = time.strftime("%H%M%S")
        debug_path = os.path.join(
            self.debug_dir,
            f"desktop_{timestamp}_{len(frames)}frames_{fps:.1f}fps.mp4"
        )
        self.save_video(frames, debug_path, fps)
        print(f"Debug video: {debug_path}")
        return debug_path
    
    def frames_to_temp_video(self, frames: List[np.ndarray], fps: float) -> str:
        """Tạo video tạm từ frames, trả về path"""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp_path = tmp.name
        self.save_video(frames, tmp_path, fps)
        return tmp_path
