"""
Desktop App - Continuous parallel prediction
Gửi predict mỗi 0.5s, không đợi response - chạy song song
"""
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import time
import os
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from desktop.camera_capture import CameraCapture
from ui.api_client import api_client
from ui.config import CONFIDENCE_THRESHOLD


class DesktopApp:
    """Desktop UI với parallel continuous prediction"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Sign Language Recognition - Desktop (Parallel)")
        self.root.geometry("1000x700")
        self.root.configure(bg="#1a1a2e")
        
        # Camera
        self.camera = CameraCapture(camera_index=0, target_fps=30)
        
        # State
        self.is_running = True
        self.is_continuous = False
        self.prediction_history = []
        
        # Parallel prediction settings
        self.window_seconds = 2.5  # Mỗi segment = 2.5 giây
        self.stride_seconds = 0.5  # Gửi mỗi 0.5 giây
        self.executor = ThreadPoolExecutor(max_workers=5)  # 5 requests song song
        self.pending_requests = 0
        
        # Frame buffer với timestamps
        self.frame_buffer = deque(maxlen=300)  # ~10 giây ở 30fps
        self.buffer_lock = threading.Lock()
        
        # Build UI
        self._build_ui()
        
        # Start camera
        self.camera.start()
        self._update_preview()
        self._collect_frames()  # Liên tục thu thập frames
        
        # Check API
        self._check_api()
    
    def _build_ui(self):
        """Xây dựng giao diện"""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left: Camera preview
        left_frame = ttk.LabelFrame(main_frame, text="Camera (30 FPS)")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.canvas = tk.Canvas(left_frame, width=640, height=480, bg="black")
        self.canvas.pack(pady=5)
        
        # Controls
        ctrl_frame = ttk.Frame(left_frame)
        ctrl_frame.pack(pady=5)
        
        self.continuous_btn = ttk.Button(
            ctrl_frame, text="Start Continuous", 
            command=self._toggle_continuous
        )
        self.continuous_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_btn = ttk.Button(
            ctrl_frame, text="Clear", 
            command=self._clear_history
        )
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Settings
        settings_frame = ttk.LabelFrame(left_frame, text="Settings")
        settings_frame.pack(pady=5, fill=tk.X, padx=10)
        
        # Window size
        row1 = ttk.Frame(settings_frame)
        row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="Window (s):").pack(side=tk.LEFT)
        self.window_var = tk.DoubleVar(value=2.5)
        ttk.Scale(row1, from_=1.0, to=5.0, variable=self.window_var, 
                  orient=tk.HORIZONTAL, length=120).pack(side=tk.LEFT, padx=5)
        self.window_label = ttk.Label(row1, text="2.5s")
        self.window_label.pack(side=tk.LEFT)
        self.window_var.trace_add("write", lambda *_: self._update_settings())
        
        # Stride
        row2 = ttk.Frame(settings_frame)
        row2.pack(fill=tk.X, pady=2)
        ttk.Label(row2, text="Stride (s):").pack(side=tk.LEFT)
        self.stride_var = tk.DoubleVar(value=0.5)
        ttk.Scale(row2, from_=0.2, to=2.0, variable=self.stride_var,
                  orient=tk.HORIZONTAL, length=120).pack(side=tk.LEFT, padx=5)
        self.stride_label = ttk.Label(row2, text="0.5s")
        self.stride_label.pack(side=tk.LEFT)
        self.stride_var.trace_add("write", lambda *_: self._update_settings())
        
        # Right: Results
        right_frame = ttk.LabelFrame(main_frame, text="Results")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        ttk.Label(right_frame, text="Current Sign:").pack(anchor=tk.W, padx=10, pady=5)
        self.current_label = ttk.Label(
            right_frame, text="Waiting...", 
            font=("Arial", 28, "bold"),
            foreground="#2196F3"
        )
        self.current_label.pack(pady=10)
        
        self.status_label = ttk.Label(right_frame, text="Ready")
        self.status_label.pack(pady=5)
        
        # Request counter
        self.request_label = ttk.Label(right_frame, text="Pending: 0")
        self.request_label.pack(pady=2)
        
        ttk.Label(right_frame, text="Sequence:").pack(anchor=tk.W, padx=10, pady=5)
        self.sequence_text = tk.Text(right_frame, height=10, width=40, state=tk.DISABLED)
        self.sequence_text.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
        
        self.api_label = ttk.Label(right_frame, text="API: Checking...")
        self.api_label.pack(side=tk.BOTTOM, pady=10)
    
    def _update_settings(self):
        self.window_seconds = self.window_var.get()
        self.stride_seconds = self.stride_var.get()
        self.window_label.config(text=f"{self.window_seconds:.1f}s")
        self.stride_label.config(text=f"{self.stride_seconds:.1f}s")
    
    def _update_preview(self):
        """Cập nhật camera preview"""
        if not self.is_running:
            return
        
        frame = self.camera.get_frame()
        if frame is not None:
            image = Image.fromarray(frame)
            image = image.resize((640, 480))
            photo = ImageTk.PhotoImage(image)
            
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas._photo = photo
        
        self.root.after(33, self._update_preview)
    
    def _collect_frames(self):
        """Liên tục thu thập frames vào buffer"""
        if not self.is_running:
            return
        
        frame = self.camera.get_frame()
        if frame is not None:
            with self.buffer_lock:
                # Lưu frame với timestamp
                self.frame_buffer.append((time.time(), frame.copy()))
        
        # Thu thập mỗi ~33ms (30 FPS)
        self.root.after(33, self._collect_frames)
    
    # ============== CONTINUOUS MODE ==============
    
    def _toggle_continuous(self):
        if self.is_continuous:
            self._stop_continuous()
        else:
            self._start_continuous()
    
    def _start_continuous(self):
        self.is_continuous = True
        self.continuous_btn.config(text="Stop")
        self.status_label.config(text="Continuous - Sending every {:.1f}s".format(self.stride_seconds))
        self._send_prediction_loop()
    
    def _stop_continuous(self):
        self.is_continuous = False
        self.continuous_btn.config(text="Start Continuous")
        self.status_label.config(text="Stopped")
    
    def _send_prediction_loop(self):
        """Gửi prediction mỗi stride_seconds - KHÔNG đợi response"""
        if not self.is_continuous:
            return
        
        # Lấy frames từ buffer (window_seconds gần nhất)
        frames = self._get_window_frames()
        
        if frames and len(frames) >= 5:
            # Gửi request song song - không block
            self.pending_requests += 1
            self._update_request_count()
            
            self.executor.submit(self._predict_async, frames)
        
        # Schedule next send sau stride_seconds
        self.root.after(int(self.stride_seconds * 1000), self._send_prediction_loop)
    
    def _get_window_frames(self):
        """Lấy frames trong window_seconds gần nhất"""
        now = time.time()
        cutoff = now - self.window_seconds
        
        with self.buffer_lock:
            # Lọc frames trong window
            frames = [f for (t, f) in self.frame_buffer if t >= cutoff]
        
        return frames
    
    def _predict_async(self, frames):
        """Predict trong background thread"""
        try:
            fps = len(frames) / self.window_seconds
            
            # Tạo video tạm
            tmp_path = self.camera.frames_to_temp_video(frames, fps)
            
            # Gọi API
            result = api_client.predict_single(tmp_path)
            
            # Cleanup
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            
            # Update UI
            self.root.after(0, lambda: self._handle_result(result))
        except Exception as e:
            print(f"Predict error: {e}")
        finally:
            self.pending_requests = max(0, self.pending_requests - 1)
            self.root.after(0, self._update_request_count)
    
    def _handle_result(self, result):
        """Xử lý kết quả predict"""
        if result.get("success"):
            label = result.get("label", "Unknown")
            confidence = result.get("confidence", 0.0)
            
            self.current_label.config(text=f"{label} ({confidence*100:.0f}%)")
            
            if confidence >= CONFIDENCE_THRESHOLD:
                if not self.prediction_history or self.prediction_history[-1] != label:
                    self.prediction_history.append(label)
                    self._update_sequence()
    
    def _update_request_count(self):
        self.request_label.config(text=f"Pending: {self.pending_requests}")
    
    def _update_sequence(self):
        self.sequence_text.config(state=tk.NORMAL)
        self.sequence_text.delete(1.0, tk.END)
        self.sequence_text.insert(tk.END, " ".join(self.prediction_history))
        self.sequence_text.config(state=tk.DISABLED)
    
    def _clear_history(self):
        self.prediction_history.clear()
        self.current_label.config(text="Waiting...")
        self._update_sequence()
    
    def _check_api(self):
        def check():
            health = api_client.check_health()
            if health.get("model_loaded"):
                status = f"API: Connected | Device: {health.get('device', 'N/A')}"
            else:
                status = "API: Disconnected"
            self.root.after(0, lambda: self.api_label.config(text=status))
        
        threading.Thread(target=check, daemon=True).start()
    
    def run(self):
        try:
            self.root.mainloop()
        finally:
            self.is_running = False
            self.is_continuous = False
            self.executor.shutdown(wait=False)
            self.camera.stop()


def main():
    print("Starting Desktop Sign Language Recognition (Parallel Mode)...")
    print("Camera: OpenCV (30 FPS)")
    print("Mode: Parallel requests every 0.5s")
    
    app = DesktopApp()
    app.run()


if __name__ == "__main__":
    main()
