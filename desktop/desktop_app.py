"""
Desktop App - Tkinter UI với OpenCV camera (30 FPS)
Continuous prediction mode - tự động predict mỗi N giây
"""
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import time
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from desktop.camera_capture import CameraCapture
from ui.api_client import api_client
from ui.config import CONFIDENCE_THRESHOLD


class DesktopApp:
    """Desktop UI với camera 30 FPS và continuous prediction"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Sign Language Recognition - Desktop")
        self.root.geometry("1000x650")
        self.root.configure(bg="#1a1a2e")
        
        # Camera
        self.camera = CameraCapture(camera_index=0, target_fps=30)
        
        # State
        self.is_running = True
        self.is_continuous = False  # Continuous mode
        self.is_predicting = False  # Đang chờ API response
        self.prediction_history = []
        self.current_prediction = ""
        self.record_duration = 2.5  # seconds per segment
        
        # Build UI
        self._build_ui()
        
        # Start camera
        self.camera.start()
        self._update_preview()
        
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
        
        # Buttons row 1: Mode selection
        mode_frame = ttk.Frame(left_frame)
        mode_frame.pack(pady=5)
        
        ttk.Label(mode_frame, text="Mode:").pack(side=tk.LEFT, padx=5)
        
        self.continuous_btn = ttk.Button(
            mode_frame, text="Start Continuous", 
            command=self._toggle_continuous
        )
        self.continuous_btn.pack(side=tk.LEFT, padx=5)
        
        self.single_btn = ttk.Button(
            mode_frame, text="Single Predict", 
            command=self._single_predict
        )
        self.single_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_btn = ttk.Button(
            mode_frame, text="Clear", 
            command=self._clear_history
        )
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Duration slider
        duration_frame = ttk.Frame(left_frame)
        duration_frame.pack(pady=5)
        
        ttk.Label(duration_frame, text="Segment (s):").pack(side=tk.LEFT)
        self.duration_var = tk.DoubleVar(value=2.5)
        self.duration_slider = ttk.Scale(
            duration_frame, from_=1.0, to=5.0, 
            variable=self.duration_var, orient=tk.HORIZONTAL, length=150
        )
        self.duration_slider.pack(side=tk.LEFT, padx=5)
        self.duration_label = ttk.Label(duration_frame, text="2.5s")
        self.duration_label.pack(side=tk.LEFT)
        self.duration_var.trace_add("write", self._update_duration_label)
        
        # Right: Results
        right_frame = ttk.LabelFrame(main_frame, text="Results")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        ttk.Label(right_frame, text="Current Sign:").pack(anchor=tk.W, padx=10, pady=5)
        self.current_label = ttk.Label(
            right_frame, text="Waiting...", 
            font=("Arial", 24, "bold")
        )
        self.current_label.pack(pady=10)
        
        self.status_label = ttk.Label(right_frame, text="Ready - Click 'Start Continuous' to begin")
        self.status_label.pack(pady=5)
        
        # Progress bar for recording
        self.progress = ttk.Progressbar(right_frame, mode='determinate', length=200)
        self.progress.pack(pady=5)
        
        ttk.Label(right_frame, text="Sequence:").pack(anchor=tk.W, padx=10, pady=5)
        self.sequence_text = tk.Text(right_frame, height=8, width=40, state=tk.DISABLED)
        self.sequence_text.pack(padx=10, pady=5, fill=tk.X)
        
        self.api_label = ttk.Label(right_frame, text="API: Checking...")
        self.api_label.pack(side=tk.BOTTOM, pady=10)
    
    def _update_duration_label(self, *args):
        self.duration_label.config(text=f"{self.duration_var.get():.1f}s")
        self.record_duration = self.duration_var.get()
    
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
    
    # ============== CONTINUOUS MODE ==============
    
    def _toggle_continuous(self):
        """Toggle continuous prediction mode"""
        if self.is_continuous:
            self._stop_continuous()
        else:
            self._start_continuous()
    
    def _start_continuous(self):
        """Bắt đầu continuous prediction"""
        self.is_continuous = True
        self.continuous_btn.config(text="Stop Continuous")
        self.single_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Continuous mode - Recording...")
        self._continuous_loop()
    
    def _stop_continuous(self):
        """Dừng continuous prediction"""
        self.is_continuous = False
        self.continuous_btn.config(text="Start Continuous")
        self.single_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Stopped")
        self.progress['value'] = 0
    
    def _continuous_loop(self):
        """Loop liên tục: record segment -> predict -> repeat"""
        if not self.is_continuous:
            return
        
        # Start recording
        self.camera.start_recording()
        self._animate_progress(0)
    
    def _animate_progress(self, elapsed_ms):
        """Animate progress bar during recording"""
        if not self.is_continuous:
            return
        
        total_ms = int(self.record_duration * 1000)
        progress_pct = min(100, (elapsed_ms / total_ms) * 100)
        self.progress['value'] = progress_pct
        
        remaining = (total_ms - elapsed_ms) / 1000
        self.status_label.config(text=f"Recording... ({remaining:.1f}s)")
        
        if elapsed_ms >= total_ms:
            # Done recording, predict
            self._do_predict_and_continue()
        else:
            # Continue animation
            self.root.after(100, lambda: self._animate_progress(elapsed_ms + 100))
    
    def _do_predict_and_continue(self):
        """Stop recording, predict, then continue if still in continuous mode"""
        if not self.is_continuous:
            return
        
        frames, fps = self.camera.stop_recording()
        
        if not frames:
            # No frames, restart
            self.root.after(100, self._continuous_loop)
            return
        
        self.status_label.config(text=f"Predicting... ({len(frames)} frames, {fps:.1f} FPS)")
        self.progress['value'] = 100
        
        # Predict in background
        threading.Thread(
            target=self._predict_and_restart,
            args=(frames, fps),
            daemon=True
        ).start()
    
    def _predict_and_restart(self, frames, fps):
        """Predict và restart continuous loop"""
        try:
            self.camera.save_debug_video(frames, fps)
            tmp_path = self.camera.frames_to_temp_video(frames, fps)
            result = api_client.predict_single(tmp_path)
            
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            
            # Update UI và restart loop
            self.root.after(0, lambda: self._update_result_and_continue(result))
        except Exception as e:
            self.root.after(0, lambda: self.status_label.config(text=f"Error: {e}"))
            # Restart anyway
            self.root.after(500, self._continuous_loop)
    
    def _update_result_and_continue(self, result):
        """Update result và tiếp tục continuous loop"""
        self._update_result(result)
        
        if self.is_continuous:
            # Đợi 1 chút rồi tiếp tục
            self.root.after(200, self._continuous_loop)
    
    # ============== SINGLE PREDICT ==============
    
    def _single_predict(self):
        """Single prediction - record once"""
        self.camera.start_recording()
        self.single_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Recording...")
        self._animate_single_progress(0)
    
    def _animate_single_progress(self, elapsed_ms):
        """Animate for single mode"""
        total_ms = int(self.record_duration * 1000)
        progress_pct = min(100, (elapsed_ms / total_ms) * 100)
        self.progress['value'] = progress_pct
        
        if elapsed_ms >= total_ms:
            self._stop_single_and_predict()
        else:
            self.root.after(100, lambda: self._animate_single_progress(elapsed_ms + 100))
    
    def _stop_single_and_predict(self):
        """Stop single recording and predict"""
        frames, fps = self.camera.stop_recording()
        self.single_btn.config(state=tk.NORMAL)
        
        if not frames:
            self.status_label.config(text="No frames")
            return
        
        self.status_label.config(text=f"Predicting... ({len(frames)} frames)")
        
        threading.Thread(
            target=self._predict_async,
            args=(frames, fps),
            daemon=True
        ).start()
    
    def _predict_async(self, frames, fps):
        try:
            self.camera.save_debug_video(frames, fps)
            tmp_path = self.camera.frames_to_temp_video(frames, fps)
            result = api_client.predict_single(tmp_path)
            
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            
            self.root.after(0, lambda: self._update_result(result))
        except Exception as e:
            self.root.after(0, lambda: self.status_label.config(text=f"Error: {e}"))
    
    def _update_result(self, result):
        if result.get("success"):
            label = result.get("label", "Unknown")
            confidence = result.get("confidence", 0.0)
            
            self.current_prediction = f"{label} ({confidence*100:.0f}%)"
            self.current_label.config(text=self.current_prediction)
            
            if confidence >= CONFIDENCE_THRESHOLD:
                if not self.prediction_history or self.prediction_history[-1] != label:
                    self.prediction_history.append(label)
                    self._update_sequence()
            
            if not self.is_continuous:
                self.status_label.config(text="Ready")
        else:
            self.status_label.config(text=f"Error: {result.get('error', 'Unknown')}")
    
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
            self.camera.stop()


def main():
    print("Starting Desktop Sign Language Recognition...")
    print("Camera: OpenCV (30 FPS)")
    print("Modes: Continuous | Single")
    
    app = DesktopApp()
    app.run()


if __name__ == "__main__":
    main()
