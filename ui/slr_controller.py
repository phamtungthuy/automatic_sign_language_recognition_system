"""
SLR Controller - Xử lý event và state cho UI
Sử dụng SLRApiClient để gọi API và CameraController để quản lý camera
"""
import numpy as np
import tempfile
import os
from typing import Tuple, Optional, List, Dict
from collections import deque

from ui.api_client import api_client
from ui.config import CONFIDENCE_THRESHOLD
from ui.controllers import CameraController


class SLRController:
    """Controller xử lý event và state cho nhận dạng ngôn ngữ ký hiệu"""
    
    def __init__(self):
        # API client
        self.api = api_client
        
        # Camera controller
        debug_dir = os.path.join(os.path.dirname(__file__), "debug_videos")
        self.camera = CameraController(debug_dir=debug_dir)
        
        # Prediction state
        self.prediction_history: deque = deque(maxlen=10)
        self.current_sequence: List[str] = []
        self.current_prediction: str = ""
        self.prediction_confidence: float = 0.0
        
        # Check API health on init
        self.api.check_health()
    
    # ============== VIDEO UPLOAD HANDLERS ==============
    
    def process_uploaded_video(self, video_path: Optional[str], mode: str) -> Tuple[str, str, str]:
        """
        Xử lý video upload
        Returns: (main_result, topk_results, sequence)
        """
        if video_path is None:
            return "❌ Vui lòng tải lên video", "", ""
        
        if not self.api.api_healthy:
            self.api.check_health()
            if not self.api.api_healthy:
                return "❌ API Server không sẵn sàng", "", ""
        
        try:
            if mode == "single":
                result = self.api.predict_topk(video_path, top_k=5)
                if result.get("success"):
                    preds = result.get("predictions", [])
                    if preds:
                        main = preds[0]
                        main_text = f"## 🎯 {main['label']} ({main['confidence']*100:.1f}%)"
                        
                        topk_lines = []
                        for i, p in enumerate(preds[1:5], 2):
                            topk_lines.append(f"{i}. {p['label']} ({p['confidence']*100:.1f}%)")
                        
                        return main_text, "\n".join(topk_lines), ""
                return f"❌ {result.get('error', 'Unknown error')}", "", ""
            else:
                # Continuous mode
                result = self.api.predict_continuous(video_path)
                if result.get("success"):
                    seq = result.get("sequence", [])
                    seq_text = " → ".join(seq) if seq else "(không phát hiện ký hiệu)"
                    
                    segments = result.get("segments", [])
                    seg_lines = []
                    for s in segments[:10]:
                        seg_lines.append(f"• {s['label']} ({s['confidence']*100:.0f}%)")
                    
                    return f"## 📝 {seq_text}", "\n".join(seg_lines), seq_text
                return f"❌ {result.get('error', 'Unknown error')}", "", ""
        except Exception as e:
            return f"❌ Lỗi: {str(e)}", "", ""
    
    # ============== WEBCAM REALTIME HANDLERS ==============
    
    def start_recording(self) -> str:
        """Bắt đầu recording real-time"""
        self.camera.start_recording()
        return "🔴 Đang recording..."
    
    def stop_recording(self) -> str:
        """Dừng recording"""
        frame_count = len(self.camera.sliding_window_buffer)
        self.camera.stop_recording()
        return f"⏹️ Đã dừng. Đã thu {frame_count} frames."
    
    def clear_sequence(self) -> str:
        """Xóa chuỗi ký hiệu đã nhận dạng"""
        self.prediction_history.clear()
        self.current_sequence.clear()
        self.current_prediction = ""
        return ""
    
    def predict_from_frames(self, frames: List[np.ndarray], fps: float = 0.0, save_debug: bool = True) -> Dict:
        """
        Convert frames thành video và gọi API predict
        Returns: {"success": bool, "label": str, "confidence": float, "debug_video": str}
        """
        if not frames:
            return {"success": False, "error": "No frames"}
        
        # Tạo video tạm
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            if not self.camera.frames_to_video(frames, tmp_path, fps=fps):
                return {"success": False, "error": "Failed to create video"}
            
            # Lưu debug video
            debug_path = ""
            if save_debug:
                debug_path = self.camera.save_debug_video(tmp_path, len(frames), fps)
            
            # Gọi API
            result = self.api.predict_single(tmp_path)
            result["debug_video"] = debug_path
            result["frame_count"] = len(frames)
            return result
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def process_realtime_simple(self, frame: Optional[np.ndarray]) -> Tuple[str, str, str, str]:
        """
        Xử lý frame webcam - thu thập 2.5 giây rồi gửi predict
        
        Returns: (current_sign_md, status_text, buffer_info, full_sequence_text)
        """
        if frame is None:
            return "### 🎯 Đang chờ...", "📷 Click camera để bật", "", ""
        
        COLLECT_SECONDS = 2.5
        
        # Init window_start_time nếu chưa có
        if self.camera.window_start_time == 0:
            self.camera.window_start_time = __import__('time').time()
        
        # Thêm frame vào buffer
        self.camera.add_frame(frame)
        
        # Default values
        current_sign_md = f"### 🎯 {self.current_prediction}" if self.current_prediction else "### 🎯 Đang chờ..."
        status_text = ""
        buffer_info = ""
        full_sequence_text = ""
        
        if self.camera.is_recording:
            elapsed_time = self.camera.get_elapsed_time()
            
            # Kiểm tra có nên gửi predict không
            if self.camera.should_send(COLLECT_SECONDS):
                frames_to_predict, real_fps = self.camera.pop_frames_for_prediction()
                num_frames = len(frames_to_predict)
                
                # Gọi predict
                result = self.predict_from_frames(frames_to_predict, fps=real_fps)
                
                if result.get("success"):
                    label = result.get("label", "")
                    confidence = result.get("confidence", 0.0)
                    
                    self.current_prediction = f"{label} ({confidence*100:.0f}%)"
                    self.prediction_confidence = confidence
                    current_sign_md = f"### 🎯 {label} ({num_frames} frames, {real_fps:.1f} fps)"
                    
                    # Chỉ append nếu confidence > threshold
                    if confidence >= CONFIDENCE_THRESHOLD:
                        if not self.prediction_history or self.prediction_history[-1] != label:
                            self.prediction_history.append(label)
                        
                        while len(self.prediction_history) > 10:
                            self.prediction_history.popleft()
                    else:
                        current_sign_md = f"### ⚠️ {label} (thấp: {confidence*100:.0f}%)"
            
            # Status
            time_remaining = max(0, COLLECT_SECONDS - elapsed_time)
            buffer_info = f"Frames: {len(self.camera.sliding_window_buffer)} | Gửi sau: {time_remaining:.1f}s"
            status_text = "🔴 Đang xử lý..." if time_remaining <= 0.1 else f"🔴 Thu thập ({time_remaining:.1f}s)"
            
            if self.prediction_history:
                full_sequence_text = " ".join(list(self.prediction_history))
        else:
            status_text = "⏸️ Bấm Start để bắt đầu"
            buffer_info = self.camera.get_buffer_info()
            if self.prediction_history:
                full_sequence_text = " ".join(list(self.prediction_history))
        
        return current_sign_md, status_text, buffer_info, full_sequence_text
    
    # ============== STATUS ==============
    
    def get_api_status_text(self) -> str:
        """Lấy text hiển thị trạng thái API"""
        health = self.api.check_health()
        if health.get("model_loaded"):
            return f"✅ Connected | Device: {health.get('device', 'N/A')} | Classes: {health.get('num_classes', 0)}"
        return "❌ API không khả dụng"
