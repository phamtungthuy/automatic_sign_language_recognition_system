"""
Gradio UI cho hệ thống nhận dạng ngôn ngữ ký hiệu tự động
Tích hợp với SLR API Server
"""
import gradio as gr
import cv2
import numpy as np
import requests
import tempfile
import time
import threading
from typing import Tuple, Optional, List, Dict
from collections import deque
import os
import sys

# Thêm path để import các module khác
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============== CONFIGURATION ==============
API_BASE_URL = os.getenv("SLR_API_URL", "http://localhost:8000")
API_PREDICT_URL = f"{API_BASE_URL}/api/v1/slr/predict"
API_PREDICT_TOPK_URL = f"{API_BASE_URL}/api/v1/slr/predict/topk"
API_PREDICT_CONTINUOUS_URL = f"{API_BASE_URL}/api/v1/slr/predict/continuous"
API_HEALTH_URL = f"{API_BASE_URL}/api/v1/slr/health"
API_LABELS_URL = f"{API_BASE_URL}/api/v1/slr/labels"

# Real-time settings
BUFFER_SECONDS = 2.0  # Số giây buffer trước khi gửi để predict
FPS_TARGET = 15  # Target FPS cho recording


class SignLanguageRecognitionUI:
    """Class quản lý UI Gradio cho hệ thống nhận dạng ngôn ngữ ký hiệu"""
    
    def __init__(self):
        self.api_healthy = False
        self.labels: List[str] = []
        self.prediction_history: deque = deque(maxlen=10)
        self.current_sequence: List[str] = []
        
        # Real-time webcam state
        self.is_recording = False
        self.frame_buffer: List[np.ndarray] = []
        self.last_prediction_time = 0
        
        # Check API health
        self.check_api_health()
    
    def check_api_health(self) -> Dict:
        """Kiểm tra trạng thái API server"""
        try:
            response = requests.get(API_HEALTH_URL, timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.api_healthy = data.get("model_loaded", False)
                return data
        except Exception as e:
            print(f"API health check failed: {e}")
        
        self.api_healthy = False
        return {"status": "disconnected", "model_loaded": False}
    
    def get_labels(self) -> List[str]:
        """Lấy danh sách labels từ API"""
        try:
            response = requests.get(API_LABELS_URL, timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.labels = data.get("labels", [])
                return self.labels
        except Exception:
            pass
        return []
    
    def predict_video(self, video_path: str, top_k: int = 5) -> Dict:
        """Gọi API predict cho video file"""
        try:
            with open(video_path, 'rb') as f:
                files = {"file": ("video.mp4", f, "video/mp4")}
                response = requests.post(
                    f"{API_PREDICT_TOPK_URL}?k={top_k}",
                    files=files,
                    timeout=60
                )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"success": False, "error": response.text}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def predict_continuous(self, video_path: str, window_seconds: float = 2.0, 
                          stride_seconds: float = 1.0) -> Dict:
        """Gọi API predict continuous cho video dài"""
        try:
            # Convert to compatible format if needed
            compatible_path = self.ensure_video_compatible(video_path)
            
            with open(compatible_path, 'rb') as f:
                files = {"file": ("video.mp4", f, "video/mp4")}
                params = {
                    "window_seconds": window_seconds,
                    "stride_seconds": stride_seconds,
                    "min_confidence": 0.3
                }
                response = requests.post(
                    API_PREDICT_CONTINUOUS_URL,
                    files=files,
                    params=params,
                    timeout=120
                )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"success": False, "error": response.text}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def ensure_video_compatible(self, video_path: str) -> str:
        """
        Đảm bảo video có format compatible với API
        Nếu video không đọc được, convert bằng OpenCV
        """
        # Thử đọc video trước
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            return video_path  # Trả về path gốc, để API xử lý lỗi
        
        # Đọc thử 1 frame
        ret, _ = cap.read()
        cap.release()
        
        if ret:
            return video_path  # Video đọc được, dùng trực tiếp
        
        # Nếu không đọc được, thử convert (hiếm khi xảy ra)
        return video_path
    
    def predict_from_frames(self, frames: List[np.ndarray]) -> Dict:
        """Convert frames to video và predict"""
        if not frames:
            return {"success": False, "error": "No frames"}
        
        # Tạo video tạm từ frames
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(tmp_path, fourcc, FPS_TARGET, (width, height))
            
            for frame in frames:
                # Chuyển từ RGB sang BGR cho OpenCV
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame
                out.write(frame_bgr)
            
            out.release()
            
            # Predict
            return self.predict_video(tmp_path, top_k=3)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    # ============== UI HANDLERS ==============
    
    def process_uploaded_video(self, video: Optional[str], mode: str) -> Tuple[str, str, str]:
        """
        Xử lý video upload
        
        Returns:
            Tuple[kết quả chính, top-k predictions, sequence (nếu continuous)]
        """
        if video is None:
            return "❌ Vui lòng tải lên video", "", ""
        
        if not self.api_healthy:
            health = self.check_api_health()
            if not self.api_healthy:
                return f"❌ API Server không sẵn sàng: {health}", "", ""
        
        try:
            if mode == "single":
                # Single prediction
                result = self.predict_video(video, top_k=5)
                
                if result.get("success"):
                    predictions = result.get("predictions", [])
                    
                    # Format main result
                    if predictions:
                        main = predictions[0]
                        main_text = f"🎯 **{main['label']}** (confidence: {main['confidence']*100:.1f}%)"
                    else:
                        main_text = "Không nhận dạng được"
                    
                    # Format top-k
                    topk_text = "\n".join([
                        f"{i+1}. {p['label']} - {p['confidence']*100:.1f}%"
                        for i, p in enumerate(predictions)
                    ])
                    
                    return main_text, topk_text, ""
                else:
                    return f"❌ Lỗi: {result.get('error', 'Unknown')}", "", ""
            
            else:
                # Continuous prediction
                result = self.predict_continuous(video, window_seconds=2.0, stride_seconds=0.5)
                
                if result.get("success"):
                    sequence = result.get("sequence", [])
                    sequence_text = result.get("sequence_text", "")
                    segments = result.get("segments", [])
                    
                    main_text = f"🎬 Nhận dạng được {len(sequence)} ký hiệu"
                    
                    # Format segments
                    segments_text = "\n".join([
                        f"[{s['start_frame']}-{s['end_frame']}] {s['label']} ({s['confidence']*100:.1f}%)"
                        for s in segments[:10]  # Chỉ show 10 segments đầu
                    ])
                    if len(segments) > 10:
                        segments_text += f"\n... và {len(segments)-10} segments khác"
                    
                    return main_text, segments_text, f"📝 Chuỗi: {sequence_text}"
                else:
                    return f"❌ Lỗi: {result.get('error', 'Unknown')}", "", ""
        
        except Exception as e:
            return f"❌ Lỗi: {str(e)}", "", ""
    
    def process_webcam_frame(self, frame: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], str, str]:
        """
        Xử lý single frame từ webcam (manual capture)
        """
        if frame is None:
            return None, "📷 Đang chờ webcam...", ""
        
        # Add frame to buffer
        self.frame_buffer.append(frame.copy())
        
        # Nếu đủ frames (khoảng 2 giây ở 15fps = 30 frames)
        min_frames = int(BUFFER_SECONDS * FPS_TARGET)
        
        if len(self.frame_buffer) >= min_frames:
            # Lấy frames và predict
            frames_to_predict = list(self.frame_buffer)
            self.frame_buffer.clear()
            
            result = self.predict_from_frames(frames_to_predict)
            
            if result.get("success"):
                predictions = result.get("predictions", [])
                if predictions:
                    main = predictions[0]
                    self.prediction_history.append(main['label'])
                    
                    # Build sequence từ history (remove duplicates)
                    sequence = []
                    for label in self.prediction_history:
                        if not sequence or sequence[-1] != label:
                            sequence.append(label)
                    self.current_sequence = sequence[-5:]  # Keep last 5
                    
                    result_text = f"🎯 **{main['label']}** ({main['confidence']*100:.1f}%)"
                    sequence_text = " → ".join(self.current_sequence) if self.current_sequence else ""
                    
                    return frame, result_text, f"📝 {sequence_text}"
            
            return frame, "⏳ Đang xử lý...", ""
        
        # Hiển thị progress
        progress = len(self.frame_buffer) / min_frames * 100
        return frame, f"📹 Recording: {progress:.0f}% ({len(self.frame_buffer)}/{min_frames} frames)", ""
    
    def start_realtime(self) -> str:
        """Bắt đầu recording real-time"""
        self.is_recording = True
        self.frame_buffer.clear()
        self.prediction_history.clear()
        self.current_sequence.clear()
        return "🔴 Đang recording... Giữ tay trước camera"
    
    def stop_realtime(self) -> str:
        """Dừng recording"""
        self.is_recording = False
        return "⏹️ Đã dừng recording"
    
    def clear_sequence(self) -> str:
        """Xóa sequence hiện tại"""
        self.prediction_history.clear()
        self.current_sequence.clear()
        self.frame_buffer.clear()
        return ""
    
    def get_status(self) -> str:
        """Lấy trạng thái hệ thống"""
        health = self.check_api_health()
        
        if health.get("model_loaded"):
            return f"✅ API Server sẵn sàng | Device: {health.get('device', 'N/A')} | Classes: {health.get('num_classes', 0)}"
        elif health.get("status") == "healthy":
            return "⚠️ API Server đang chạy nhưng model chưa load"
        else:
            return "❌ Không thể kết nối đến API Server"


def create_ui():
    """Tạo giao diện Gradio"""
    ui = SignLanguageRecognitionUI()
    
    # Custom CSS
    custom_css = """
    .prediction-box {
        font-size: 24px !important;
        font-weight: bold !important;
        text-align: center !important;
        padding: 20px !important;
    }
    .sequence-box {
        font-size: 18px !important;
        color: #2196F3 !important;
    }
    """
    
    with gr.Blocks(
        title="🤟 Nhận dạng Ngôn ngữ Ký hiệu",
        css=custom_css,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="cyan")
    ) as demo:
        
        # Header
        gr.Markdown(
            """
            # 🤟 Hệ thống Nhận dạng Ngôn ngữ Ký hiệu Việt Nam
            
            Sử dụng model **ConvNeXt-Transformer** để nhận dạng 100 ký hiệu ngôn ngữ ký hiệu Việt Nam.
            """
        )
        
        # Status bar
        status_text = gr.Textbox(
            value=ui.get_status(),
            label="Trạng thái hệ thống",
            interactive=False
        )
        refresh_btn = gr.Button("🔄 Refresh Status", size="sm")
        refresh_btn.click(fn=ui.get_status, outputs=status_text)
        
        with gr.Tabs():
            # ============== TAB 1: Upload Video ==============
            with gr.Tab("📹 Upload Video"):
                gr.Markdown("### Tải lên video để nhận dạng")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        video_input = gr.Video(
                            label="Video",
                            sources=["upload"]
                        )
                        mode_radio = gr.Radio(
                            choices=["single", "continuous"],
                            value="single",
                            label="Chế độ nhận dạng",
                            info="single: 1 ký hiệu | continuous: chuỗi ký hiệu"
                        )
                        predict_btn = gr.Button("🚀 Nhận dạng", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        result_main = gr.Markdown(
                            value="Kết quả sẽ hiển thị ở đây",
                            elem_classes=["prediction-box"]
                        )
                        result_topk = gr.Textbox(
                            label="Chi tiết predictions",
                            lines=6,
                            interactive=False
                        )
                        result_sequence = gr.Textbox(
                            label="Chuỗi ký hiệu (continuous mode)",
                            lines=2,
                            interactive=False,
                            elem_classes=["sequence-box"]
                        )
                
                predict_btn.click(
                    fn=ui.process_uploaded_video,
                    inputs=[video_input, mode_radio],
                    outputs=[result_main, result_topk, result_sequence]
                )
            
            # ============== TAB 2: Webcam Real-time ==============
            with gr.Tab("📷 Webcam Real-time"):
                gr.Markdown(
                    """
                    ### Nhận dạng real-time qua webcam
                    
                    1. Bấm **Start** để bắt đầu
                    2. Thực hiện ký hiệu trước camera
                    3. Hệ thống sẽ tự động nhận dạng mỗi 2 giây
                    """
                )
                
                with gr.Row():
                    with gr.Column(scale=1):
                        webcam = gr.Image(
                            label="Webcam",
                            sources=["webcam"],
                            streaming=True,
                            type="numpy"
                        )
                        with gr.Row():
                            start_btn = gr.Button("▶️ Start", variant="primary")
                            stop_btn = gr.Button("⏹️ Stop", variant="secondary")
                            clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                    
                    with gr.Column(scale=1):
                        webcam_status = gr.Textbox(
                            label="Trạng thái",
                            value="Chưa bắt đầu",
                            interactive=False
                        )
                        webcam_result = gr.Markdown(
                            value="Đang chờ...",
                            elem_classes=["prediction-box"]
                        )
                        webcam_sequence = gr.Textbox(
                            label="Chuỗi đã nhận dạng",
                            value="",
                            lines=2,
                            interactive=False,
                            elem_classes=["sequence-box"]
                        )
                
                # Button handlers
                start_btn.click(fn=ui.start_realtime, outputs=webcam_status)
                stop_btn.click(fn=ui.stop_realtime, outputs=webcam_status)
                clear_btn.click(fn=ui.clear_sequence, outputs=webcam_sequence)
                
                # Streaming handler
                webcam.stream(
                    fn=ui.process_webcam_frame,
                    inputs=webcam,
                    outputs=[webcam, webcam_result, webcam_sequence]
                )
            
            # ============== TAB 3: Demo Videos ==============
            with gr.Tab("🎬 Demo Sentences"):
                gr.Markdown(
                    """
                    ### Demo video câu ghép
                    
                    Các video demo được tạo từ việc ghép các ký hiệu đơn lẻ thành câu có nghĩa.
                    """
                )
                
                with gr.Row():
                    demo_video = gr.Video(label="Demo Video")
                    demo_result = gr.Textbox(
                        label="Kết quả nhận dạng continuous",
                        lines=8,
                        interactive=False
                    )
                
                # Example sentences
                gr.Examples(
                    examples=[
                        ["output/sentence_videos/Tôi_Ăn_Cá.mp4"],
                        ["output/sentence_videos/Hôm-nay_Tôi_Đi_Bệnh-viện.mp4"],
                        ["output/sentence_videos/Chúng-ta_Cần_Giúp.mp4"],
                    ],
                    inputs=demo_video,
                    label="Câu mẫu"
                )
            
            # ============== TAB 4: Thông tin ==============
            with gr.Tab("ℹ️ Thông tin"):
                gr.Markdown(
                    """
                    ## Về hệ thống
                    
                    Hệ thống sử dụng kiến trúc **ConvNeXt-Tiny + Transformer** để nhận dạng ngôn ngữ ký hiệu Việt Nam.
                    
                    ### Thông số kỹ thuật
                    - **Model**: ConvNeXt-Tiny (pretrained ImageNet) + Transformer Encoder
                    - **Input**: Video 16 frames @ 224x224
                    - **Output**: 100 classes ngôn ngữ ký hiệu
                    
                    ### Chế độ nhận dạng
                    
                    | Chế độ | Mô tả |
                    |--------|-------|
                    | **Single** | Nhận dạng 1 ký hiệu từ toàn bộ video |
                    | **Continuous** | Nhận dạng chuỗi ký hiệu từ video dài |
                    
                    ### API Endpoints
                    
                    ```
                    POST /api/v1/slr/predict       - Nhận dạng đơn
                    POST /api/v1/slr/predict/topk  - Top-k predictions
                    POST /api/v1/slr/predict/continuous - Nhận dạng chuỗi
                    GET  /api/v1/slr/health        - Health check
                    GET  /api/v1/slr/labels        - Danh sách labels
                    ```
                    
                    ### Hướng dẫn
                    
                    1. **Ánh sáng**: Đảm bảo đủ ánh sáng, tránh ngược sáng
                    2. **Vị trí**: Đặt tay trong khung hình, nền đơn giản
                    3. **Tốc độ**: Thực hiện ký hiệu với tốc độ bình thường
                    """
                )
                
                # Show available labels
                with gr.Accordion("📋 Danh sách 100 ký hiệu", open=False):
                    labels_text = gr.Textbox(
                        value="Loading...",
                        lines=10,
                        interactive=False
                    )
                    load_labels_btn = gr.Button("Load Labels")
                    
                    def load_labels():
                        labels = ui.get_labels()
                        if labels:
                            return ", ".join(sorted(labels))
                        return "Không thể load labels. Kiểm tra API server."
                    
                    load_labels_btn.click(fn=load_labels, outputs=labels_text)
        
        # Footer
        gr.Markdown(
            """
            ---
            🤟 *Hệ thống nhận dạng ngôn ngữ ký hiệu Việt Nam - v2.0* | 
            API: `http://localhost:8000/docs`
            """
        )
    
    return demo


if __name__ == "__main__":
    print("🚀 Starting Sign Language Recognition UI...")
    print(f"📡 API Server: {API_BASE_URL}")
    
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
