"""
Gradio UI cho hệ thống nhận dạng ngôn ngữ ký hiệu tự động
"""
import gradio as gr
import cv2
import numpy as np
from typing import Tuple, Optional
import os
import sys

# Thêm path để import các module khác
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils.video_processor import VideoProcessor
    from ai.model import SignLanguageModel
except ImportError:
    # Fallback nếu chưa có các module này
    VideoProcessor = None
    SignLanguageModel = None


class SignLanguageRecognitionUI:
    """Class quản lý UI Gradio cho hệ thống nhận dạng ngôn ngữ ký hiệu"""
    
    def __init__(self):
        self.model = None
        self.video_processor = None
        self.initialize_model()
    
    def initialize_model(self):
        """Khởi tạo model nhận dạng"""
        try:
            if SignLanguageModel:
                self.model = SignLanguageModel()
            if VideoProcessor:
                self.video_processor = VideoProcessor()
        except Exception as e:
            print(f"Cảnh báo: Không thể khởi tạo model: {e}")
    
    def process_video(self, video: Optional[str]) -> Tuple[Optional[np.ndarray], str]:
        """
        Xử lý video và nhận dạng ngôn ngữ ký hiệu
        
        Args:
            video: Đường dẫn đến file video hoặc None
            
        Returns:
            Tuple[frame đã xử lý, text kết quả]
        """
        if video is None:
            return None, "Vui lòng tải lên video hoặc sử dụng webcam"
        
        try:
            # Đọc video
            cap = cv2.VideoCapture(video)
            if not cap.isOpened():
                return None, "Không thể đọc video"
            
            # Đọc frame đầu tiên
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                return None, "Không thể đọc frame từ video"
            
            # Xử lý frame
            processed_frame = self.process_frame(frame)
            
            # Nhận dạng (giả lập nếu chưa có model)
            if self.model:
                result_text = self.model.predict(frame)
            else:
                result_text = self.mock_recognition(frame)
            
            return processed_frame, result_text
            
        except Exception as e:
            return None, f"Lỗi xử lý video: {str(e)}"
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Xử lý frame để highlight vùng tay
        
        Args:
            frame: Frame video gốc
            
        Returns:
            Frame đã xử lý
        """
        if self.video_processor:
            return self.video_processor.process_frame(frame)
        
        # Xử lý cơ bản nếu chưa có processor
        # Chuyển sang HSV để detect màu da
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Tạo mask cho màu da (cần điều chỉnh theo điều kiện ánh sáng)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Áp dụng mask lên frame gốc
        result = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Vẽ contour của tay
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(result, [largest_contour], -1, (0, 255, 0), 2)
        
        return result
    
    def process_webcam(self, frame: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], str]:
        """
        Xử lý frame từ webcam
        
        Args:
            frame: Frame từ webcam hoặc None
            
        Returns:
            Tuple[frame đã xử lý, text kết quả]
        """
        if frame is None:
            return None, "Đang chờ dữ liệu từ webcam..."
        
        try:
            # Xử lý frame
            processed_frame = self.process_frame(frame)
            
            # Nhận dạng
            if self.model:
                result_text = self.model.predict(frame)
            else:
                result_text = self.mock_recognition(frame)
            
            return processed_frame, result_text
            
        except Exception as e:
            return None, f"Lỗi xử lý webcam: {str(e)}"
    
    def mock_recognition(self, frame: np.ndarray) -> str:
        """
        Mock recognition function khi chưa có model thật
        
        Args:
            frame: Frame video
            
        Returns:
            Text kết quả giả lập
        """
        # Đây là hàm giả lập, sẽ được thay thế bằng model thật
        height, width = frame.shape[:2]
        hand_detected = self.detect_hand_region(frame)
        
        if hand_detected:
            return f"Đã phát hiện cử chỉ tay\nKích thước frame: {width}x{height}\n[Model thật sẽ được tích hợp sau]"
        else:
            return "Không phát hiện được cử chỉ tay. Vui lòng đảm bảo tay được hiển thị rõ trong khung hình."
    
    def detect_hand_region(self, frame: np.ndarray) -> bool:
        """
        Phát hiện vùng tay trong frame
        
        Args:
            frame: Frame video
            
        Returns:
            True nếu phát hiện được tay
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Kiểm tra xem có đủ pixel màu da không
        skin_pixels = np.sum(mask > 0)
        total_pixels = mask.shape[0] * mask.shape[1]
        ratio = skin_pixels / total_pixels
        
        return ratio > 0.05  # Ít nhất 5% pixel là màu da


def create_ui():
    """Tạo giao diện Gradio"""
    ui = SignLanguageRecognitionUI()
    
    with gr.Blocks(title="Hệ thống nhận dạng ngôn ngữ ký hiệu") as demo:
        gr.Markdown(
            """
            # 🤟 Hệ thống nhận dạng ngôn ngữ ký hiệu tự động
            
            Hệ thống này giúp nhận dạng và dịch các cử chỉ ngôn ngữ ký hiệu thành văn bản.
            
            **Cách sử dụng:**
            1. Tải lên video hoặc sử dụng webcam
            2. Đảm bảo tay được hiển thị rõ trong khung hình
            3. Hệ thống sẽ tự động nhận dạng và hiển thị kết quả
            """
        )
        
        with gr.Tabs():
            # Tab 1: Upload video
            with gr.Tab("📹 Tải video lên"):
                with gr.Row():
                    with gr.Column():
                        video_input = gr.Video(
                            label="Tải video lên",
                            sources=["upload"]
                        )
                        video_btn = gr.Button("Nhận dạng", variant="primary", size="lg")
                    
                    with gr.Column():
                        video_output = gr.Image(label="Frame đã xử lý")
                        video_result = gr.Textbox(
                            label="Kết quả nhận dạng",
                            lines=5,
                            interactive=False
                        )
                
                video_btn.click(
                    fn=ui.process_video,
                    inputs=video_input,
                    outputs=[video_output, video_result]
                )
            
            # Tab 2: Webcam
            with gr.Tab("📷 Webcam"):
                with gr.Row():
                    with gr.Column():
                        webcam_input = gr.Image(
                            label="Webcam",
                            sources=["webcam"],
                            type="numpy"
                        )
                        webcam_btn = gr.Button("Nhận dạng", variant="primary", size="lg")
                    
                    with gr.Column():
                        webcam_output = gr.Image(label="Frame đã xử lý")
                        webcam_result = gr.Textbox(
                            label="Kết quả nhận dạng",
                            lines=5,
                            interactive=False
                        )
                
                webcam_btn.click(
                    fn=ui.process_webcam,
                    inputs=webcam_input,
                    outputs=[webcam_output, webcam_result]
                )
            
            # Tab 3: Thông tin
            with gr.Tab("ℹ️ Thông tin"):
                gr.Markdown(
                    """
                    ## Về hệ thống
                    
                    Hệ thống nhận dạng ngôn ngữ ký hiệu tự động sử dụng:
                    - **Computer Vision**: Phát hiện và theo dõi cử chỉ tay
                    - **Deep Learning**: Nhận dạng và phân loại các ký hiệu
                    - **NLP**: Dịch các ký hiệu thành văn bản
                    
                    ## Hướng dẫn sử dụng
                    
                    1. **Tải video**: Chọn file video từ máy tính của bạn
                    2. **Webcam**: Sử dụng webcam để nhận dạng real-time
                    3. Đảm bảo ánh sáng đủ và tay được hiển thị rõ
                    4. Giữ tay trong khung hình và thực hiện cử chỉ
                    
                    ## Lưu ý
                    
                    - Hệ thống hoạt động tốt nhất với ánh sáng tự nhiên
                    - Nền đơn giản giúp tăng độ chính xác
                    - Đảm bảo tay được hiển thị đầy đủ trong khung hình
                    """
                )
        
        # Footer
        gr.Markdown(
            """
            ---
            *Hệ thống nhận dạng ngôn ngữ ký hiệu tự động - Phiên bản 1.0*
            """
        )
    
    return demo


if __name__ == "__main__":
    demo = create_ui()
    # Tạo theme tùy chỉnh
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="cyan",
    )
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=theme
    )

