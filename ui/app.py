import gradio as gr

from ui.slr_controller import SLRController
from ui.config import API_BASE_URL

from ui.loader import load_content, load_css


def create_ui():
    """Tạo giao diện Gradio"""
    controller = SLRController()
    
    with gr.Blocks(title="🤟 Nhận dạng Ngôn ngữ Ký hiệu") as ui:
        
        gr.Markdown(load_content("header.md"))
        
        status_text = gr.Textbox(
            value=controller.get_api_status_text(),
            label="Trạng thái hệ thống",
            interactive=False
        )
        refresh_btn = gr.Button("🔄 Refresh Status", size="sm")
        refresh_btn.click(fn=controller.get_api_status_text, outputs=status_text)  # pylint: disable=no-member
        
        with gr.Tabs():
            with gr.Tab("📹 Upload Video"):
                _build_upload_video_tab(controller)
            
            with gr.Tab("📷 Webcam Real-time"):
                _build_webcam_tab(controller)
        
        gr.Markdown(load_content("footer.md"))
    
    return ui


def _build_upload_video_tab(controller: SLRController):
    gr.Markdown("### Tải lên video để nhận dạng")
    
    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="Video", sources=["upload"])
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
    
    predict_btn.click( # pylint: disable=no-member
        fn=controller.process_uploaded_video,
        inputs=[video_input, mode_radio],
        outputs=[result_main, result_topk, result_sequence]
    )


def _build_webcam_tab(controller: SLRController):
    gr.Markdown(load_content("webcam_instructions.md"))
    
    with gr.Row():
        with gr.Column(scale=1):
            webcam_input = gr.Image(
                label="📷 Camera (Click để bật)",
                sources=["webcam"],
                streaming=True,
                type="numpy"
            )
            with gr.Row():
                start_btn = gr.Button("▶️ Start", variant="primary", size="lg")
                stop_btn = gr.Button("⏹️ Stop", variant="secondary", size="lg")
                clear_btn = gr.Button("🗑️ Clear", variant="secondary", size="lg")
        
        with gr.Column(scale=1):
            current_sign = gr.Markdown(
                value="### 🎯 Đang chờ...",
                elem_classes=["prediction-box"]
            )
            webcam_status = gr.Textbox(
                label="📊 Trạng thái",
                value="⏸️ Click camera để bật, sau đó bấm Start",
                interactive=False
            )
            buffer_info = gr.Textbox(
                label="📦 Buffer",
                value="",
                interactive=False
            )
            full_sequence = gr.Textbox(
                label="📝 Chuỗi ký hiệu (nối liên tục)",
                value="",
                lines=3,
                interactive=False,
                elem_classes=["sequence-box"]
            )
    
    # Event handlers
    start_btn.click(fn=controller.start_recording, outputs=webcam_status) # pylint: disable=no-member
    stop_btn.click(fn=controller.stop_recording, outputs=webcam_status) # pylint: disable=no-member
    clear_btn.click(fn=controller.clear_sequence, outputs=full_sequence) # pylint: disable=no-member
    
    # Streaming - outputs: [current_sign, status, buffer_info, full_sequence]
    webcam_input.stream( # pylint: disable=no-member
        fn=controller.process_realtime_simple,
        inputs=webcam_input,
        outputs=[current_sign, webcam_status, buffer_info, full_sequence]
    )


if __name__ == "__main__":
    print("Starting Sign Language Recognition UI...")
    print(f"API Server: {API_BASE_URL}")
    
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="cyan"), # pylint: disable=no-member
        css=load_css()
    )
