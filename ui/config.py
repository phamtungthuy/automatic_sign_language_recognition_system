"""
Configuration constants cho UI
"""
import os

# ============== API CONFIGURATION ==============
API_BASE_URL = os.getenv("SLR_API_URL", "https://slr1.iselab.info")
API_PREDICT_URL = f"{API_BASE_URL}/api/v1/slr/predict"
API_PREDICT_TOPK_URL = f"{API_BASE_URL}/api/v1/slr/predict/topk"
API_PREDICT_CONTINUOUS_URL = f"{API_BASE_URL}/api/v1/slr/predict/continuous"
API_HEALTH_URL = f"{API_BASE_URL}/api/v1/slr/health"
API_LABELS_URL = f"{API_BASE_URL}/api/v1/slr/labels"

# ============== REAL-TIME SETTINGS ==============
BUFFER_SECONDS = 2.0  # Số giây buffer trước khi gửi để predict
FPS_TARGET = 100  # Gradio streaming thực tế ~5fps

# ============== SLIDING WINDOW SETTINGS ==============
WINDOW_SECONDS = 2.0    # Mỗi window = 2 giây video để predict
STRIDE_SECONDS = 0.5    # Gửi predict mỗi 0.5 giây (overlap 75%)

# ============== PREDICTION SETTINGS ==============
CONFIDENCE_THRESHOLD = 0.2  # Chỉ append nếu confidence > 60%
