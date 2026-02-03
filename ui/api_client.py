"""
SLR API Client - Gọi API backend
Tách biệt khỏi logic xử lý UI/event
"""
import requests
from typing import Dict, List

from ui.config import (
    API_PREDICT_URL, API_PREDICT_TOPK_URL, API_PREDICT_CONTINUOUS_URL,
    API_HEALTH_URL, API_LABELS_URL
)


class SLRApiClient:
    """Client để gọi SLR API backend"""
    
    def __init__(self):
        self.api_healthy = False
        self.labels: List[str] = []
    
    def check_health(self) -> Dict:
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
    
    def predict_topk(self, video_path: str, top_k: int = 5) -> Dict:
        """
        Gọi API predict/topk cho video file
        Returns: {"success": bool, "predictions": [...]}
        """
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
    
    def predict_single(self, video_path: str) -> Dict:
        """
        Gọi API /predict để nhận 1 prediction duy nhất
        Returns: {"success": bool, "label": str, "confidence": float}
        """
        try:
            with open(video_path, 'rb') as f:
                files = {"file": ("video.mp4", f, "video/mp4")}
                response = requests.post(
                    API_PREDICT_URL,
                    files=files,
                    timeout=60
                )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success") and data.get("prediction"):
                    pred = data["prediction"]
                    return {
                        "success": True,
                        "label": pred["label"],
                        "confidence": pred["confidence"]
                    }
                return {"success": False, "error": "No prediction"}
            else:
                return {"success": False, "error": response.text}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def predict_continuous(self, video_path: str, 
                          window_seconds: float = 2.0,
                          stride_seconds: float = 1.0,
                          min_confidence: float = 0.3) -> Dict:
        """
        Gọi API predict/continuous cho video dài
        Returns: {"success": bool, "segments": [...], "sequence": [...]}
        """
        try:
            with open(video_path, 'rb') as f:
                files = {"file": ("video.mp4", f, "video/mp4")}
                params = {
                    "window_seconds": window_seconds,
                    "stride_seconds": stride_seconds,
                    "min_confidence": min_confidence
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


# Global instance
api_client = SLRApiClient()
