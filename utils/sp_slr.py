"""
Sign Language Recognition - Model & Preprocessing Utilities
"""
import os
import math
import pickle
import tempfile
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠️ MediaPipe not installed. Person detection disabled. Install with: pip install mediapipe")

from utils.constants import (
    SLR_MODEL_PATH,
    SLR_LABEL_MAPPING_PATH,
    SLR_NUM_CLASSES,
    SLR_TARGET_FRAMES,
)

# ============== CONFIGURATION ==============
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Person detection settings
PERSON_DETECTION_ENABLED = True  # Set to False to disable
CROP_PADDING = 0.15  # 15% padding around detected person


# ============== MODEL ARCHITECTURE ==============
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class AttentionPooling(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.Tanh(),
            nn.Linear(dim // 4, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_weights = self.attention(x)
        attn_weights = F.softmax(attn_weights, dim=1)
        pooled = torch.sum(attn_weights * x, dim=1)
        return pooled


class ConvNeXtTransformer(nn.Module):
    """
    ConvNeXt-Tiny + Transformer for Video Classification
    Input:  (B, T, C, H, W) = (B, 16, 3, 224, 224)
    Output: (B, num_classes) = (B, 100)
    """
    def __init__(self, num_classes: int = SLR_NUM_CLASSES):
        super().__init__()

        convnext = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        self.cnn = convnext.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = 768

        self.pos_encoder = PositionalEncoding(
            d_model=self.feature_dim,
            max_len=64,
            dropout=0.1
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=8,
            dim_feedforward=self.feature_dim * 4,
            dropout=0.3,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.attention_pool = AttentionPooling(self.feature_dim)

        self.fc = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(0.4),
            nn.Linear(self.feature_dim, num_classes)
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.transformer.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for m in self.attention_pool.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.cnn(x)
        x = self.pool(x)
        x = x.view(B, T, self.feature_dim)

        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = self.attention_pool(x)
        x = self.fc(x)
        return x


# ============== PERSON DETECTION (MediaPipe) ==============
class PersonDetector:
    """
    Detect person in frame using MediaPipe Holistic
    Returns bounding box around detected person/hands
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.holistic = None
        if MEDIAPIPE_AVAILABLE and PERSON_DETECTION_ENABLED:
            self.holistic = mp.solutions.holistic.Holistic(
                static_image_mode=False,
                model_complexity=0,  # 0=lite, 1=full, 2=heavy
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("✅ MediaPipe Holistic initialized for person detection")
        
        self._initialized = True
    
    def get_person_bbox(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect person and return bounding box (x1, y1, x2, y2)
        Returns None if no person detected
        """
        if self.holistic is None:
            return None
        
        h, w = frame.shape[:2]
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame.shape[2] == 3 else frame
        
        results = self.holistic.process(rgb_frame)
        
        # Collect all landmark points
        points = []
        
        # Pose landmarks (body)
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                if lm.visibility > 0.5:
                    points.append((int(lm.x * w), int(lm.y * h)))
        
        # Left hand
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                points.append((int(lm.x * w), int(lm.y * h)))
        
        # Right hand  
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                points.append((int(lm.x * w), int(lm.y * h)))
        
        if not points:
            return None
        
        # Calculate bounding box
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        
        # Add padding
        pad_x = int((x2 - x1) * CROP_PADDING)
        pad_y = int((y2 - y1) * CROP_PADDING)
        
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        
        return (x1, y1, x2, y2)
    
    def crop_person(self, frame: np.ndarray) -> np.ndarray:
        """
        Crop frame to person bounding box
        Returns original frame if no person detected
        """
        bbox = self.get_person_bbox(frame)
        
        if bbox is None:
            return frame
        
        x1, y1, x2, y2 = bbox
        
        # Ensure valid crop
        if x2 - x1 < 50 or y2 - y1 < 50:
            return frame
        
        return frame[y1:y2, x1:x2]


# Global instance
person_detector = PersonDetector()


# ============== PREPROCESSING ==============
def read_video_from_path(video_path: str, crop_person: bool = True) -> torch.Tensor:
    """
    Read video frames from file path
    If crop_person=True and MediaPipe available, crop to detected person
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Crop to person if enabled (frame is BGR here)
        if crop_person and PERSON_DETECTION_ENABLED and person_detector.holistic is not None:
            frame = person_detector.crop_person(frame)
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()
    
    if len(frames) == 0:
        raise ValueError(f"Could not read any frames from {video_path}")
    
    return torch.from_numpy(np.stack(frames, axis=0))


def read_video_from_bytes(video_bytes: bytes) -> torch.Tensor:
    """Read video from bytes (uploaded file)"""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        frames = read_video_from_path(tmp_path)
    finally:
        os.unlink(tmp_path)
    
    return frames


def downsample_frames(frames: torch.Tensor, target_frames: int = SLR_TARGET_FRAMES) -> torch.Tensor:
    """Sample target_frames from video uniformly"""
    total = frames.shape[0]
    if total >= target_frames:
        indices = torch.linspace(0, total - 1, target_frames).long()
    else:
        indices = torch.arange(total)
        pad = target_frames - total
        indices = torch.cat([indices, indices[-1].repeat(pad)])

    frames = frames[indices]

    # Resize to 224x224 if needed
    if frames.shape[1] != 224 or frames.shape[2] != 224:
        frames = frames.permute(0, 3, 1, 2).float()
        frames = F.interpolate(frames, size=(224, 224), mode='bilinear', align_corners=False)
        frames = frames.permute(0, 2, 3, 1).to(torch.uint8)

    return frames


def normalize_frames(frames: torch.Tensor) -> torch.Tensor:
    """Normalize to ImageNet mean/std"""
    frames = frames.float() / 255.0
    frames = frames.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)

    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    frames = (frames - mean) / std

    return frames


def preprocess_video(video_bytes: bytes) -> torch.Tensor:
    """Full preprocessing pipeline: bytes -> model input tensor"""
    frames = read_video_from_bytes(video_bytes)
    frames = downsample_frames(frames)
    frames = normalize_frames(frames)
    return frames.unsqueeze(0)  # Add batch dim: (1, T, C, H, W)


# ============== MODEL MANAGER (Singleton) ==============
class SLRModelManager:
    """Singleton to manage model loading and inference"""
    _instance: Optional["SLRModelManager"] = None
    
    def __new__(cls) -> "SLRModelManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self) -> None:
        if self._initialized:
            return
        
        self.model: Optional[ConvNeXtTransformer] = None
        self.label_mapping: Dict[str, int] = {}
        self.idx_to_label: Dict[int, str] = {}
        self.device = DEVICE
        self._initialized = True
    
    def load(
        self, 
        model_path: Optional[str] = None, 
        label_mapping_path: Optional[str] = None
    ) -> None:
        """Load model and label mapping"""
        # Use defaults from constants if not provided
        _model_path = model_path or str(SLR_MODEL_PATH)
        _label_path = label_mapping_path or str(SLR_LABEL_MAPPING_PATH)
        
        # Load model
        self.model = ConvNeXtTransformer(num_classes=SLR_NUM_CLASSES)
        self.model.load_state_dict(torch.load(_model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load label mapping
        with open(_label_path, 'rb') as f:
            self.label_mapping = pickle.load(f)
        self.idx_to_label = {v: k for k, v in self.label_mapping.items()}
        
        print(f"✅ Model loaded from {_model_path}")
        print(f"✅ Label mapping loaded: {len(self.idx_to_label)} classes")
        print(f"🖥️ Device: {self.device}")
    
    def is_loaded(self) -> bool:
        return self.model is not None
    
    def predict(self, video_bytes: bytes, top_k: int = 1) -> List[Dict]:
        """
        Run inference on video bytes
        Returns list of {"label": str, "confidence": float, "label_idx": int}
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load() first.")
        
        if self.model is None:
            raise RuntimeError("Model is None")
        
        frames = preprocess_video(video_bytes)
        frames = frames.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(frames)
            probs = F.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probs, k=min(top_k, SLR_NUM_CLASSES), dim=1)
            
            results: List[Dict] = []
            for prob, idx in zip(top_probs[0], top_indices[0]):
                idx_int = int(idx.item())
                results.append({
                    "label": self.idx_to_label[idx_int],
                    "confidence": round(float(prob.item()), 4),
                    "label_idx": idx_int
                })
        
        return results
    
    def _predict_frames(self, frames: torch.Tensor) -> Dict:
        """
        Predict on a batch of frames (already preprocessed)
        frames: (T, C, H, W) tensor
        """
        if self.model is None:
            raise RuntimeError("Model is None")
        
        # Add batch dimension
        frames = frames.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(frames)
            probs = F.softmax(outputs, dim=1)
            confidence, predicted = probs.max(1)
            
            idx_int = int(predicted.item())
            return {
                "label": self.idx_to_label[idx_int],
                "confidence": round(float(confidence.item()), 4),
                "label_idx": idx_int
            }
    
    def predict_continuous(
        self,
        video_bytes: bytes,
        window_seconds: float = 2.0,
        stride_seconds: float = 1.0,
        min_confidence: float = 0.5
    ) -> Dict:
        """
        Continuous sign language recognition using sliding window.
        
        Args:
            video_bytes: Raw video bytes
            window_seconds: Length of each analysis window
            stride_seconds: How much to slide between windows
            min_confidence: Minimum confidence to include prediction
        
        Returns:
            Dict with segments, sequence, total_frames
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Read all frames
        all_frames = read_video_from_bytes(video_bytes)
        total_frames = all_frames.shape[0]
        
        # Estimate FPS (assume 30 if can't detect)
        fps = self._estimate_fps(video_bytes)
        
        window_frames = int(window_seconds * fps)
        stride_frames = int(stride_seconds * fps)
        
        # Ensure minimum window size
        window_frames = max(window_frames, SLR_TARGET_FRAMES)
        stride_frames = max(stride_frames, 1)
        
        segments = []
        
        # Sliding window
        start = 0
        while start < total_frames:
            end = min(start + window_frames, total_frames)
            
            # Skip if window too small
            if end - start < SLR_TARGET_FRAMES // 2:
                break
            
            # Extract window frames
            window = all_frames[start:end]
            
            # Downsample and normalize
            window = downsample_frames(window, SLR_TARGET_FRAMES)
            window = normalize_frames(window)
            
            # Predict
            result = self._predict_frames(window)
            
            if result["confidence"] >= min_confidence:
                segments.append({
                    "start_frame": start,
                    "end_frame": end,
                    "label": result["label"],
                    "confidence": result["confidence"]
                })
            
            start += stride_frames
        
        # Merge consecutive duplicates into sequence
        sequence: List[str] = []
        for seg in segments:
            if not sequence or sequence[-1] != seg["label"]:
                sequence.append(seg["label"])
        
        return {
            "total_frames": total_frames,
            "fps": fps,
            "segments": segments,
            "sequence": sequence
        }
    
    def _estimate_fps(self, video_bytes: bytes) -> float:
        """Estimate video FPS from bytes"""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name
        
        try:
            cap = cv2.VideoCapture(tmp_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            return fps if fps > 0 else 30.0
        finally:
            os.unlink(tmp_path)


# Global instance
model_manager = SLRModelManager()

# Export commonly used constants
NUM_CLASSES = SLR_NUM_CLASSES
TARGET_FRAMES = SLR_TARGET_FRAMES