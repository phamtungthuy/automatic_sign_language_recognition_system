"""
Sign Language Recognition API Endpoints
"""
from typing import List, Optional
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from pydantic import BaseModel

from utils.sp_slr import model_manager, NUM_CLASSES

router = APIRouter()


# ============== RESPONSE MODELS ==============
class PredictionResult(BaseModel):
    label: str
    confidence: float
    label_idx: int


class PredictionResponse(BaseModel):
    success: bool
    prediction: PredictionResult


class TopKPredictionResponse(BaseModel):
    success: bool
    predictions: List[PredictionResult]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    num_classes: int


class LabelsResponse(BaseModel):
    total_classes: int
    labels: List[str]


class SegmentPrediction(BaseModel):
    """Prediction for a video segment"""
    segment_idx: int
    start_frame: int
    end_frame: int
    label: str
    confidence: float


class ContinuousPredictionResponse(BaseModel):
    """Response for continuous sign recognition"""
    success: bool
    total_frames: int
    total_segments: int
    segments: List[SegmentPrediction]
    # Merged sequence (consecutive duplicates removed)
    sequence: List[str]
    sequence_text: str  # Human readable


# ============== ENDPOINTS ==============
@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API and model status"""
    return HealthResponse(
        status="healthy" if model_manager.is_loaded() else "model_not_loaded",
        model_loaded=model_manager.is_loaded(),
        device=model_manager.device,
        num_classes=NUM_CLASSES
    )


@router.get("/labels", response_model=LabelsResponse)
async def get_labels():
    """Get list of all available sign language labels"""
    if not model_manager.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return LabelsResponse(
        total_classes=len(model_manager.label_mapping),
        labels=list(model_manager.label_mapping.keys())
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Upload a video file and get the predicted sign language label.
    
    Supported formats: mp4, avi, mov, mkv
    """
    if not model_manager.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded. Please wait for initialization.")
    
    # Validate file type
    filename = file.filename or ""
    if not filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file format. Supported: mp4, avi, mov, mkv"
        )
    
    try:
        video_bytes = await file.read()
        results = model_manager.predict(video_bytes, top_k=1)
        
        return PredictionResponse(
            success=True,
            prediction=PredictionResult(**results[0])
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")


@router.post("/predict/topk", response_model=TopKPredictionResponse)
async def predict_topk(
    file: UploadFile = File(...),
    k: int = Query(default=5, ge=1, le=NUM_CLASSES, description="Number of top predictions")
):
    """
    Upload a video file and get top-k predicted sign language labels.
    """
    if not model_manager.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded. Please wait for initialization.")
    
    filename = file.filename or ""
    if not filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file format. Supported: mp4, avi, mov, mkv"
        )
    
    try:
        video_bytes = await file.read()
        results = model_manager.predict(video_bytes, top_k=k)
        print([PredictionResult(**r) for r in results])
        return TopKPredictionResponse(
            success=True,
            predictions=[PredictionResult(**r) for r in results]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")


@router.post("/predict/continuous", response_model=ContinuousPredictionResponse)
async def predict_continuous(
    file: UploadFile = File(...),
    window_seconds: float = Query(default=2.0, ge=0.5, le=10.0, description="Window size in seconds"),
    stride_seconds: float = Query(default=1.0, ge=0.25, le=5.0, description="Stride between windows in seconds"),
    min_confidence: float = Query(default=0.5, ge=0.0, le=1.0, description="Minimum confidence to include prediction")
):
    """
    🎬 **Continuous Sign Language Recognition**
    
    Upload a long video (e.g., presentation) and get a sequence of predicted signs.
    
    The video is split into overlapping windows, each window is classified,
    and consecutive duplicate predictions are merged into a sequence.
    
    **Parameters:**
    - **window_seconds**: Length of each analysis window (default: 2s)
    - **stride_seconds**: How much to slide between windows (default: 1s)  
    - **min_confidence**: Minimum confidence to include a prediction (default: 0.5)
    
    **Example:** A 10s video with window=2s, stride=1s will produce ~9 predictions.
    """
    if not model_manager.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded. Please wait for initialization.")
    
    filename = file.filename or ""
    if not filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file format. Supported: mp4, avi, mov, mkv"
        )
    
    try:
        video_bytes = await file.read()
        
        # Use the new continuous prediction method
        result = model_manager.predict_continuous(
            video_bytes,
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
            min_confidence=min_confidence
        )
        
        segments = [
            SegmentPrediction(
                segment_idx=i,
                start_frame=seg["start_frame"],
                end_frame=seg["end_frame"],
                label=seg["label"],
                confidence=seg["confidence"]
            )
            for i, seg in enumerate(result["segments"])
        ]
        
        return ContinuousPredictionResponse(
            success=True,
            total_frames=result["total_frames"],
            total_segments=len(segments),
            segments=segments,
            sequence=result["sequence"],
            sequence_text=" → ".join(result["sequence"]) if result["sequence"] else "(no signs detected)"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")