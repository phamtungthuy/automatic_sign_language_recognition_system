"""
Sign Language Recognition - FastAPI Server
Standalone server, no database dependencies.

Usage:
    uvicorn svr.slr_inference:app --host 0.0.0.0 --port 8000 --reload
"""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from utils.sp_slr import model_manager


# ============== CONFIGURATION ==============
# Override model/label paths via environment variables
MODEL_PATH = os.getenv("SLR_MODEL_PATH", None)
LABEL_MAPPING_PATH = os.getenv("SLR_LABEL_MAPPING_PATH", None)


# ============== LIFESPAN (load model on startup) ==============
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model when server starts"""
    print("🚀 Starting Sign Language Recognition Server...")
    try:
        model_manager.load(
            model_path=MODEL_PATH,
            label_mapping_path=LABEL_MAPPING_PATH
        )
    except FileNotFoundError as e:
        print(f"⚠️ Warning: Could not load model: {e}")
        print("   Server will start but predictions will fail.")
        print("   Set SLR_MODEL_PATH and SLR_LABEL_MAPPING_PATH env vars.")
    
    yield
    
    print("👋 Shutting down...")


# ============== CREATE APP ==============
def create_app() -> FastAPI:
    application = FastAPI(
        title="🤟 Sign Language Recognition API",
        description="""
        API for Vietnamese Sign Language Recognition using ConvNeXt-Transformer.
        
        ## Endpoints
        - `POST /api/v1/slr/predict` - Predict single label
        - `POST /api/v1/slr/predict/topk` - Predict top-k labels
        - `GET /api/v1/slr/labels` - List all labels
        - `GET /api/v1/slr/health` - Health check
        
        ## Usage
        ```bash
        curl -X POST "http://localhost:8000/api/v1/slr/predict" -F "file=@video.mp4"
        ```
        """,
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )

    # CORS - allow all origins for development
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Import and include router
    from api.slr.api_router import router
    application.include_router(router)
    
    # Root endpoint
    @application.get("/")
    async def root():
        return {
            "message": "🤟 Sign Language Recognition API",
            "docs": "/docs",
            "health": "/api/v1/slr/health"
        }

    return application


app = create_app()


# ============== RUN WITH PYTHON ==============
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "svr.slr_inference:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
