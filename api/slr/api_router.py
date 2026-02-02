"""
Sign Language Recognition API Router
"""
from fastapi import APIRouter

from api.slr import api_slr

API_VERSION_PREFIX = "/api/v1"

router = APIRouter()

router.include_router(
    api_slr.router, 
    tags=["sign-language-recognition"], 
    prefix=f"{API_VERSION_PREFIX}/slr"
)
