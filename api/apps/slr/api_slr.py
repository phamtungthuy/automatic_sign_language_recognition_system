from fastapi import APIRouter

router = APIRouter()

@router.post("/slr")
def slr():
    return {"message": "SLR"}