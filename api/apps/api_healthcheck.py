from fastapi import APIRouter

from db.schemas import ResponseSchemaBase

router = APIRouter()


@router.get("", response_model=ResponseSchemaBase)
async def get():
    return {"message": "Health check succes"}
