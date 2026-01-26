import enum
from typing import Optional

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from db.schemas import DataResponse


class ExceptionType(enum.Enum):
    MS_UNAVAILABLE = 500, "990", "System is under maintenance, please try again later"
    MS_INVALID_API_PATH = (
        500,
        "991",
        "System is under maintenance, please try again later",
    )
    DATA_RESPONSE_MALFORMED = 500, "992", "Error occurred, please contact admin!"

    def __new__(cls, *args, **kwds):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, http_code, code, message):
        self.http_code = http_code
        self.code = code
        self.message = message


class CustomException(Exception):
    """Custom exception with unified response format"""

    http_code: int
    code: str
    message: str

    def __init__(
        self,
        http_code: Optional[int] = None,
        code: Optional[str] = None,
        message: Optional[str] = None,
    ):
        self.http_code = http_code if http_code else 500
        self.code = code if code else str(self.http_code)
        self.message = message if message else "Internal server error"


async def custom_exception_handler(request: Request, exc: CustomException):
    """Handle CustomException with unified format: {code, message, data}"""
    response = DataResponse.custom_response(
        code=exc.code, message=exc.message, data=None
    )
    return JSONResponse(status_code=exc.http_code, content=response.model_dump())


async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle FastAPI HTTPException with unified format"""
    response = DataResponse.custom_response(
        code=str(exc.status_code), message=exc.detail, data=None
    )
    return JSONResponse(status_code=exc.status_code, content=response.model_dump())


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with unified format"""
    response = DataResponse.custom_response(
        code="422", message=_get_validation_message(exc), data=None
    )
    return JSONResponse(status_code=422, content=response.model_dump())


async def general_exception_handler(request: Request, exc: Exception):
    """Handle all other exceptions with unified format"""
    response = DataResponse.custom_response(
        code="500", message="Internal server error", data=None
    )
    return JSONResponse(status_code=500, content=response.model_dump())


def _get_validation_message(exc: RequestValidationError) -> str:
    """Format validation error messages"""
    errors = []
    for error in exc.errors():
        loc = error.get("loc", [])
        field = loc[-1] if loc else "unknown"
        msg = error.get("msg", "")
        errors.append(f"'{field}': {msg}")

    return ", ".join(errors)
