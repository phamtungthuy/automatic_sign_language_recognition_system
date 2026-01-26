from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
from db.database import DBSessionMiddleware
from starlette.middleware.cors import CORSMiddleware

from utils.logs import logger
from utils.constants import API_PREFIX, PROJECT_NAME, BACKEND_CORS_ORIGINS, DATABASE_URL
from db.helpers.exception_handler import (
    CustomException,
    custom_exception_handler,
    http_exception_handler,
    validation_exception_handler,
    general_exception_handler,
)
from api.apps.api_router import router


def get_application() -> FastAPI:
    application = FastAPI(
        title=PROJECT_NAME,
        docs_url="/docs",
        redoc_url="/re-docs",
        openapi_url=f"{API_PREFIX}/openapi.json",
        description="""
        Base frame with FastAPI micro framework + Postgresql
            - Login/Register with JWT
            - Permission
            - CRUD User
            - Unit testing with Pytest
            - Dockerize
        """,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.add_middleware(DBSessionMiddleware, db_url=DATABASE_URL)


    application.add_exception_handler(CustomException, custom_exception_handler)  # type: ignore
    application.add_exception_handler(HTTPException, http_exception_handler)  # type: ignore
    application.add_exception_handler(RequestValidationError, validation_exception_handler)  # type: ignore
    application.add_exception_handler(Exception, general_exception_handler)

    # Routers
    application.include_router(router, prefix=API_PREFIX)

    return application


app = get_application()
