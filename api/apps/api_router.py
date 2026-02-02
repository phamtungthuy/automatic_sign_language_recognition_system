from fastapi import APIRouter

from api.apps import api_healthcheck
from api.apps.slr import api_slr
from utils.constants import API_VERSION_PREFIX

ROUTERS = [
    (api_healthcheck.router, "health-check", "/healthy"),
    (api_slr.router, "sign-language-recognition", "/slr"),
]

router = APIRouter()

for router_item, tag, prefix in ROUTERS:
    router.include_router(
        router_item, tags=[tag], prefix=f"{API_VERSION_PREFIX}{prefix}"
    )
