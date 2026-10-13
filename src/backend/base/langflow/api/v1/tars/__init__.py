"""Routes for the TARS (pwc_tars / LibreChat) integration, mounted at ``/api/v1/tars``."""

from fastapi import APIRouter

from langflow.api.v1.tars.accounts import router as accounts_router
from langflow.api.v1.tars.sso import router as sso_router
from langflow.api.v1.tars.tools import router as tools_router

router = APIRouter(prefix="/tars")
router.include_router(sso_router)
router.include_router(accounts_router)
router.include_router(tools_router)

__all__ = ["router"]
