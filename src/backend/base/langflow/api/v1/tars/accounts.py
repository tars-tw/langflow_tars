"""Service routes LibreChat's backend calls about TARS-linked Langflow accounts."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, HTTPException, status
from lfx.log.logger import logger
from pydantic import BaseModel, Field

from langflow.api.utils import DbSession, DbSessionReadOnly
from langflow.services.tars.accounts import list_tars_accounts
from langflow.services.tars.service_auth import TarsService
from langflow.services.tars.transfer import TransferError, transfer_account

router = APIRouter(tags=["TARS"], include_in_schema=False, dependencies=[TarsService])


class TarsAccountRead(BaseModel):
    tars_id: str
    username: str
    last_login_at: datetime | None


@router.get("/accounts")
async def list_accounts(db: DbSessionReadOnly) -> list[TarsAccountRead]:
    """pwc_tars users who have entered Workflow, i.e. whose workspace an admin can switch into."""
    accounts = await list_tars_accounts(db)
    return [TarsAccountRead(**vars(account)) for account in accounts]


class TransferRequest(BaseModel):
    label: str = Field(min_length=1, max_length=120)
    """Who the flows came from, as people know them; names the receiving project."""
    deactivate: bool = False
    """Also deactivate their Langflow account (their pwc_tars account is gone)."""


class TransferRead(BaseModel):
    moved: int
    project: str | None
    deactivated: bool


@router.post("/accounts/{tars_id}/transfer")
async def transfer(tars_id: str, body: TransferRequest, db: DbSession) -> TransferRead:
    """Hand every flow ``tars_id`` owns to the system account, unpublished; see ``services.tars.transfer``."""
    try:
        result = await transfer_account(db, tars_id, label=body.label.strip(), deactivate=body.deactivate)
    except TransferError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    if result.moved or result.deactivated:
        logger.info(
            f"Transferred {result.moved} flow(s) of tars user {tars_id} to the system account"
            f"{f' as project {result.project}' if result.project else ''}"
            f"{'; Langflow account deactivated' if result.deactivated else ''}"
        )
    return TransferRead(**vars(result))
