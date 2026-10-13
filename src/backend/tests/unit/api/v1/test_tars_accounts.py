import time

import jwt
import pytest
from httpx import AsyncClient
from langflow.services.database.models.auth import SSOUserProfile
from langflow.services.database.models.user.model import User
from langflow.services.deps import session_scope
from langflow.services.tars.ticket import TICKET_SECRET_ENV
from sqlmodel import select

from tests.unit.api.v1.test_tars_sso import SECRET, make_ticket

ACCOUNTS_URL = "api/v1/tars/accounts"


def service_token(*, secret: str = SECRET, audience: str = "langflow-service", lifetime: int = 60) -> str:
    now = int(time.time())
    claims = {"iss": "librechat", "aud": audience, "iat": now, "exp": now + lifetime}
    return jwt.encode(claims, secret, algorithm="HS256")


@pytest.fixture
def sso_secret(monkeypatch):
    monkeypatch.setenv(TICKET_SECRET_ENV, SECRET)


async def enter_workflow(client: AsyncClient, tars_id: str, username: str) -> None:
    client.cookies.clear()
    response = await client.get(
        "api/v1/tars/sso", params={"ticket": make_ticket(sub=tars_id, preferred_username=username)}
    )
    assert response.status_code == 302
    client.cookies.clear()


async def list_accounts(client: AsyncClient, token: str | None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return await client.get(ACCOUNTS_URL, headers=headers)


@pytest.mark.usefixtures("sso_secret")
async def test_lists_active_tars_accounts_by_username(client: AsyncClient):
    await enter_workflow(client, "tars-b", "bob")
    await enter_workflow(client, "tars-a", "amy")
    await enter_workflow(client, "tars-c", "cara")
    async with session_scope() as session:
        stmt = select(SSOUserProfile.user_id).where(SSOUserProfile.sso_user_id == "tars-c")
        user = await session.get(User, (await session.exec(stmt)).first())
        user.is_active = False
        session.add(user)

    response = await list_accounts(client, service_token())

    assert response.status_code == 200
    accounts = response.json()
    assert [(a["tars_id"], a["username"]) for a in accounts] == [("tars-a", "amy"), ("tars-b", "bob")]
    assert all(a["last_login_at"] for a in accounts)


@pytest.mark.usefixtures("sso_secret")
@pytest.mark.parametrize(
    "token",
    [
        pytest.param(None, id="missing"),
        pytest.param(
            service_token(secret="another-secret-that-is-long-enough-to-sign"),  # noqa: S106
            id="wrong-signature",
        ),
        pytest.param(service_token(audience="langflow"), id="sign-in-audience"),
        pytest.param(service_token(lifetime=3600), id="lifetime-too-long"),
    ],
)
async def test_refuses_anything_but_a_valid_service_token(client: AsyncClient, token: str | None):
    response = await list_accounts(client, token)

    assert response.status_code == 401


async def test_refuses_until_the_secret_is_configured(client: AsyncClient, monkeypatch):
    monkeypatch.delenv(TICKET_SECRET_ENV, raising=False)

    response = await list_accounts(client, service_token())

    assert response.status_code == 503


@pytest.mark.usefixtures("sso_secret")
async def test_a_sign_in_ticket_is_not_a_service_token(client: AsyncClient):
    response = await list_accounts(client, make_ticket())

    assert response.status_code == 401


async def transfer(
    client: AsyncClient, tars_id: str, *, label: str, deactivate: bool = False, token: str | None = None
):
    headers = {"Authorization": f"Bearer {token or service_token()}"}
    return await client.post(
        f"{ACCOUNTS_URL}/{tars_id}/transfer", json={"label": label, "deactivate": deactivate}, headers=headers
    )


@pytest.mark.usefixtures("sso_secret")
async def test_transfer_hands_every_flow_to_the_system_account_unpublished(client: AsyncClient):
    from langflow.services.database.models.flow.model import Flow
    from langflow.services.database.models.flow_version.model import FlowVersion
    from langflow.services.database.models.folder.model import Folder

    from tests.unit.api.v1.test_tars_tools import add_flow, default_folder, system_user

    system = await system_user()
    await add_flow(system.id, await default_folder(system.id), "guide")
    await enter_workflow(client, "tars-x", "xavier")
    async with session_scope() as session:
        xavier = (
            await session.exec(select(SSOUserProfile.user_id).where(SSOUserProfile.sso_user_id == "tars-x"))
        ).first()
    xavier_project = await default_folder(xavier)
    published = await add_flow(xavier, xavier_project, "guide")
    await add_flow(xavier, xavier_project, "notes", mcp=False)
    async with session_scope() as session:
        session.add(FlowVersion(flow_id=published.id, user_id=xavier, version_number=1, data={}))

    response = await transfer(client, "tars-x", label="王小明")

    assert response.status_code == 200
    assert response.json() == {"moved": 2, "project": "王小明（已移交）", "deactivated": False}  # noqa: RUF001
    async with session_scope() as session:
        project = (
            await session.exec(select(Folder).where(Folder.user_id == system.id, Folder.name == "王小明（已移交）"))  # noqa: RUF001
        ).first()
        moved = (await session.exec(select(Flow).where(Flow.folder_id == project.id))).all()
        assert {flow.name for flow in moved} == {"guide（王小明）", "notes"}  # noqa: RUF001
        assert all(flow.user_id == system.id and flow.mcp_enabled is False for flow in moved)
        assert (await session.exec(select(Flow).where(Flow.user_id == xavier))).all() == []
        version = (await session.exec(select(FlowVersion).where(FlowVersion.flow_id == published.id))).first()
        assert version.user_id == system.id
        account = await session.get(User, xavier)
        assert account.is_active is True

    again = await transfer(client, "tars-x", label="王小明")
    assert again.json() == {"moved": 0, "project": None, "deactivated": False}


@pytest.mark.usefixtures("sso_secret")
async def test_transfer_can_deactivate_a_deleted_users_account(client: AsyncClient):
    await enter_workflow(client, "tars-y", "yvonne")

    response = await transfer(client, "tars-y", label="yvonne", deactivate=True)

    assert response.json() == {"moved": 0, "project": None, "deactivated": True}
    client.cookies.clear()
    sign_in = await client.get(
        "api/v1/tars/sso", params={"ticket": make_ticket(sub="tars-y", preferred_username="yvonne")}
    )
    assert sign_in.status_code == 403


@pytest.mark.usefixtures("sso_secret")
async def test_transfer_of_someone_who_never_entered_workflow_does_nothing(client: AsyncClient):
    response = await transfer(client, "tars-nobody", label="nobody", deactivate=True)

    assert response.json() == {"moved": 0, "project": None, "deactivated": False}


@pytest.mark.usefixtures("sso_secret")
async def test_transfer_needs_a_service_token(client: AsyncClient):
    response = await client.post(f"{ACCOUNTS_URL}/tars-x/transfer", json={"label": "x"})

    assert response.status_code == 401
