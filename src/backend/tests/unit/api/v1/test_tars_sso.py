import time
from uuid import uuid4

import jwt
import pytest
from httpx import AsyncClient
from langflow.services.database.models.auth import SSOUserProfile
from langflow.services.database.models.user.model import User
from langflow.services.deps import session_scope
from langflow.services.tars.ticket import TICKET_SECRET_ENV
from sqlmodel import select

SECRET = "tars-sso-test-secret-with-enough-length"  # noqa: S105 # pragma: allowlist secret
SSO_URL = "api/v1/tars/sso"


def make_ticket(*, secret: str = SECRET, drop: tuple[str, ...] = (), **overrides) -> str:
    now = int(time.time())
    claims = {
        "iss": "librechat",
        "aud": "langflow",
        "sub": "tars-user-1",
        "preferred_username": "alice",
        "name": "Alice",
        "iat": now,
        "exp": now + 60,
        "jti": uuid4().hex,
        **overrides,
    }
    for key in drop:
        claims.pop(key, None)
    return jwt.encode(claims, secret, algorithm="HS256")


@pytest.fixture
def sso_secret(monkeypatch):
    monkeypatch.setenv(TICKET_SECRET_ENV, SECRET)


async def sign_in(client: AsyncClient, ticket: str):
    client.cookies.clear()
    return await client.get(SSO_URL, params={"ticket": ticket})


async def whoami(client: AsyncClient) -> dict:
    response = await client.get("api/v1/users/whoami")
    assert response.status_code == 200, response.text
    return response.json()


async def tars_profile(tars_id: str) -> SSOUserProfile | None:
    async with session_scope() as session:
        stmt = select(SSOUserProfile).where(
            SSOUserProfile.sso_provider == "tars",
            SSOUserProfile.sso_user_id == tars_id,
        )
        return (await session.exec(stmt)).first()


@pytest.mark.usefixtures("sso_secret")
async def test_valid_ticket_opens_a_session_on_the_mapped_account(client: AsyncClient):
    response = await sign_in(client, make_ticket())

    assert response.status_code == 302
    assert response.headers["location"] == "/"
    assert response.headers["cache-control"] == "no-store"
    assert {"access_token_lf", "refresh_token_lf"} <= set(response.cookies)

    me = await whoami(client)
    assert me["username"] == "alice"
    assert me["is_superuser"] is False

    profile = await tars_profile("tars-user-1")
    assert profile is not None
    assert str(profile.user_id) == me["id"]


@pytest.mark.usefixtures("sso_secret")
async def test_renamed_tars_user_keeps_the_same_account(client: AsyncClient):
    await sign_in(client, make_ticket(sub="tars-user-2", preferred_username="bob"))
    first = await whoami(client)

    await sign_in(client, make_ticket(sub="tars-user-2", preferred_username="robert"))
    second = await whoami(client)

    assert second["id"] == first["id"]


@pytest.mark.usefixtures("sso_secret")
async def test_admin_claim_is_synced_to_superuser_on_every_sign_in(client: AsyncClient):
    await sign_in(client, make_ticket(sub="tars-admin", preferred_username="boss", admin=True))
    assert (await whoami(client))["is_superuser"] is True

    await sign_in(client, make_ticket(sub="tars-admin", preferred_username="boss", admin=False))
    assert (await whoami(client))["is_superuser"] is False


@pytest.mark.usefixtures("sso_secret")
async def test_a_ticket_can_only_be_redeemed_once(client: AsyncClient):
    ticket = make_ticket(sub="tars-user-3", preferred_username="carol")

    assert (await sign_in(client, ticket)).status_code == 302
    replay = await sign_in(client, ticket)

    assert replay.status_code == 401
    assert "access_token_lf" not in replay.cookies


@pytest.mark.usefixtures("sso_secret")
@pytest.mark.parametrize(
    "ticket_kwargs",
    [
        pytest.param({"secret": "another-secret-that-is-long-enough-to-sign"}, id="wrong-signature"),
        pytest.param({"aud": "someone-else"}, id="wrong-audience"),
        pytest.param({"iss": "someone-else"}, id="wrong-issuer"),
        pytest.param({"iat": int(time.time()) - 120, "exp": int(time.time()) - 60}, id="expired"),
        pytest.param({"exp": int(time.time()) + 3600}, id="lifetime-too-long"),
        pytest.param({"drop": ("sub",)}, id="no-subject"),
        pytest.param({"drop": ("jti",)}, id="no-ticket-id"),
    ],
)
async def test_invalid_tickets_are_refused(client: AsyncClient, ticket_kwargs: dict):
    response = await sign_in(client, make_ticket(**ticket_kwargs))

    assert response.status_code == 401
    assert "access_token_lf" not in response.cookies


async def test_sign_in_is_refused_until_the_secret_is_configured(client: AsyncClient, monkeypatch):
    monkeypatch.delenv(TICKET_SECRET_ENV, raising=False)

    response = await sign_in(client, make_ticket())

    assert response.status_code == 503


@pytest.mark.usefixtures("sso_secret")
async def test_inactive_account_is_refused(client: AsyncClient):
    await sign_in(client, make_ticket(sub="tars-user-4", preferred_username="dave"))
    profile = await tars_profile("tars-user-4")
    assert profile is not None
    async with session_scope() as session:
        user = await session.get(User, profile.user_id)
        user.is_active = False
        session.add(user)

    response = await sign_in(client, make_ticket(sub="tars-user-4", preferred_username="dave"))

    assert response.status_code == 403
    assert "access_token_lf" not in response.cookies


async def last_login_of(tars_id: str):
    profile = await tars_profile(tars_id)
    async with session_scope() as session:
        return (await session.get(User, profile.user_id)).last_login_at


@pytest.mark.usefixtures("sso_secret")
async def test_admin_can_enter_the_system_account(client: AsyncClient):
    response = await sign_in(
        client, make_ticket(sub="tars-root", preferred_username="root", admin=True, act_as_system=True)
    )

    assert response.status_code == 302
    me = await whoami(client)
    assert me["username"] == "langflow"
    assert me["is_superuser"] is True


@pytest.mark.usefixtures("sso_secret")
async def test_admin_can_enter_a_users_workspace_without_touching_their_last_login(client: AsyncClient):
    await sign_in(client, make_ticket(sub="tars-user-5", preferred_username="erin"))
    erin = await whoami(client)
    before = await last_login_of("tars-user-5")

    response = await sign_in(
        client, make_ticket(sub="tars-root", preferred_username="root", admin=True, act_as="tars-user-5")
    )

    assert response.status_code == 302
    assert (await whoami(client))["id"] == erin["id"]
    assert await last_login_of("tars-user-5") == before


@pytest.mark.usefixtures("sso_secret")
@pytest.mark.parametrize(
    "switch",
    [pytest.param({"act_as_system": True}, id="system"), pytest.param({"act_as": "tars-user-1"}, id="user")],
)
async def test_non_admins_cannot_switch_identity(client: AsyncClient, switch: dict):
    response = await sign_in(client, make_ticket(sub="tars-user-6", preferred_username="frank", **switch))

    assert response.status_code == 403
    assert "access_token_lf" not in response.cookies


@pytest.mark.usefixtures("sso_secret")
async def test_switching_to_someone_who_never_entered_workflow_is_refused(client: AsyncClient):
    response = await sign_in(
        client, make_ticket(sub="tars-root", preferred_username="root", admin=True, act_as="tars-nobody")
    )

    assert response.status_code == 404
    assert "access_token_lf" not in response.cookies


@pytest.mark.usefixtures("sso_secret")
async def test_switching_to_an_inactive_account_is_refused(client: AsyncClient):
    await sign_in(client, make_ticket(sub="tars-user-7", preferred_username="grace"))
    profile = await tars_profile("tars-user-7")
    async with session_scope() as session:
        user = await session.get(User, profile.user_id)
        user.is_active = False
        session.add(user)

    response = await sign_in(
        client, make_ticket(sub="tars-root", preferred_username="root", admin=True, act_as="tars-user-7")
    )

    assert response.status_code == 403


@pytest.mark.usefixtures("sso_secret")
async def test_acting_as_oneself_is_an_ordinary_sign_in(client: AsyncClient):
    response = await sign_in(
        client, make_ticket(sub="tars-root", preferred_username="root", admin=True, act_as="tars-root")
    )

    assert response.status_code == 302
    assert (await whoami(client))["username"] == "root"


@pytest.mark.usefixtures("sso_secret")
async def test_a_ticket_cannot_name_both_a_user_and_the_system_account(client: AsyncClient):
    response = await sign_in(
        client,
        make_ticket(sub="tars-root", admin=True, act_as="tars-user-1", act_as_system=True),
    )

    assert response.status_code == 401


def actor_cookie(response) -> tuple[str, str]:
    from langflow.services.tars.actor import ACTOR_COOKIE, decode_actor

    actor = decode_actor(response.cookies.get(ACTOR_COOKIE))
    assert actor is not None
    return actor.tars_id, actor.langflow_user_id


@pytest.mark.usefixtures("sso_secret")
async def test_sign_in_records_the_operator_for_the_session_account(client: AsyncClient):
    response = await sign_in(client, make_ticket(sub="tars-user-8", preferred_username="heidi"))

    assert actor_cookie(response) == ("tars-user-8", (await whoami(client))["id"])


@pytest.mark.usefixtures("sso_secret")
async def test_an_admin_in_another_workspace_stays_the_operator(client: AsyncClient):
    response = await sign_in(
        client, make_ticket(sub="tars-root", preferred_username="root", admin=True, act_as_system=True)
    )

    me = await whoami(client)
    assert me["username"] == "langflow"
    assert actor_cookie(response) == ("tars-root", me["id"])


@pytest.mark.usefixtures("sso_secret")
async def test_logout_ends_the_session_and_forgets_the_operator(client: AsyncClient):
    await sign_in(client, make_ticket(sub="tars-user-9", preferred_username="ivan"))

    response = await client.post("api/v1/tars/logout")

    assert response.status_code == 204
    cleared = response.headers.get_list("set-cookie")
    for name in ("access_token_lf", "refresh_token_lf", "apikey_tkn_lflw", "tars_actor_lf"):
        assert any(header.startswith(f"{name}=") and "Max-Age=0" in header for header in cleared), name
    client.cookies.clear()
    assert (await client.get("api/v1/users/whoami")).status_code in (401, 403)
