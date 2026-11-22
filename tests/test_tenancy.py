"""P9: project isolation, scoped tokens and artifact ACLs (real enforcement)."""

from __future__ import annotations

import pytest

from uzen.core.nws.families import build_architecture
from uzen.errors import AuthorizationDenied
from uzen.infrastructure.security.access import (
    Principal,
    TokenAuthority,
    ensure_artifact_access,
    principal_context,
)
from uzen.jobs.service import JobService

fastapi_testclient = pytest.importorskip("fastapi.testclient")

BOOTSTRAP = "bootstrap-secret-value"


@pytest.fixture()
def api(service: JobService):
    from uzen.interfaces.api.app import create_app

    return fastapi_testclient.TestClient(
        create_app(service=service, auto_run=False, api_token=BOOTSTRAP)
    )


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _mint(api, project: str, scopes: list[str]) -> str:
    response = api.post(
        "/v1/tokens",
        json={"project_id": project, "scopes": scopes, "description": "test"},
        headers=_bearer(BOOTSTRAP),
    )
    assert response.status_code == 201, response.text
    return response.json()["token"]


def test_authentication_is_mandatory_and_health_stays_public(api) -> None:
    assert api.get("/v1/jobs").status_code == 401
    assert api.get("/v1/jobs", headers=_bearer("wrong-token")).status_code == 401
    assert api.get("/health").status_code == 200
    assert api.get("/v1/operations").status_code == 200
    assert api.get("/v1/jobs", headers=_bearer(BOOTSTRAP)).status_code == 200


def test_jobs_are_owned_by_the_submitting_project(api) -> None:
    alpha = _mint(api, "alpha", ["jobs:read", "jobs:submit"])
    beta = _mint(api, "beta", ["jobs:read", "jobs:submit"])
    created = api.post(
        "/v1/jobs",
        json={"task_scope": "inspection", "objective": "tenant test"},
        headers=_bearer(alpha),
    )
    assert created.status_code == 201, created.text
    job = created.json()
    assert job["project_id"] == "alpha"

    hidden = api.get(f"/v1/jobs/{job['id']}", headers=_bearer(beta))
    assert hidden.status_code == 403, hidden.text

    own = api.get(f"/v1/jobs/{job['id']}", headers=_bearer(alpha))
    assert own.status_code == 200 and own.json()["project_id"] == "alpha"

    alpha_list = api.get("/v1/jobs", headers=_bearer(alpha)).json()
    assert [row["id"] for row in alpha_list["jobs"]] == [job["id"]]
    beta_list = api.get("/v1/jobs", headers=_bearer(beta)).json()
    assert beta_list["jobs"] == []

    # An admin (bootstrap) may inspect every project.
    admin_list = api.get("/v1/jobs", headers=_bearer(BOOTSTRAP)).json()
    assert job["id"] in [row["id"] for row in admin_list["jobs"]]


def test_scopes_are_enforced_not_just_authentication(api) -> None:
    reader = _mint(api, "readers", ["jobs:read"])
    assert api.get("/v1/jobs", headers=_bearer(reader)).status_code == 200
    denied = api.post(
        "/v1/jobs",
        json={"task_scope": "inspection"},
        headers=_bearer(reader),
    )
    assert denied.status_code == 403
    assert denied.json()["detail"]["code"] == "auth.denied"
    assert api.get("/v1/tokens", headers=_bearer(reader)).status_code == 403
    assert api.get("/v1/traces", headers=_bearer(reader)).status_code == 403


def test_project_tokens_cannot_escalate_or_cross_projects(api, service: JobService) -> None:
    manager = _mint(api, "team-a", ["jobs:read", "tokens:admin"])
    escalate = api.post(
        "/v1/tokens",
        json={"scopes": ["admin"]},
        headers=_bearer(manager),
    )
    assert escalate.status_code == 403
    foreign = api.post(
        "/v1/tokens",
        json={"project_id": "team-b", "scopes": ["jobs:read"]},
        headers=_bearer(manager),
    )
    assert foreign.status_code == 403
    within = api.post(
        "/v1/tokens",
        json={"scopes": ["jobs:read"]},
        headers=_bearer(manager),
    )
    assert within.status_code == 201
    assert within.json()["project_id"] == "team-a"

    authority = TokenAuthority(service.db, legacy_token=BOOTSTRAP)
    with pytest.raises(AuthorizationDenied):
        authority.mint(
            Principal("team-a", "tok", frozenset({"jobs:read"})),
            scopes=["jobs:read", "admin"],
        )


def test_revoked_token_is_rejected(api) -> None:
    minted = api.post(
        "/v1/tokens",
        json={"project_id": "alpha", "scopes": ["jobs:read"]},
        headers=_bearer(BOOTSTRAP),
    ).json()
    assert api.get("/v1/jobs", headers=_bearer(minted["token"])).status_code == 200
    revoked = api.delete(f"/v1/tokens/{minted['token_id']}", headers=_bearer(BOOTSTRAP))
    assert revoked.status_code == 200 and revoked.json()["revoked"] is True
    assert api.get("/v1/jobs", headers=_bearer(minted["token"])).status_code == 401


def test_artifact_acl_grants_read_but_not_write(api, service: JobService) -> None:
    definition = build_architecture("mlp", dict(input_dim=4, hidden_dims=[8], output_dim=2))
    package = definition.build_package(seed=5, name="tenant_artifact")
    record = service.store.put(package, operation="init")
    service.db.index_artifact(
        artifact_id=record.artifact_id, path=str(record.path), project_id="alpha"
    )

    beta = _mint(api, "beta", ["artifacts:read", "artifacts:write", "artifacts:share"])
    denied = api.get(f"/v1/artifacts/{record.artifact_id}", headers=_bearer(beta))
    assert denied.status_code == 403, denied.text

    beta_principal = Principal("beta", "tok", frozenset({"artifacts:read", "artifacts:write"}))
    with pytest.raises(AuthorizationDenied):
        ensure_artifact_access(service.db, beta_principal, record.artifact_id, permission="read")

    shared = api.post(
        f"/v1/artifacts/{record.artifact_id}/share",
        json={"project_id": "beta", "permission": "read"},
        headers=_bearer(_mint(api, "alpha", ["artifacts:share", "artifacts:read"])),
    )
    assert shared.status_code == 201, shared.text

    # The read grant now passes scoping; write still does not.
    ensure_artifact_access(service.db, beta_principal, record.artifact_id, permission="read")
    with pytest.raises(AuthorizationDenied):
        ensure_artifact_access(service.db, beta_principal, record.artifact_id, permission="write")

    acl = api.get(f"/v1/artifacts/{record.artifact_id}/acl", headers=_bearer(BOOTSTRAP)).json()
    assert acl["owner_project"] == "alpha"
    assert {grant["project_id"] for grant in acl["grants"]} == {"beta"}

    # A non-owner cannot widen a grant.
    stolen = api.post(
        f"/v1/artifacts/{record.artifact_id}/share",
        json={"project_id": "gamma", "permission": "read"},
        headers=_bearer(beta),
    )
    assert stolen.status_code == 403


def test_ambient_principal_scopes_service_writes(service: JobService) -> None:
    alpha = Principal("alpha", "tok", frozenset({"jobs:submit", "jobs:read"}), True)
    with principal_context(alpha):
        job = service.submit({"task_scope": "inspection"})
    assert job["project_id"] == "alpha"
    # A different project's scoped caller cannot read alpha's job; open single-tenant
    # mode (the anonymous engine principal) is an admin of 'default' and can.
    beta = Principal("beta", "tok", frozenset({"jobs:read"}), True)
    with principal_context(beta):
        with pytest.raises(AuthorizationDenied):
            service.get(job["id"])


def test_admin_header_can_act_in_another_project(api) -> None:
    created = api.post(
        "/v1/jobs",
        json={"task_scope": "inspection"},
        headers={**_bearer(BOOTSTRAP), "X-UZEN-Project": "ops"},
    )
    assert created.status_code == 201
    assert created.json()["project_id"] == "ops"
