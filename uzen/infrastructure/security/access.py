"""Multi-tenant access control: projects, scoped tokens, artifact ACLs.

UZEN's control plane is multi-tenant at the *project* boundary. A project owns its
jobs, artifacts and models; a token belongs to exactly one project and carries a set
of scopes. Three rules are enforced, in this order:

1. **Authentication** — a bearer token is either a minted project token (only its
   SHA-256 is stored) or the deployment's bootstrap ``UZEN_API_TOKEN`` (admin).
   Authentication is mandatory when a bootstrap token is configured or
   ``UZEN_REQUIRE_AUTH=1``; otherwise the process runs in documented open
   single-tenant mode where the caller is an anonymous admin of ``default``.
2. **Scope** — each route has a required scope (``jobs:submit``, ``weights:write``,
   …). Admin implies every scope.
3. **Ownership** — a job or artifact id from another project is a 403 even with a
   valid scope, unless the artifact was explicitly shared with the caller's project
   through :class:`ArtifactAcl` (a row in ``artifact_acl``).

The principal is exposed as a :class:`~contextvars.ContextVar` while a request is
being served, so the job service attributes new work to the authenticated project
without every service method growing an extra argument. Engine/SDK/CLI callers
outside an HTTP request see the ``default`` project.
"""

from __future__ import annotations

import hmac
import os
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Sequence
from urllib.parse import unquote

from uzen.errors import AuthenticationRequired, AuthorizationDenied, InvalidOperationParameters
from uzen.infrastructure.security.tokens import hash_secret

# Scopes are deliberately resource-oriented and small. Adding one means adding it
# to the route table below, which is the single place authorization is decided.
SCOPES: frozenset[str] = frozenset(
    {
        "jobs:read",
        "jobs:submit",
        "jobs:cancel",
        "weights:read",
        "weights:write",
        "artifacts:read",
        "artifacts:write",
        "artifacts:share",
        "tokens:admin",
        "admin",
    }
)
ADMIN_SCOPES: frozenset[str] = frozenset(SCOPES)


@dataclass(frozen=True)
class Principal:
    """The authenticated caller for one request or engine call."""

    project_id: str = "default"
    token_id: str = "anonymous"
    scopes: frozenset[str] = field(default_factory=frozenset)
    authenticated: bool = False

    @property
    def is_admin(self) -> bool:
        return "admin" in self.scopes

    def allows(self, scope: str | None) -> bool:
        if scope is None:
            return True
        return self.is_admin or scope in self.scopes

    def require(self, scope: str | None) -> None:
        if not self.allows(scope):
            raise AuthorizationDenied(
                f"token '{self.token_id}' lacks required scope '{scope}'",
                required_scope=scope,
                project_id=self.project_id,
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "token_id": self.token_id,
            "scopes": sorted(self.scopes),
            "authenticated": self.authenticated,
        }


#: Open single-tenant mode: the caller may do anything inside ``default``.
ANONYMOUS = Principal(project_id="default", token_id="anonymous", scopes=ADMIN_SCOPES, authenticated=False)

_CURRENT: ContextVar[Principal | None] = ContextVar("uzen_principal", default=None)


def current_principal() -> Principal:
    return _CURRENT.get() or ANONYMOUS


def current_project() -> str:
    return current_principal().project_id


@contextmanager
def principal_context(principal: Principal | None) -> Iterator[Principal]:
    """Install a principal for the duration of a request or engine call."""
    token = _CURRENT.set(principal or ANONYMOUS)
    try:
        yield principal or ANONYMOUS
    finally:
        _CURRENT.reset(token)


def bearer_token(authorization: str | None) -> str | None:
    if not isinstance(authorization, str):
        return None
    scheme, _, value = authorization.partition(" ")
    if scheme.lower() != "bearer" or not value.strip():
        return None
    return value.strip()


class TokenAuthority:
    """Authenticate bearer tokens and mint project-scoped tokens."""

    def __init__(
        self,
        db: Any,
        *,
        legacy_token: str | None = None,
        required: bool | None = None,
    ) -> None:
        self.db = db
        self.legacy_token = legacy_token
        if required is None:
            required = bool(legacy_token) or os.environ.get("UZEN_REQUIRE_AUTH", "") == "1"
        self.required = bool(required)

    def authenticate(self, authorization: str | None, *, project: str | None = None) -> Principal:
        presented = bearer_token(authorization)
        if presented is None:
            if self.required:
                raise AuthenticationRequired(
                    "a bearer token is required for this deployment "
                    "(configure UZEN_API_TOKEN or mint a project token)"
                )
            return (
                ANONYMOUS
                if not project or project == ANONYMOUS.project_id
                else Principal(project_id=str(project), token_id="anonymous", scopes=ADMIN_SCOPES)
            )
        if self.legacy_token and hmac.compare_digest(
            hash_secret(presented), hash_secret(self.legacy_token)
        ):
            return Principal(
                project_id=str(project or "default"),
                token_id="bootstrap-admin",
                scopes=ADMIN_SCOPES,
                authenticated=True,
            )
        record = self.db.authenticate_api_token(presented)
        if record is None:
            raise AuthenticationRequired("invalid or revoked API token")
        return Principal(
            project_id=str(record["project_id"]),
            token_id=str(record["token_id"]),
            scopes=frozenset(record["scopes"]),
            authenticated=True,
        )

    def mint(
        self,
        principal: Principal,
        *,
        project_id: str | None = None,
        scopes: Sequence[str],
        description: str | None = None,
    ) -> dict[str, Any]:
        requested = [str(scope) for scope in scopes]
        unknown = sorted(set(requested) - SCOPES)
        if unknown:
            raise InvalidOperationParameters(
                f"unknown scopes: {unknown}", known_scopes=sorted(SCOPES)
            )
        if not requested:
            raise InvalidOperationParameters("a token needs at least one scope")
        target = str(project_id or principal.project_id)
        if not principal.is_admin:
            if target != principal.project_id:
                raise AuthorizationDenied(
                    "a project token cannot mint tokens for another project",
                    project_id=principal.project_id,
                    requested_project=target,
                )
            extra = sorted(set(requested) - set(principal.scopes))
            if extra:
                raise AuthorizationDenied(
                    f"token cannot grant scopes it does not hold: {extra}",
                    missing=extra,
                )
        return self.db.create_api_token(target, requested, description=description)


# --------------------------------------------------------------------------- #
# route policy
# --------------------------------------------------------------------------- #

# Ordered rules: the first match decides. ``None`` means the route is public
# metadata (health, metrics, operation catalogue) or authenticated by the worker
# plane's own secret instead of a bearer token.
_SCOPE_RULES: tuple[tuple[str, str, str | None], ...] = (
    (r"^/health$", "*", None),
    (r"^/metrics$", "*", None),
    (r"^/docs$", "*", None),
    (r"^/openapi\.json$", "*", None),
    (r"^/v1/operations(/.*)?$", "*", None),
    (r"^/v1/policies(/.*)?$", "*", None),
    (r"^/v1/architectures/families$", "*", None),
    (r"^/v1/workers/(register|heartbeat|claim|renew|report)$", "*", None),
    (r"^/v1/workers(/.*)?$", "*", "admin"),
    (r"^/v1/scheduler(/.*)?$", "*", "admin"),
    (r"^/v1/traces$", "GET", "tokens:admin"),
    (r"^/v1/tokens(/.*)?$", "*", "tokens:admin"),
    (r"^/v1/artifacts/[^/]+/share$", "*", "artifacts:share"),
    (r"^/v1/artifacts/[^/]+/acl$", "GET", "artifacts:read"),
    (r"^/v1/artifacts(/.*)?$", "GET", "artifacts:read"),
    (r"^/v1/jobs$", "GET", "jobs:read"),
    (r"^/v1/jobs$", "POST", "jobs:submit"),
    (r"^/v1/jobs/[^/]+/cancel$", "*", "jobs:cancel"),
    (r"^/v1/jobs/[^/]+/preempt$", "*", "jobs:cancel"),
    (r"^/v1/jobs(/.*)?$", "*", "jobs:read"),
    (r"^/v1/models$", "GET", "weights:read"),
    (r"^/v1/models(/.*)?$", "GET", "weights:read"),
    (r"^/v1/weights$", "GET", "weights:read"),
    (r"^/v1/weights/analyze$", "*", "weights:read"),
    (r"^/v1/weights/evaluate$", "*", "weights:read"),
    (r"^/v1/weights(/.*)?$", "*", "weights:write"),
    (r"^/v1/models(/.*)?$", "*", "weights:write"),
    (r"^/v1/campaigns$", "GET", "jobs:read"),
    (r"^/v1/campaigns(/.*)?$", "*", "jobs:submit"),
    (r"^/v1/benchmarks(/.*)?$", "*", "artifacts:read"),
    (r"^/v1/architectures(/.*)?$", "*", "jobs:submit"),
    (r"^/v1/stats$", "*", "jobs:read"),
)

_JOB_PATH = re.compile(r"^/v1/jobs/([^/]+)")
_ARTIFACT_PATHS = (re.compile(r"^/v1/artifacts/([^/]+)"), re.compile(r"^/v1/models/([^/]+)"))
# Path segments that are collection verbs, not resource ids.
_NON_ID_SEGMENTS = {"jobs", "models", "weights", "artifacts", "campaigns", "architectures", "verify"}


def required_scope(method: str, path: str) -> str | None:
    """Return the scope a request needs, or ``None`` when it needs no bearer scope."""
    method = (method or "GET").upper()
    for pattern, rule_method, scope in _SCOPE_RULES:
        if rule_method != "*" and rule_method != method:
            continue
        if re.match(pattern, path):
            return scope
    # Unknown route: keep it behind authentication so a new endpoint is never
    # accidentally public; the router will answer 404/405 for a real bad path.
    return "jobs:read"


def enforce_resource_scope(db: Any, principal: Principal, method: str, path: str) -> None:
    """Refuse cross-project access to a concrete job or artifact id.

    A missing id is left to the route handler (which returns its normal 404), so
    this check never leaks whether another tenant's id exists.
    """
    if principal.is_admin:
        return
    job_match = _JOB_PATH.match(path)
    if job_match:
        job_id = unquote(job_match.group(1))
        if job_id not in _NON_ID_SEGMENTS:
            try:
                project = db.project_of_job(job_id)
            except Exception:
                return
            if project != principal.project_id:
                raise AuthorizationDenied(
                    "job belongs to another project",
                    job_id=job_id,
                    project_id=principal.project_id,
                )
    for pattern in _ARTIFACT_PATHS:
        match = pattern.match(path)
        if not match:
            continue
        artifact_id = unquote(match.group(1))
        if artifact_id in _NON_ID_SEGMENTS:
            return
        try:
            ensure_artifact_access(db, principal, artifact_id, permission="read")
        except AuthorizationDenied:
            raise
        except Exception:
            return
        return


def ensure_artifact_access(
    db: Any, principal: Principal, artifact_id: str, *, permission: str = "read"
) -> None:
    """Check ownership or an explicit ACL grant for one artifact."""
    project = db.artifact_project(artifact_id)
    if project is None:
        return  # unknown artifact: the route returns artifact.not_found
    if principal.is_admin or project == principal.project_id:
        return
    grants = {
        row["permission"]
        for row in db.artifact_acls(artifact_id)
        if row["project_id"] == principal.project_id
    }
    # A read grant never authorizes a write; a write grant implies read access.
    allowed = "write" in grants if permission == "write" else bool(grants)
    if not allowed:
        raise AuthorizationDenied(
            "artifact belongs to another project and was not shared with this one",
            artifact_id=artifact_id,
            project_id=principal.project_id,
            permission=permission,
        )


def enforce_artifact_scope(db: Any, artifact_id: str, *, permission: str = "read") -> None:
    ensure_artifact_access(db, current_principal(), artifact_id, permission=permission)


def require_project_access(project_id: str) -> None:
    """Refuse a concrete resource that belongs to another project."""
    principal = current_principal()
    if principal.is_admin or str(project_id) == principal.project_id:
        return
    raise AuthorizationDenied(
        "resource belongs to another project",
        project_id=principal.project_id,
        resource_project=str(project_id),
    )


def project_filter(project_id: str | None = None) -> str | None:
    """Resolve a list filter: admins may cross projects, others are pinned."""
    principal = current_principal()
    if project_id is not None:
        if not principal.is_admin and project_id != principal.project_id:
            raise AuthorizationDenied(
                "cannot list another project's resources",
                project_id=principal.project_id,
                requested_project=project_id,
            )
        return project_id
    return None if principal.is_admin else principal.project_id


__all__ = [
    "ADMIN_SCOPES",
    "ANONYMOUS",
    "Principal",
    "SCOPES",
    "TokenAuthority",
    "bearer_token",
    "current_principal",
    "current_project",
    "enforce_artifact_scope",
    "enforce_resource_scope",
    "ensure_artifact_access",
    "principal_context",
    "project_filter",
    "require_project_access",
    "required_scope",
]
