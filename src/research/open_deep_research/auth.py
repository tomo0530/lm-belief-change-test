# auth.py  — single-file dev auth that works with:  "auth": { "path": "auth:verify" }
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping

from langgraph_sdk import Auth

# IMPORTANT: expose *verify* as the Auth instance so the loader at "auth:verify" finds an Auth, not a function
verify = Auth()


# Concrete user object that satisfies the BaseUser protocol + behaves like a Mapping (no Protocol instantiation)
@dataclass
class DevUser(Mapping[str, Any]):
    identity: str
    display_name: str | None = None
    is_authenticated: bool = True
    permissions: list[str] = field(default_factory=list)

    # Mapping interface (middleware may treat user like a dict)
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(("identity", "display_name", "is_authenticated", "permissions"))

    def __len__(self) -> int:
        return 4


# This function is registered on the Auth instance above; DO NOT change the exported name "verify"
@verify.authenticate
async def _authenticate(authorization: str | None) -> DevUser:
    # Accept "Bearer <token>" or plain "<token>"
    token = (authorization or "").split(" ", 1)[-1].strip()
    expected = os.getenv("AUTH_TOKEN", "dev")

    if not token:
        raise Auth.exceptions.HTTPException(
            403, {"message": "Authorization header missing or malformed"}
        )
    if token != expected:
        raise Auth.exceptions.HTTPException(401, {"message": "Invalid token"})

    # Return a concrete user object (NOT a dict, NOT BaseUser Protocol)
    return DevUser(identity="hc-dev", display_name="HC Dev")
