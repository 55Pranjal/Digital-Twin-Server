"""
auth.py
-------
Supabase JWT verification + role/ownership guards for FastAPI routes.

The client authenticates directly against Supabase Auth (supabase-js) and
sends the resulting access token as `Authorization: Bearer <token>` on every
API call. This module verifies that token's signature, then looks up the
caller's app role (teacher / student) and linked student_id from the
`user_profiles` table.

Supabase projects sign access tokens one of two ways, and this module
supports both (picked per-token from its `alg` header):
  - Newer projects (JWT Signing Keys): asymmetric ES256, verified against
    the project's public JWKS — no shared secret needed.
  - Legacy projects: symmetric HS256, verified against SUPABASE_JWT_SECRET.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import jwt
from dotenv import load_dotenv  # type: ignore[import-untyped]
from fastapi import Header, HTTPException
from jwt import PyJWKClient

import db

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "")
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
JWT_AUDIENCE = "authenticated"

_jwks_client = (
    PyJWKClient(f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json")
    if SUPABASE_URL
    else None
)


@dataclass
class CurrentUser:
    user_id: str
    email: str | None
    role: str | None          # "teacher" | "student" | None (profile not yet completed)
    student_id: int | None    # set for role == "student" once linked


def _decode(token: str) -> dict:
    try:
        alg = jwt.get_unverified_header(token).get("alg", "HS256")

        if alg == "HS256":
            if not JWT_SECRET:
                raise HTTPException(
                    status_code=500,
                    detail="Server misconfigured: SUPABASE_JWT_SECRET is not set.",
                )
            return jwt.decode(
                token, JWT_SECRET, algorithms=["HS256"], audience=JWT_AUDIENCE
            )

        # Asymmetric (ES256/RS256) — verify against the project's public JWKS.
        if _jwks_client is None:
            raise HTTPException(
                status_code=500,
                detail="Server misconfigured: SUPABASE_URL is not set.",
            )
        signing_key = _jwks_client.get_signing_key_from_jwt(token)
        return jwt.decode(
            token, signing_key.key, algorithms=[alg], audience=JWT_AUDIENCE
        )
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail=f"Invalid or expired token: {exc}")


def get_current_user(authorization: str | None = Header(default=None)) -> CurrentUser:
    """
    FastAPI dependency — requires a valid Supabase access token.
    Role/student_id are None if the user has authenticated but not yet
    completed profile setup (see /api/v1/auth/register-* endpoints).
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token.")

    token = authorization.split(" ", 1)[1].strip()
    claims = _decode(token)
    user_id = claims.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Token missing subject.")

    profile = db.get_user_profile(user_id)
    return CurrentUser(
        user_id=user_id,
        email=claims.get("email"),
        role=profile["role"] if profile else None,
        student_id=profile.get("student_id") if profile else None,
    )


def require_teacher(user: CurrentUser) -> CurrentUser:
    if user.role != "teacher":
        raise HTTPException(status_code=403, detail="Teacher role required.")
    return user


def require_self_or_teacher(user: CurrentUser, student_id: int) -> CurrentUser:
    if user.role == "teacher":
        return user
    if user.role == "student" and user.student_id == student_id:
        return user
    raise HTTPException(
        status_code=403,
        detail="Not authorized to access this student's data.",
    )
