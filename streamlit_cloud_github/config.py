from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
_ENV_PATH = PROJECT_ROOT / ".env"

SERENE_API_BASE_URL: str = ""
SERENE_API_TOKEN: str = ""
SERENE_API_TIMEOUT: int = 30
SERENE_AUTH_SCHEME: str = "Token"
SERENE_AIDA_ARCHIVE_START: str = "2024-09-28T00:00:00Z"


def _parse_timeout(value: object, default: int = 30) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _load_env_file() -> None:
    if not _ENV_PATH.exists():
        return

    for encoding in ("utf-8", "utf-8-sig", "utf-16", "utf-16-le"):
        try:
            load_dotenv(_ENV_PATH, encoding=encoding)
            return
        except UnicodeDecodeError:
            continue

    logger.warning(
        "Could not decode .env — re-save as UTF-8. "
        "Using environment / Streamlit secrets only."
    )


def _read_os_env() -> None:
    global SERENE_API_BASE_URL, SERENE_API_TOKEN, SERENE_API_TIMEOUT
    global SERENE_AUTH_SCHEME, SERENE_AIDA_ARCHIVE_START

    SERENE_API_BASE_URL = os.getenv("SERENE_API_BASE_URL", SERENE_API_BASE_URL).strip()
    SERENE_API_TOKEN = os.getenv("SERENE_API_TOKEN", SERENE_API_TOKEN).strip()
    SERENE_API_TIMEOUT = _parse_timeout(
        os.getenv("SERENE_API_TIMEOUT", str(SERENE_API_TIMEOUT)),
    )
    SERENE_AUTH_SCHEME = (
        os.getenv("SERENE_AUTH_SCHEME", SERENE_AUTH_SCHEME).strip() or "Token"
    )
    SERENE_AIDA_ARCHIVE_START = (
        os.getenv("SERENE_AIDA_ARCHIVE_START", SERENE_AIDA_ARCHIVE_START).strip()
        or "2024-09-28T00:00:00Z"
    )


def _get_secret(secrets: object, key: str) -> str | None:
    try:
        if key in secrets:
            return str(secrets[key]).strip()
        if "serene" in secrets and key in secrets["serene"]:
            return str(secrets["serene"][key]).strip()
    except Exception:
        return None
    return None


def _load_streamlit_secrets() -> None:
    try:
        import streamlit as st
    except ImportError:
        return

    try:
        secrets = st.secrets
    except Exception:
        return

    global SERENE_API_BASE_URL, SERENE_API_TOKEN, SERENE_API_TIMEOUT
    global SERENE_AUTH_SCHEME, SERENE_AIDA_ARCHIVE_START

    base = _get_secret(secrets, "SERENE_API_BASE_URL")
    token = _get_secret(secrets, "SERENE_API_TOKEN")
    timeout = _get_secret(secrets, "SERENE_API_TIMEOUT")
    scheme = _get_secret(secrets, "SERENE_AUTH_SCHEME")
    archive_start = _get_secret(secrets, "SERENE_AIDA_ARCHIVE_START")

    if base:
        SERENE_API_BASE_URL = base
    if token:
        SERENE_API_TOKEN = token
    if timeout:
        SERENE_API_TIMEOUT = _parse_timeout(timeout)
    if scheme:
        SERENE_AUTH_SCHEME = scheme
    if archive_start:
        SERENE_AIDA_ARCHIVE_START = archive_start


def reload_config() -> None:
    _load_env_file()
    _read_os_env()
    _load_streamlit_secrets()


reload_config()


def validate_config() -> list[str]:
    messages: list[str] = []

    if not SERENE_API_BASE_URL:
        messages.append(
            "SERENE_API_BASE_URL is not set. "
            "For local dev: copy .env.example to .env. "
            "For Streamlit Cloud: add secrets in the app settings."
        )

    if not SERENE_API_TOKEN:
        messages.append(
            "SERENE_API_TOKEN is not set. "
            "Without it, authenticated SERENE API endpoints may be unavailable."
        )

    return messages
