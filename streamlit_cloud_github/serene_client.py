from __future__ import annotations

import logging
import json
import os
import time
from io import StringIO
from typing import Any

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import (
    SERENE_API_BASE_URL,
    SERENE_API_TIMEOUT,
    SERENE_API_TOKEN,
    SERENE_AUTH_SCHEME,
)

KP_AP_CACHE_TTL_SECONDS = int(os.getenv("SERENE_KP_AP_CACHE_TTL", "3600"))
GFZ_INDEX_CACHE_MAX_ENTRIES = 64
GFZ_INDEX_CACHE_MAX_ENTRIES_ENV = "SERENE_GFZ_INDEX_CACHE_MAX_ENTRIES"
GFZ_KP_AP_BASE_URL = "https://kp.gfz.de"
GFZ_KP_AP_JSON_PATH = "/app/json/"
GFZ_KP_FORECAST_URL = (
    "https://spaceweather.gfz.de/fileadmin/Kp-Forecast/CSV/"
    "kp_product_file_FORECAST_PAGER_SWIFT_LAST.json"
)
AIDA_RAW_CACHE_MAX_ENTRIES = 16
AIDA_RAW_CACHE_MAX_ENTRIES_ENV = "SERENE_AIDA_RAW_CACHE_MAX_ENTRIES"

logger = logging.getLogger(__name__)



def _aida_raw_cache_max_entries() -> int:
    try:
        return int(os.getenv(
            AIDA_RAW_CACHE_MAX_ENTRIES_ENV,
            str(AIDA_RAW_CACHE_MAX_ENTRIES),
        ))
    except ValueError:
        return AIDA_RAW_CACHE_MAX_ENTRIES


def _gfz_index_cache_max_entries() -> int:
    try:
        return int(os.getenv(
            GFZ_INDEX_CACHE_MAX_ENTRIES_ENV,
            str(GFZ_INDEX_CACHE_MAX_ENTRIES),
        ))
    except ValueError:
        return GFZ_INDEX_CACHE_MAX_ENTRIES


def normalise_aida_request_time(value: str) -> pd.Timestamp:
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        raise ValueError(f"Invalid requested AIDA time: {value}")
    rounded_epoch = np.round(parsed.timestamp() / 300.0) * 300.0
    return pd.to_datetime(rounded_epoch, unit="s", utc=True)


def _safe_response_detail(
    response: requests.Response,
    token: str | None = None,
) -> str:
    text = str(getattr(response, "text", "") or "").strip()
    if not text or text.startswith("<"):
        return ""
    content_type = str((getattr(response, "headers", {}) or {}).get(
        "Content-Type", ""
    )).lower()
    if "json" in content_type:
        try:
            body = json.loads(text)
        except (TypeError, ValueError):
            body = None
        if isinstance(body, dict):
            text = str(
                body.get("detail")
                or body.get("error")
                or body.get("message")
                or ""
            ).strip()
    if token:
        text = text.replace(token, "[redacted]")
    return " ".join(text.split())[:240]

ENDPOINTS: dict[str, str] = {
    "aida_raw_output": "/api/download-output/",
    "aida_raw_forecast": "/api/download-forecast/",
}


class SereneClient:

    _gfz_index_cache: dict[
        tuple[str, str, str], tuple[float, object]
    ] = {}
    _gfz_kp_forecast_cache: tuple[float, pd.DataFrame] | None = None
    _aida_raw_cache: dict[tuple[str, str, str, str, int | None], bytes] = {}

    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        timeout: int | None = None,
        auth_scheme: str | None = None,
    ) -> None:
        self.base_url = (base_url or SERENE_API_BASE_URL).rstrip("/")
        self.token = token or SERENE_API_TOKEN
        self.timeout = timeout if timeout is not None else SERENE_API_TIMEOUT
        self.auth_scheme = (auth_scheme or SERENE_AUTH_SCHEME).strip()
        self.kp_ap_source_latest_time: pd.Timestamp | None = None
        self.kp_ap_data_statuses: list[str] = []
        self.kp_ap_missing_indices: list[str] = []

        self._session = requests.Session()
        retry_strategy = Retry(
            total=2,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)


    def _auth_headers(self) -> dict[str, str]:
        if not self.token:
            return {}
        return {"Authorization": f"{self.auth_scheme} {self.token}"}

    def _request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
    ) -> tuple[bool, str, Any]:
        if not self.base_url:
            return False, "SERENE_API_BASE_URL is not configured.", None

        url = f"{self.base_url}{endpoint}"
        headers = self._auth_headers()

        try:
            response = self._session.request(
                method=method.upper(),
                url=url,
                headers=headers,
                params=params,
                json=json,
                data=data,
                timeout=self.timeout,
            )
        except requests.exceptions.Timeout:
            msg = f"SERENE API request timed out after {self.timeout}s: {url}"
            logger.warning(msg)
            return False, msg, None
        except requests.exceptions.ConnectionError as exc:
            detail = self._redact_token(str(exc))
            msg = (
                f"Cannot connect to SERENE API at {self.base_url}. "
                f"Check SERENE_API_BASE_URL and network. ({detail})"
            )
            logger.warning(msg)
            return False, msg, None
        except requests.exceptions.RequestException as exc:
            msg = f"SERENE API request failed: {self._redact_token(str(exc))}"
            logger.warning(msg)
            return False, msg, None

        if response.status_code == 401:
            msg = (
                "SERENE API returned 401 Unauthorized. "
                "Check SERENE_API_TOKEN. Official auth is: Authorization: Token <token>."
            )
            logger.warning(msg)
            return False, msg, None

        if response.status_code == 403:
            msg = "SERENE API returned 403 Forbidden. Token may lack permission."
            logger.warning(msg)
            return False, msg, None

        if response.status_code == 404:
            msg = f"SERENE API endpoint not found (404): {url}"
            logger.warning(msg)
            return False, msg, None

        if response.status_code >= 500:
            msg = f"SERENE API server error ({response.status_code}): {url}"
            logger.warning(msg)
            return False, msg, None

        if not response.ok:
            msg = f"SERENE API unexpected status {response.status_code}: {url}"
            logger.warning(msg)
            return False, msg, None

        if not response.content:
            return True, "OK (empty response body)", None

        try:
            body = response.json()
        except ValueError:
            text = response.text.strip()
            if not text:
                return True, "OK (empty response)", None
            return True, "OK (non-JSON response)", text

        if body is None or body == "" or body == [] or body == {}:
            return True, "OK (no data in response)", body

        return True, "OK", body


    def test_connection(self) -> tuple[bool, str]:
        if not self.base_url:
            return False, "SERENE_API_BASE_URL is not configured."
        if not self.token:
            return False, "SERENE_API_TOKEN is not configured."

        ok, msg, _payload = self.download_aida_raw_output(None, "ultra")
        if ok:
            return (
                True,
                f"Connected to SERENE AIDA raw-output API at {self.base_url}.",
            )

        if "401" in msg or "403" in msg:
            return False, msg

        return False, msg

    def download_aida_raw_output(
        self,
        requested_time: str | None,
        latency: str,
    ) -> tuple[bool, str, bytes | None]:
        if latency not in {"ultra", "rapid", "final"}:
            return False, f"Unsupported AIDA latency: {latency}", None
        if not self.base_url:
            return False, "SERENE_API_BASE_URL is not configured.", None
        if not self.token:
            return False, "SERENE_API_TOKEN is not configured.", None

        if requested_time is None:
            cache_time = "latest"
            request_data: dict[str, Any] = {
                "latest": True,
                "product": latency,
                "file_type": "raw",
            }
        else:
            try:
                parsed = normalise_aida_request_time(requested_time)
            except ValueError as exc:
                return False, str(exc), None
            cache_time = parsed.isoformat()
            upstream_file_time = parsed.tz_convert("UTC").tz_localize(None).isoformat()
            request_data = {
                "file_time": upstream_file_time,
                "product": latency,
                "file_type": "raw",
            }

        cache_key = (self.base_url, "analysis", cache_time, latency, None)
        cached = type(self)._get_cached_aida_raw(cache_key)
        if cached is not None:
            return True, f"Loaded cached AIDA raw state for {cache_time}.", cached

        ok, message, content = self._request_aida_hdf5(
            endpoint=ENDPOINTS["aida_raw_output"],
            request_data=request_data,
            api_name="raw-output",
            auth_resource="AIDA raw output",
        )
        if not ok or content is None:
            return False, message, None

        type(self)._cache_aida_raw(cache_key, content)
        return True, f"Downloaded AIDA raw state for {cache_time}.", content

    def download_aida_forecast(
        self,
        requested_time: str | None,
        latency: str,
        period_minutes: int,
    ) -> tuple[bool, str, bytes | None]:
        if latency not in {"ultra", "rapid", "final"}:
            return False, f"Unsupported AIDA latency: {latency}", None
        if period_minutes not in {30, 90, 180, 360}:
            return False, f"Unsupported AIDA forecast period: {period_minutes}", None
        if requested_time is None:
            return False, "AIDA forecasts require an explicit requested time.", None
        if not self.base_url:
            return False, "SERENE_API_BASE_URL is not configured.", None
        if not self.token:
            return False, "SERENE_API_TOKEN is not configured.", None

        try:
            parsed = normalise_aida_request_time(requested_time)
        except ValueError as exc:
            return False, str(exc), None

        cache_time = parsed.isoformat()
        cache_key = (
            self.base_url,
            "forecast",
            cache_time,
            latency,
            period_minutes,
        )
        cached = type(self)._get_cached_aida_raw(cache_key)
        if cached is not None:
            return (
                True,
                f"Loaded cached AIDA raw forecast for {cache_time} "
                f"({period_minutes} minutes).",
                cached,
            )

        request_data: dict[str, Any] = {
            "file_time": parsed.tz_convert("UTC").tz_localize(None).isoformat(),
            "product": latency,
            "file_type": "raw",
            "period": period_minutes,
        }
        ok, message, content = self._request_aida_hdf5(
            endpoint=ENDPOINTS["aida_raw_forecast"],
            request_data=request_data,
            api_name="forecast",
            auth_resource="AIDA forecast",
        )
        if not ok or content is None:
            return False, message, None

        type(self)._cache_aida_raw(cache_key, content)
        return (
            True,
            f"Downloaded AIDA raw forecast for {cache_time} "
            f"({period_minutes} minutes).",
            content,
        )

    def _request_aida_hdf5(
        self,
        *,
        endpoint: str,
        request_data: dict[str, Any],
        api_name: str,
        auth_resource: str,
    ) -> tuple[bool, str, bytes | None]:
        url = f"{self.base_url}{endpoint}"
        try:
            response = self._session.request(
                method="GET",
                url=url,
                headers=self._auth_headers(),
                data=request_data,
                timeout=self.timeout,
            )
        except requests.exceptions.RequestException as exc:
            detail = self._redact_token(str(exc))
            return False, f"SERENE AIDA {api_name} request failed: {detail}", None

        if response.status_code in {401, 403}:
            return False, f"SERENE rejected the API token for {auth_resource}.", None
        if not response.ok:
            detail = _safe_response_detail(response, token=self.token)
            request_detail = _describe_aida_request(request_data)
            suffix = f" {detail}" if detail else ""
            return (
                False,
                f"SERENE AIDA {api_name} API returned status "
                f"{response.status_code} for {request_detail}.{suffix}",
                None,
            )

        content = bytes(getattr(response, "content", b""))
        headers = getattr(response, "headers", {}) or {}
        content_type = str(headers.get("Content-Type", "")).lower()
        if (
            not content
            or "html" in content_type
            or not content.startswith(b"\x89HDF\r\n\x1a\n")
        ):
            return False, f"SERENE AIDA {api_name} API returned a non-HDF5 response.", None

        return True, "OK", content

    def _redact_token(self, detail: str) -> str:
        if self.token:
            return detail.replace(self.token, "[redacted]")
        return detail

    @classmethod
    def _get_cached_aida_raw(
        cls,
        cache_key: tuple[str, str, str, str, int | None],
    ) -> bytes | None:
        max_entries = _aida_raw_cache_max_entries()
        if max_entries <= 0:
            cls._aida_raw_cache.clear()
            return None
        while len(cls._aida_raw_cache) > max_entries:
            oldest_key = next(iter(cls._aida_raw_cache))
            del cls._aida_raw_cache[oldest_key]

        cached = cls._aida_raw_cache.pop(cache_key, None)
        if cached is not None:
            cls._aida_raw_cache[cache_key] = cached
        return cached

    @classmethod
    def _cache_aida_raw(
        cls,
        cache_key: tuple[str, str, str, str, int | None],
        content: bytes,
    ) -> None:
        max_entries = _aida_raw_cache_max_entries()

        if max_entries <= 0:
            cls._aida_raw_cache.clear()
            return

        cls._aida_raw_cache.pop(cache_key, None)
        cls._aida_raw_cache[cache_key] = content
        while len(cls._aida_raw_cache) > max_entries:
            oldest_key = next(iter(cls._aida_raw_cache))
            del cls._aida_raw_cache[oldest_key]

    def fetch_kp_ap_indices(
        self,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> tuple[bool, str, pd.DataFrame]:
        self.kp_ap_source_latest_time = None
        self.kp_ap_data_statuses = []
        self.kp_ap_missing_indices = []
        start = _parse_optional_utc(start_time)
        end = _parse_optional_utc(end_time)
        if start is None or end is None:
            return (
                False,
                "GFZ Kp/ap JSON requires valid start and end times.",
                pd.DataFrame(),
            )
        if start > end:
            return (
                False,
                "GFZ Kp/ap JSON start time must not follow end time.",
                pd.DataFrame(),
            )

        normalized_start = start.strftime("%Y-%m-%dT%H:%M:%SZ")
        normalized_end = end.strftime("%Y-%m-%dT%H:%M:%SZ")
        frames: dict[str, pd.DataFrame] = {}
        errors: dict[str, str] = {}
        for index in ("Kp", "ap"):
            ok, message, payload = self._fetch_gfz_json_index(
                index,
                normalized_start,
                normalized_end,
            )
            if not ok:
                errors[index] = message
                continue
            frame = self.parse_gfz_json_index(payload, index)
            if frame.empty:
                errors[index] = "GFZ returned no valid rows."
                continue
            frame = frame[
                frame["time"].between(start, end, inclusive="both")
            ].copy()
            if frame.empty:
                errors[index] = "GFZ returned no rows in the requested range."
                continue
            frames[index] = frame

        self.kp_ap_missing_indices = [
            index for index in ("Kp", "ap") if index not in frames
        ]
        if "Kp" not in frames:
            detail = errors.get("Kp", "GFZ returned no usable Kp rows.")
            return False, f"GFZ Kp unavailable: {detail}", pd.DataFrame()

        df = pd.concat(list(frames.values()), ignore_index=True)
        df = df.sort_values(["time", "variable"]).reset_index(drop=True)
        self.kp_ap_source_latest_time = pd.Timestamp(df["time"].max())
        if "data_status" in df.columns:
            self.kp_ap_data_statuses = sorted(
                df["data_status"].dropna().astype(str).unique().tolist()
            )
        df.attrs["kp_ap_latest_time"] = self.kp_ap_source_latest_time
        df.attrs["kp_ap_missing_indices"] = self.kp_ap_missing_indices.copy()
        df.attrs["kp_ap_source"] = "GFZ Kp/ap JSON service"

        message = f"Loaded {len(df)} Kp/ap row(s) from GFZ JSON service."
        if "ap" not in frames:
            message += f" ap unavailable: {errors.get('ap', 'no usable rows')}"
        return True, message, df

    def _fetch_gfz_json_index(
        self,
        index: str,
        normalized_start: str,
        normalized_end: str,
    ) -> tuple[bool, str, object]:
        cache_key = (index, normalized_start, normalized_end)
        now = time.monotonic()
        type(self)._prune_gfz_index_cache(now)
        cached = type(self)._gfz_index_cache.get(cache_key)
        if cached and now - cached[0] < KP_AP_CACHE_TTL_SECONDS:
            return True, "OK (cached)", cached[1]

        ok, message, text = self._request_from_base(
            "GET",
            GFZ_KP_AP_BASE_URL,
            GFZ_KP_AP_JSON_PATH,
            params={
                "start": normalized_start,
                "end": normalized_end,
                "index": index,
            },
        )
        if not ok or not isinstance(text, str):
            return False, message, None
        try:
            payload = json.loads(text)
        except (TypeError, ValueError):
            return False, "GFZ returned malformed JSON.", None
        if not isinstance(payload, dict):
            return False, "GFZ returned an unexpected JSON structure.", None

        type(self)._cache_gfz_index(cache_key, payload, now=time.monotonic())
        return True, "OK", payload

    @classmethod
    def _prune_gfz_index_cache(cls, now: float) -> None:
        expired = [
            key
            for key, (stored_at, _payload) in cls._gfz_index_cache.items()
            if now - stored_at >= KP_AP_CACHE_TTL_SECONDS
        ]
        for key in expired:
            del cls._gfz_index_cache[key]

    @classmethod
    def _cache_gfz_index(
        cls,
        cache_key: tuple[str, str, str],
        payload: object,
        *,
        now: float,
    ) -> None:
        max_entries = _gfz_index_cache_max_entries()
        if max_entries <= 0:
            cls._gfz_index_cache.clear()
            return
        cls._gfz_index_cache.pop(cache_key, None)
        cls._gfz_index_cache[cache_key] = (now, payload)
        while len(cls._gfz_index_cache) > max_entries:
            oldest_key = next(iter(cls._gfz_index_cache))
            del cls._gfz_index_cache[oldest_key]

    def fetch_gfz_kp_forecast(self) -> tuple[bool, str, pd.DataFrame]:
        cached = type(self)._gfz_kp_forecast_cache
        if cached and time.monotonic() - cached[0] < KP_AP_CACHE_TTL_SECONDS:
            return True, "Loaded cached GFZ Kp ensemble forecast.", cached[1].copy()

        try:
            response = self._session.request(
                method="GET",
                url=GFZ_KP_FORECAST_URL,
                headers={},
                params=None,
                timeout=self.timeout,
            )
        except requests.exceptions.RequestException as exc:
            message = (
                "GFZ Kp ensemble forecast request failed: "
                f"{self._redact_token(str(exc))}"
            )
            logger.warning(message)
            return False, message, pd.DataFrame()

        if not response.ok:
            message = (
                "GFZ Kp ensemble forecast returned status "
                f"{response.status_code}."
            )
            logger.warning(message)
            return False, message, pd.DataFrame()

        try:
            payload = json.loads(str(response.text).strip())
        except (TypeError, ValueError):
            return False, "GFZ Kp ensemble forecast returned malformed JSON.", pd.DataFrame()

        frame = self.parse_gfz_kp_forecast(payload)
        if frame.empty:
            return False, "GFZ Kp ensemble forecast has no valid rows.", frame

        issue_time = pd.to_datetime(
            (getattr(response, "headers", {}) or {}).get("Last-Modified"),
            errors="coerce",
            utc=True,
        )
        if pd.isna(issue_time):
            return (
                False,
                "GFZ Kp ensemble forecast has no valid Last-Modified time.",
                pd.DataFrame(),
            )
        frame["issue_time"] = pd.Timestamp(issue_time)
        type(self)._gfz_kp_forecast_cache = (time.monotonic(), frame.copy())
        return True, f"Loaded {len(frame)} GFZ Kp ensemble forecast row(s).", frame

    @staticmethod
    def parse_kp_ap_csv(
        csv_text: str,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> pd.DataFrame:
        frame, _latest_time = SereneClient._parse_kp_ap_csv_with_latest(
            csv_text,
            start_time=start_time,
            end_time=end_time,
        )
        return frame

    @staticmethod
    def parse_gfz_kp_ap(
        text: str,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> pd.DataFrame:
        frame, _latest_time = SereneClient._parse_gfz_kp_ap_with_latest(
            text,
            start_time=start_time,
            end_time=end_time,
        )
        return frame

    @staticmethod
    def parse_gfz_json_index(payload: object, index: str) -> pd.DataFrame:
        if index not in {"Kp", "ap"} or not isinstance(payload, dict):
            return pd.DataFrame()

        timestamps = payload.get("datetime")
        values = payload.get(index)
        statuses = payload.get("status")
        if not all(isinstance(items, list) for items in (
            timestamps, values, statuses
        )):
            return pd.DataFrame()
        if not timestamps or not (
            len(timestamps) == len(values) == len(statuses)
        ):
            return pd.DataFrame()

        times = pd.to_datetime(timestamps, errors="coerce", utc=True)
        numeric = pd.to_numeric(pd.Series(values), errors="coerce")
        valid = times.notna() & numeric.notna() & np.isfinite(numeric)
        valid &= numeric.ne(-1.0)

        rows: list[dict[str, Any]] = []
        status_names = {"def": "definitive", "pre": "preliminary"}
        for position in np.flatnonzero(valid):
            rows.append({
                "time": times[position],
                "lat": None,
                "lon": None,
                "alt": None,
                "variable": index,
                "value": float(numeric.iloc[position]),
                "model": "GFZ Geomagnetic Indices",
                "source": "GFZ Kp/ap JSON service",
                "data_status": status_names.get(
                    str(statuses[position]).strip().lower(),
                    str(statuses[position]).strip().lower() or "unknown",
                ),
            })

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)

    @staticmethod
    def parse_gfz_kp_forecast(payload: object) -> pd.DataFrame:
        if not isinstance(payload, dict):
            return pd.DataFrame()

        field_names = ("Time (UTC)", "median", "maximum", "prob >= 8")
        fields = [payload.get(name) for name in field_names]
        if not all(isinstance(field, dict) and field for field in fields):
            return pd.DataFrame()

        row_keys = set(fields[0])
        if any(set(field) != row_keys for field in fields[1:]):
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []
        for row_key in row_keys:
            timestamp = pd.to_datetime(
                fields[0][row_key],
                format="%d-%m-%Y %H:%M",
                errors="coerce",
                utc=True,
            )
            numeric = pd.to_numeric(pd.Series([
                fields[1][row_key],
                fields[2][row_key],
                fields[3][row_key],
            ]), errors="coerce")
            if pd.isna(timestamp) or numeric.isna().any():
                return pd.DataFrame()
            median, maximum, probability = (float(value) for value in numeric)
            if not all(np.isfinite([median, maximum, probability])):
                return pd.DataFrame()
            if not (0.0 <= median <= 9.0 and 0.0 <= maximum <= 9.0):
                return pd.DataFrame()
            if maximum < median or not 0.0 <= probability <= 1.0:
                return pd.DataFrame()
            rows.append({
                "interval_start": pd.Timestamp(timestamp),
                "median": median,
                "maximum": maximum,
                "probability_kp_ge_8": probability,
                "source": "GFZ official PAGER/SWIFT ensemble forecast",
            })

        return pd.DataFrame(rows).sort_values("interval_start").reset_index(drop=True)

    @staticmethod
    def _parse_gfz_kp_ap_with_latest(
        text: str,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> tuple[pd.DataFrame, pd.Timestamp | None]:
        columns = [
            "year", "month", "day", "start_hour", "mid_hour",
            "days", "days_mid", "Kp", "ap", "definitive",
        ]
        try:
            raw = pd.read_csv(
                StringIO(text),
                sep=r"\s+",
                comment="#",
                names=columns,
                header=None,
            )
        except (TypeError, ValueError, pd.errors.ParserError):
            return pd.DataFrame(), None
        if raw.empty:
            return pd.DataFrame(), None

        for column in columns:
            raw[column] = pd.to_numeric(raw[column], errors="coerce")
        dates = pd.to_datetime(
            raw[["year", "month", "day"]],
            errors="coerce",
            utc=True,
        )
        raw["time"] = dates + pd.to_timedelta(raw["start_hour"], unit="h")
        raw = raw.dropna(subset=["time"])
        valid_measurement = (
            (raw["Kp"].notna() & raw["Kp"].ne(-1.0))
            | (raw["ap"].notna() & raw["ap"].ne(-1.0))
        )
        raw = raw[valid_measurement].copy()
        if raw.empty:
            return pd.DataFrame(), None

        latest_time = pd.Timestamp(raw["time"].max())
        start = _parse_optional_utc(start_time)
        end = _parse_optional_utc(end_time)
        if start is not None and end is not None and start > end:
            start, end = end, start
        if start is not None:
            raw = raw[raw["time"] >= start]
        if end is not None:
            raw = raw[raw["time"] <= end]

        rows: list[dict[str, Any]] = []
        for _, row in raw.iterrows():
            status = (
                "definitive" if row["definitive"] == 1 else "preliminary"
            )
            for variable, missing in (("Kp", -1.0), ("ap", -1.0)):
                value = row[variable]
                if pd.isna(value) or float(value) == missing:
                    continue
                rows.append({
                    "time": row["time"],
                    "lat": None,
                    "lon": None,
                    "alt": None,
                    "variable": variable,
                    "value": float(value),
                    "model": "GFZ Geomagnetic Indices",
                    "source": "GFZ Kp/ap nowcast",
                    "data_status": status,
                })

        return pd.DataFrame(rows), latest_time

    @staticmethod
    def _parse_kp_ap_csv_with_latest(
        csv_text: str,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> tuple[pd.DataFrame, pd.Timestamp | None]:
        raw = pd.read_csv(StringIO(csv_text))
        if raw.empty or "time" not in raw.columns:
            return pd.DataFrame(), None

        raw["time"] = pd.to_datetime(raw["time"], errors="coerce", utc=True)
        raw = raw.dropna(subset=["time"])
        latest_time = (
            pd.Timestamp(raw["time"].max())
            if not raw.empty
            else None
        )

        start = _parse_optional_utc(start_time)
        end = _parse_optional_utc(end_time)
        if start is not None and end is not None and start > end:
            start, end = end, start
        if start is not None:
            raw = raw[raw["time"] >= start]
        if end is not None:
            raw = raw[raw["time"] <= end]

        rows: list[dict[str, Any]] = []
        for _, row in raw.iterrows():
            for variable in ("Kp", "ap"):
                if variable not in raw.columns:
                    continue
                value = pd.to_numeric(pd.Series([row[variable]]), errors="coerce").iloc[0]
                if pd.isna(value):
                    continue
                rows.append({
                    "time": row["time"],
                    "lat": None,
                    "lon": None,
                    "alt": None,
                    "variable": variable,
                    "value": float(value),
                    "model": "SERENE Indices",
                    "source": "SERENE API Kp/ap",
                })

        return pd.DataFrame(rows), latest_time

    def _request_from_base(
        self,
        method: str,
        base_url: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
    ) -> tuple[bool, str, Any]:
        url = f"{base_url.rstrip('/')}{endpoint}"
        try:
            response = self._session.request(
                method=method.upper(),
                url=url,
                headers={},
                params=params,
                timeout=self.timeout,
            )
        except requests.exceptions.RequestException as exc:
            msg = f"Public resource request failed: {self._redact_token(str(exc))}"
            logger.warning(msg)
            return False, msg, None

        if not response.ok:
            msg = f"Public resource unexpected status {response.status_code}: {url}"
            logger.warning(msg)
            return False, msg, None

        text = response.text.strip()
        if not text:
            return False, "Public resource returned an empty response.", None
        return True, "OK", text


    def parse_response_to_dataframe(
        self,
        response_data: Any,
        model: str | None = None,
    ) -> pd.DataFrame:
        if response_data is None:
            logger.warning("parse_response_to_dataframe received None.")
            return pd.DataFrame()

        if isinstance(response_data, list) and response_data and isinstance(response_data[0], dict):
            if "response" in response_data[0]:
                frames = []
                for item in response_data:
                    sub = self.parse_response_to_dataframe(
                        item.get("response"),
                        model=item.get("model") or model,
                    )
                    if sub.empty:
                        continue
                    if item.get("lat") is not None:
                        if "lat" not in sub.columns:
                            sub["lat"] = item["lat"]
                        else:
                            sub["lat"] = sub["lat"].fillna(item["lat"])
                    if item.get("lon") is not None:
                        if "lon" not in sub.columns:
                            sub["lon"] = item["lon"]
                        else:
                            sub["lon"] = sub["lon"].fillna(item["lon"])
                    frames.append(sub)
                if frames:
                    return pd.concat(frames, ignore_index=True)
                return pd.DataFrame()

        records = _extract_records(response_data)
        if not records:
            logger.warning("Could not extract records from SERENE response.")
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df = _normalise_columns(df)

        if model and "model" not in df.columns:
            df["model"] = model

        for col in ("time", "timestamp", "date", "datetime"):
            if col in df.columns:
                try:
                    df["time"] = pd.to_datetime(df[col])
                except (ValueError, TypeError):
                    pass
                break

        return df




def _describe_aida_request(request_data: dict[str, Any]) -> str:
    product = request_data.get("product", "unknown")
    file_type = request_data.get("file_type", "unknown")
    if request_data.get("latest"):
        time_label = "latest"
    else:
        time_label = request_data.get("file_time", "unknown time")
    period = request_data.get("period")
    if period is None:
        return f"product={product}, file_type={file_type}, file_time={time_label}"
    return (
        f"product={product}, file_type={file_type}, "
        f"file_time={time_label}, forecast={period} min"
    )


def _parse_optional_utc(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    try:
        parsed = pd.to_datetime(value, errors="coerce", utc=True)
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    return parsed


def _parse_calc_text_response(text: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        name, _, raw_val = line.partition(":")
        name = name.strip()
        raw_val = raw_val.strip()
        try:
            value = float(raw_val)
        except ValueError:
            value = raw_val
        records.append({"variable": name, "value": value})
    if not records and text.strip():
        records.append({"value": text.strip()})
    return records


def _extract_records(response_data: Any) -> list[dict[str, Any]]:
    if isinstance(response_data, str):
        parsed = _parse_calc_text_response(response_data)
        return parsed if parsed else [{"value": response_data}]

    if isinstance(response_data, list):
        if response_data and isinstance(response_data[0], dict) and "response" in response_data[0]:
            return []
        return [r for r in response_data if isinstance(r, dict)]

    if isinstance(response_data, dict):
        for key in ("data", "results", "records", "output", "variables", "grid", "parameters"):
            if key not in response_data:
                continue
            candidate = response_data[key]
            if isinstance(candidate, list):
                return [r for r in candidate if isinstance(r, dict)]
            if isinstance(candidate, dict):
                return _columnar_to_records(candidate)

        flat = _flatten_dict(response_data)
        return flat if flat else [response_data]

    return []


def _columnar_to_records(data: dict[str, Any]) -> list[dict[str, Any]]:
    if not data:
        return []
    arrays = {k: v if isinstance(v, (list, tuple)) else [v] for k, v in data.items()}
    length = max(len(v) for v in arrays.values())
    return [
        {key: values[i] if i < len(values) else None for key, values in arrays.items()}
        for i in range(length)
    ]


def _flatten_dict(data: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for key, val in data.items():
        if isinstance(val, dict) and "value" in val:
            record: dict[str, Any] = {"variable": key}
            record.update(val)
            records.append(record)
    return records


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "timestamp": "time",
        "date": "time",
        "datetime": "time",
        "t": "time",
        "latitude": "lat",
        "longitude": "lon",
        "long": "lon",
        "altitude": "alt",
        "height": "alt",
        "h": "alt",
        "var": "variable",
        "param": "variable",
        "parameter": "variable",
        "field": "variable",
        "val": "value",
        "data_value": "value",
    }
    rename = {k: v for k, v in mapping.items() if k in df.columns and v not in df.columns}
    return df.rename(columns=rename)
