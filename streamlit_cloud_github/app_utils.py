from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Any

import pandas as pd

from config import SERENE_AIDA_ARCHIVE_START


DISCLAIMER = (
    "This is a research-prototype interpretation of the available data. "
    "It is not an official ICAO warning and must not be used for operational "
    "aviation decision-making."
)

DEFAULT_AIDA_ARCHIVE_START = "2024-09-28T00:00:00Z"


def _parse_archive_start(value: str) -> pd.Timestamp:
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        parsed = pd.Timestamp(DEFAULT_AIDA_ARCHIVE_START)
    return parsed


AIDA_ARCHIVE_START_UTC = _parse_archive_start(SERENE_AIDA_ARCHIVE_START)
AIDA_ARCHIVE_START = AIDA_ARCHIVE_START_UTC.date()


def validate_requested_window(
    start_time: str,
    end_time: str,
    publication_safe_now: pd.Timestamp | None = None,
) -> str | None:
    start = pd.to_datetime(start_time, errors="coerce", utc=True)
    end = pd.to_datetime(end_time, errors="coerce", utc=True)
    if pd.isna(start) or pd.isna(end):
        return "The requested analysis window is invalid."
    if start > end:
        return "The analysis start must be before the analysis end."
    safe_now = publication_safe_now
    if safe_now is None:
        safe_now = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=15)
    if end > safe_now:
        return "The analysis end is in the unpublished future window."
    return None


def advisory_metadata_for_load(
    success: bool,
    current_sequence: int,
    generated_time: pd.Timestamp,
) -> dict[str, Any]:
    if not success:
        return {
            "sequence": int(current_sequence),
            "generated_time": None,
            "number": None,
        }
    sequence = int(current_sequence) + 1
    generated = pd.Timestamp(generated_time)
    return {
        "sequence": sequence,
        "generated_time": generated,
        "number": f"{generated.year}/{sequence:03d}",
    }


def loaded_api_state(
    status: Any,
    explicit_connected: bool | None,
    explicit_message: str,
) -> tuple[str, str]:
    if explicit_connected is True:
        return "success", explicit_message
    if explicit_connected is False:
        return "warning", explicit_message
    if getattr(status, "source", None) == "api" and bool(getattr(status, "ok", False)):
        return "success", "Live load succeeded; the SERENE API returned AIDA data."
    return "info", "Not tested. Use the sidebar connection test or load live data."


def build_provenance_metadata(
    requested_time: Any,
    actual_time: Any,
    retrieved_time: Any,
    now: Any,
    official_forecasts: int,
) -> list[dict[str, str]]:
    actual = _as_utc_timestamp(actual_time)
    current = _as_utc_timestamp(now)
    if actual is None or current is None:
        age = "N/A"
    else:
        minutes = max(0, int((current - actual).total_seconds() // 60))
        age = f"{minutes} min"
    return [
        {"label": "Requested analysis", "value": _format_utc_value(requested_time)},
        {"label": "Actual AIDA output", "value": _format_utc_value(actual_time)},
        {"label": "Retrieved", "value": _format_utc_value(retrieved_time)},
        {"label": "Data age", "value": age},
        {"label": "Forecast horizons", "value": f"{int(official_forecasts)} official"},
    ]


def _as_utc_timestamp(value: Any) -> pd.Timestamp | None:
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    return None if pd.isna(parsed) else pd.Timestamp(parsed)


def _format_utc_value(value: Any) -> str:
    parsed = _as_utc_timestamp(value)
    return "N/A" if parsed is None else parsed.strftime("%Y-%m-%d %H:%M UTC")


def combine_date_time_iso(date_value: date, time_value: time) -> str:
    return datetime.combine(date_value, time_value).strftime("%Y-%m-%dT%H:%M:%S")


def default_time_range(reference_time: datetime | None = None) -> tuple[datetime, datetime]:
    now = reference_time or datetime.now(timezone.utc)
    delayed = now - timedelta(minutes=15)
    end = delayed.replace(
        minute=(delayed.minute // 5) * 5,
        second=0,
        microsecond=0,
    )
    return end - timedelta(hours=6), end


def historical_risk_windows() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Time UTC": "2024-10-10 18:00 to 2024-10-11 03:00",
            "Peak Kp": "8.7",
            "Peak ap": "300",
            "Risk": "G4 Severe geomagnetic storm",
            "Select range": "2024-10-10T18:00:00 to 2024-10-11T02:55:00",
        },
        {
            "Time UTC": "2025-01-01 15:00 to 18:00",
            "Peak Kp": "8.0",
            "Peak ap": "207",
            "Risk": "G4 Severe geomagnetic storm",
            "Select range": "2025-01-01T15:00:00 to 2025-01-01T17:55:00",
        },
        {
            "Time UTC": "2025-11-12 00:00 to 06:00",
            "Peak Kp": "8.7",
            "Peak ap": "300",
            "Risk": "G4 Severe geomagnetic storm",
            "Select range": "2025-11-12T00:00:00 to 2025-11-12T05:55:00",
        },
        {
            "Time UTC": "2026-01-19 18:00 to 2026-01-20 00:00",
            "Peak Kp": "8.7",
            "Peak ap": "300",
            "Risk": "G4 Severe geomagnetic storm",
            "Select range": "2026-01-19T18:00:00 to 2026-01-19T23:55:00",
        },
    ])


def generate_historical_risk_alerts(start_time: str | None, end_time: str | None) -> pd.DataFrame:
    selected_start = _parse_datetime(start_time)
    selected_end = _parse_datetime(end_time)
    if selected_start is None or selected_end is None:
        return pd.DataFrame()
    if selected_start > selected_end:
        selected_start, selected_end = selected_end, selected_start

    alerts: list[dict[str, Any]] = []
    for _, window in historical_risk_windows().iterrows():
        window_start, window_end = _parse_range(str(window["Select range"]))
        if window_start is None or window_end is None:
            continue
        if selected_start <= window_end and selected_end >= window_start:
            risk_text = str(window["Risk"])
            risk_level = _extract_risk_level(risk_text)
            reason = (
                f"Selected range overlaps historical {risk_text} window "
                f"({window['Select range']}); peak Kp {window['Peak Kp']}, "
                f"peak ap {window['Peak ap']}."
            )
            alerts.append({
                "timestamp": window["Time UTC"],
                "region": "Global",
                "alert_type": "Historical geomagnetic storm window",
                "risk_level": risk_level,
                "value": window["Peak Kp"],
                "threshold_info": reason,
                "reason": reason,
                "possible_aviation_impact": (
                    "Historical geomagnetic storm conditions may affect GNSS and HF systems."
                ),
                "interpretation": (
                    "This advisory is based on the selected time range matching a known "
                    "high-risk geomagnetic storm window, not on the point-based SERENE "
                    "API values alone."
                ),
                "disclaimer": DISCLAIMER,
            })

    if not alerts:
        return pd.DataFrame()
    return pd.DataFrame(alerts)


def build_data_preview(df: pd.DataFrame, alerts: pd.DataFrame) -> pd.DataFrame:
    preview = df.copy()
    for col in ("alert_type", "risk_level", "alert_reason", "possible_aviation_impact"):
        if col not in preview.columns:
            preview[col] = ""
    if preview.empty or alerts.empty:
        return make_streamlit_safe_dataframe(preview)

    for idx, row in preview.iterrows():
        match = _match_alert(row, alerts)
        if match is None:
            continue
        preview.at[idx, "alert_type"] = match.get("alert_type", "")
        preview.at[idx, "risk_level"] = match.get("risk_level", "")
        preview.at[idx, "alert_reason"] = match.get("reason", "")
        preview.at[idx, "possible_aviation_impact"] = match.get("possible_aviation_impact", "")
    return make_streamlit_safe_dataframe(preview)


def make_streamlit_safe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    safe = df.copy()
    for column in safe.columns:
        series = safe[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            safe[column] = series.apply(_format_streamlit_value)
        elif series.dtype == "object":
            safe[column] = series.apply(_format_streamlit_value)
    return safe


def _format_streamlit_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, set)):
        return str(value)
    if isinstance(value, pd.Timestamp):
        timestamp = value
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert("UTC")
            return timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")
    if value is None or pd.isna(value):
        return ""
    return value


def mappable_variable_options(df: pd.DataFrame, contains_any: tuple[str, ...] = ()) -> list[str]:
    required = {"variable", "lat", "lon"}
    if df.empty or not required.issubset(df.columns):
        return []

    work = df[list(required)].copy()
    work["lat"] = pd.to_numeric(work["lat"], errors="coerce")
    work["lon"] = pd.to_numeric(work["lon"], errors="coerce")
    work = work.dropna(subset=["lat", "lon"])
    if work.empty:
        return []
    variables = sorted(work["variable"].dropna().astype(str).unique())
    if contains_any:
        needles = tuple(part.lower() for part in contains_any)
        variables = [
            var for var in variables
            if any(part in var.lower() for part in needles)
        ]
    return variables


def parse_select_range_to_widgets(select_range: str) -> dict[str, Any] | None:
    start, end = _parse_range(select_range)
    if start is None or end is None:
        return None
    return {
        "start_date": start.date(),
        "start_time_clock": start.time().replace(microsecond=0),
        "end_date": end.date(),
        "end_time_clock": end.time().replace(microsecond=0),
    }


def _match_alert(row: pd.Series, alerts: pd.DataFrame) -> pd.Series | None:
    candidates = alerts.copy()
    merged_geomagnetic = False
    if "variable" in row and "reason" in candidates.columns:
        variable = str(row.get("variable", ""))
        if variable.lower() in {"kp", "ap"} and "alert_type" in candidates.columns:
            candidates = candidates[
                (candidates["alert_type"] == "Geomagnetic storm risk")
                & candidates["reason"].astype(str).str.contains(
                    f"{variable} =", regex=False,
                )
            ]
            merged_geomagnetic = not candidates.empty
        else:
            candidates = candidates[
                candidates["reason"].astype(str).str.startswith(f"{variable} =")
            ]
    if not merged_geomagnetic and "value" in row and "value" in candidates.columns:
        row_value = pd.to_numeric(pd.Series([row.get("value")]), errors="coerce").iloc[0]
        alert_values = pd.to_numeric(candidates["value"], errors="coerce")
        if not pd.isna(row_value):
            candidates = candidates[(alert_values - row_value).abs() < 1e-9]
    if "time" in row and "timestamp" in candidates.columns:
        row_time = _parse_datetime(row.get("time"))
        if row_time is not None:
            alert_times = pd.to_datetime(candidates["timestamp"], errors="coerce")
            candidates = candidates[alert_times == row_time]
    if candidates.empty:
        return None
    return candidates.iloc[0]


def _parse_range(value: str) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    start_text, sep, end_text = value.partition(" to ")
    if not sep:
        return None, None
    return _parse_datetime(start_text), _parse_datetime(end_text)


def _extract_risk_level(risk_text: str) -> str:
    parts = risk_text.split()
    if len(parts) >= 2 and parts[0].startswith("G"):
        return " ".join(parts[:2])
    return "Warning"


def _parse_datetime(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    try:
        parsed = pd.to_datetime(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    return parsed
