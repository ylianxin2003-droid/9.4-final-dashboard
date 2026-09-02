from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Mapping

_CATEGORIES = {"OK", "MODERATE", "SEVERE"}
_CATEGORY_TEXT = {
    "OK": "NO SWX EXP",
    "MODERATE": "MOD",
    "SEVERE": "SEV",
}


def _as_utc_datetime(value: object) -> datetime:
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    elif isinstance(value, datetime):
        parsed = value
    elif hasattr(value, "to_pydatetime"):
        parsed = value.to_pydatetime()
    else:
        raise TypeError(f"Unsupported timestamp type: {type(value).__name__}")

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _format_icao_time(value: object) -> str:
    return _as_utc_datetime(value).strftime("%d/%H%MZ")


def _format_region(region: Mapping[str, float]) -> str:
    return (
        "USER-SELECTED BOUNDING BOX "
        f"LAT {float(region['lat_min']):.2f} TO {float(region['lat_max']):.2f}, "
        f"LON {float(region['lon_min']):.2f} TO {float(region['lon_max']):.2f}"
    )


def _validate_category(category: str) -> None:
    if category not in _CATEGORIES:
        raise ValueError(f"Unsupported category: {category}")


def generate_icao_message(
    *,
    effect: str,
    observed_time: object,
    observed_category: str,
    region: Mapping[str, float],
    forecasts: Mapping[int, str | None],
    generated_time: object,
    advisory_number: str,
) -> str:
    if effect not in {"GNSS", "HF COM"}:
        raise ValueError(f"Unsupported SWX effect: {effect}")

    _validate_category(observed_category)
    unsupported_horizons = set(forecasts) - {30, 90}
    if unsupported_horizons:
        raise ValueError(
            "Unsupported forecast horizon: "
            + ", ".join(str(value) for value in sorted(unsupported_horizons))
        )
    for category in forecasts.values():
        if category is not None:
            _validate_category(category)

    observed_timestamp = _as_utc_datetime(observed_time)
    generated_timestamp = _as_utc_datetime(generated_time)
    region_text = _format_region(region)
    lines = [
        "SWX ADVISORY",
        "STATUS: TEST",
        f"DTG: {generated_timestamp.strftime('%Y%m%d/%H%MZ')}",
        "SWXC: UOB RESEARCH PROTOTYPE",
        f"SWX EFFECT: {effect}",
        f"ADVISORY NR: {advisory_number}",
        (
            f"OBS SWX: {_format_icao_time(observed_timestamp)} "
            f"{_CATEGORY_TEXT[observed_category]} {region_text}"
        ),
    ]

    for period_minutes in (30, 90):
        category = forecasts.get(period_minutes)
        if category is None:
            continue
        label = "+30 MIN" if period_minutes == 30 else "+90 MIN"

        forecast_time = observed_timestamp + timedelta(minutes=period_minutes)
        lines.append(
            f"FCST SWX {label}: {_format_icao_time(forecast_time)} "
            f"{_CATEGORY_TEXT[category]} {region_text}"
        )

    lines.extend(
        [
            "RMK: GENERATED ONLY FROM SERENE AIDA/KP DATA.",
            "NXT ADVISORY: NO FURTHER ADVISORIES",
            "RESEARCH PROTOTYPE - NOT FOR OPERATIONAL USE",
        ]
    )
    return "\n".join(lines)
