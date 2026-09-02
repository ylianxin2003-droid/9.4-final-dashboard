from __future__ import annotations

import math

import pandas as pd


ICAO_COLORS = {
    "OK": "#2E7D32",
    "MODERATE": "#F9A825",
    "SEVERE": "#C62828",
    "UNAVAILABLE": "#95A5A6",
    "N/A": "#9E9E9E",
}

CELL_COLUMNS = [
    "indicator",
    "horizon",
    "display_value",
    "unit",
    "category",
    "color",
    "time",
    "lat",
    "lon",
    "source",
    "threshold_explanation",
    "product_state",
]

SUMMARY_COLUMNS = [
    "Domain",
    "Indicator",
    "Moderate threshold",
    "Severe threshold",
    "Time UTC",
    "Latest value",
    "Latest status",
    "Status",
    "Alert",
    "Max-3h value",
    "Max-3h status",
    "+30 min forecast",
    "+30 min status",
    "+30 min source",
    "+90 min forecast",
    "+90 min status",
    "+90 min source",
    "+3h forecast",
    "+3h status",
    "+3h source",
    "+6h forecast",
    "+6h status",
    "+6h source",
    "Source / Availability",
]

_SUPPORTED_INDICATORS = {"Vertical TEC", "Post-Storm Depression"}
FORECAST_HORIZONS = {
    "+30 min": 30,
    "+90 min": 90,
    "+3h": 180,
    "+6h": 360,
}
_SUPPORTED_MAP_HORIZONS = {"Latest", *FORECAST_HORIZONS.keys()}


def classify_tec(value):
    number = _finite_float(value)
    if number is None:
        return "UNAVAILABLE"
    if number < 125:
        return "OK"
    if number < 175:
        return "MODERATE"
    return "SEVERE"


def classify_auroral_absorption(kp):
    number = _finite_float(kp)
    if number is None:
        return "UNAVAILABLE"
    if number < 8:
        return "OK"
    if number < 9:
        return "MODERATE"
    return "SEVERE"


def calculate_psd_percent(current, reference):
    current_number = _finite_float(current)
    reference_number = _finite_float(reference)
    if current_number is None or reference_number is None or reference_number <= 0:
        return None
    return max(0.0, (reference_number - current_number) / reference_number * 100.0)


def classify_psd(value, kp_storm_eligible=False):
    number = _finite_float(value)
    if number is None:
        return "UNAVAILABLE"
    if kp_storm_eligible is None:
        return "UNAVAILABLE"
    if not kp_storm_eligible:
        return "OK"
    if number >= 50:
        return "SEVERE"
    if number >= 30:
        return "MODERATE"
    return "OK"


def worst_category(values):
    priority = {"UNAVAILABLE": -1, "OK": 0, "MODERATE": 1, "SEVERE": 2}
    valid = [value for value in values if value in priority]
    return max(valid, key=priority.get) if valid else "UNAVAILABLE"


def build_categorical_cells(
    products, indicator, horizon, kp_storm_eligible=False
):
    empty = pd.DataFrame(columns=CELL_COLUMNS + ["status"])
    canonical_indicator = _canonical_indicator(indicator)
    canonical_horizon = _canonical_horizon(horizon)
    if (
        canonical_indicator not in _SUPPORTED_INDICATORS
        or canonical_horizon not in _SUPPORTED_MAP_HORIZONS
    ):
        return empty

    frame = _as_frame(products)
    if frame.empty:
        return empty
    frame = _normalise_product_columns(frame)
    if not {"indicator", "horizon", "lat", "lon"}.issubset(frame.columns):
        return empty

    work = _rows_for_indicator_horizon(frame, canonical_indicator, canonical_horizon)
    if work.empty:
        return empty
    if canonical_horizon == "Latest" and "time" in work.columns:
        parsed_time = pd.to_datetime(work["time"], errors="coerce", utc=True)
        if parsed_time.notna().any():
            work = work[parsed_time == parsed_time.max()].copy()

    work["lat"] = pd.to_numeric(work["lat"], errors="coerce")
    work["lon"] = pd.to_numeric(work["lon"], errors="coerce")
    work = work.dropna(subset=["lat", "lon"])
    if work.empty:
        return empty

    rows = []
    for _, item in work.iterrows():
        if canonical_indicator == "Vertical TEC":
            display_value = _finite_float(item.get("value"))
            category = classify_tec(display_value)
            unit = "TECU"
        else:
            display_value = _psd_value(item)
            category = classify_psd(display_value, kp_storm_eligible)
            unit = "%"
        rows.append({
            "indicator": canonical_indicator,
            "horizon": canonical_horizon,
            "display_value": _na(display_value),
            "unit": unit,
            "category": category,
            "status": category,
            "color": ICAO_COLORS[category],
            "time": item.get("time", pd.NaT),
            "lat": float(item["lat"]),
            "lon": float(item["lon"]),
            "source": _source_value(item.get("source")),
            "threshold_explanation": _threshold_explanation(
                canonical_indicator, kp_storm_eligible
            ),
            "product_state": _product_state(item, canonical_horizon),
        })
    return pd.DataFrame(rows, columns=CELL_COLUMNS + ["status"])


def build_icao_summary(products, indices, eligible=False, kp_horizons=None):
    product_frame = _normalise_product_columns(_as_frame(products))
    rows = [
        _spatial_summary_row(product_frame, "GNSS", "Vertical TEC", eligible),
        _kp_summary_row(_as_frame(indices), _as_frame(kp_horizons)),
        _spatial_summary_row(product_frame, "HF COM", "Post-Storm Depression", eligible),
    ]
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def build_overall_risk_cards(summary):
    frame = _as_frame(summary)
    if frame.empty or not {"Domain", "Status"}.issubset(frame.columns):
        return {
            "GNSS Risk": "UNAVAILABLE",
            "HF COM Risk": "UNAVAILABLE",
            "Overall Risk": "UNAVAILABLE",
            "Data Completeness": "UNAVAILABLE",
        }
    completeness = build_evidence_completeness(frame)
    cards = {}
    for domain, label in (
        ("GNSS", "GNSS Risk"),
        ("HF COM", "HF COM Risk"),
    ):
        statuses = frame.loc[frame["Domain"] == domain, "Status"].tolist()
        cards[label] = _worst_available_or_unavailable(statuses)
    available = [
        status for status in cards.values()
        if status in {"OK", "MODERATE", "SEVERE"}
    ]
    if not available:
        overall = "UNAVAILABLE"
    else:
        worst = worst_category(available)
        if completeness["status"] == "PARTIAL":
            overall = (
                "PARTIAL DATA" if worst == "OK"
                else f"{worst} + PARTIAL DATA"
            )
        else:
            overall = worst
    cards["Overall Risk"] = overall
    cards["Data Completeness"] = completeness["status"]
    return cards


def build_evidence_completeness(summary):
    frame = _as_frame(summary)
    if frame.empty or "Status" not in frame.columns:
        return {
            "available": 0,
            "required": 0,
            "percent": 0,
            "status": "UNAVAILABLE",
            "missing": [],
        }

    required_rows = frame.copy()
    if "Indicator" in required_rows.columns:
        supported = {
            "Vertical TEC",
            "Post-Storm Depression",
            "Auroral Absorption",
        }
        selected = required_rows[required_rows["Indicator"].isin(supported)]
        if not selected.empty:
            required_rows = selected

    usable = {"OK", "MODERATE", "SEVERE"}
    available_mask = required_rows["Status"].isin(usable)
    available = int(available_mask.sum())
    required = int(len(required_rows))
    percent = round(available / required * 100) if required else 0
    if not required or not available:
        status = "UNAVAILABLE"
    elif available == required:
        status = "COMPLETE"
    else:
        status = "PARTIAL"
    label_column = "Indicator" if "Indicator" in required_rows.columns else "Domain"
    missing = (
        required_rows.loc[~available_mask, label_column]
        .dropna()
        .astype(str)
        .tolist()
    )
    return {
        "available": available,
        "required": required,
        "percent": percent,
        "status": status,
        "missing": missing,
    }


def _spatial_summary_row(frame, domain, indicator, eligible):
    values = {}
    sources = []
    for horizon in ("Latest", "Max3h", *FORECAST_HORIZONS.keys()):
        selected = _regional_max(frame, indicator, horizon)
        values[horizon] = selected
        if selected is not None:
            sources.append(_source_value(selected.get("source")))

    latest = values["Latest"]
    latest_value = _indicator_value(latest, indicator)
    max3 = _indicator_value(values["Max3h"], indicator)
    forecast_values = {
        horizon: _indicator_value(values[horizon], indicator)
        for horizon in FORECAST_HORIZONS
    }
    classifier = classify_tec if indicator == "Vertical TEC" else (
        lambda value: classify_psd(value, eligible)
    )
    latest_status = classifier(latest_value) if latest_value is not None else "UNAVAILABLE"
    max3_status = classifier(max3) if max3 is not None else "UNAVAILABLE"
    forecast_statuses = {
        horizon: (
            classifier(value) if value is not None else "UNAVAILABLE"
        )
        for horizon, value in forecast_values.items()
    }
    summary = {
        "Domain": domain,
        "Indicator": indicator,
        "Moderate threshold": _moderate_threshold(indicator),
        "Severe threshold": _severe_threshold(indicator),
        "Time UTC": _format_utc(latest.get("time")) if latest is not None else "N/A",
        "Latest value": _na(latest_value),
        "Latest status": latest_status,
        "Status": latest_status,
        "Alert": _alert_icon(latest_status),
        "Max-3h value": _na(max3),
        "Max-3h status": max3_status,
        "Source / Availability": (
            ", ".join(dict.fromkeys(sources))
            if sources else _availability_note(indicator, eligible)
        ),
    }
    for horizon, value in forecast_values.items():
        summary[f"{horizon} forecast"] = _na(value)
        summary[f"{horizon} status"] = forecast_statuses[horizon]
        summary[f"{horizon} source"] = _row_forecast_source(values[horizon])
    return summary


def _kp_summary_row(frame, kp_horizons=None):
    row = None
    max3_value = None
    if not frame.empty:
        work = frame.copy()
        variable_column = "variable" if "variable" in work.columns else "indicator"
        if variable_column in work.columns and "value" in work.columns:
            work = work[work[variable_column].astype(str).str.casefold() == "kp"].copy()
            work["value"] = pd.to_numeric(work["value"], errors="coerce")
            work = work.dropna(subset=["value"])
            if not work.empty:
                if "time" in work.columns:
                    work["_parsed_time"] = pd.to_datetime(
                        work["time"], errors="coerce", utc=True
                    )
                    row = work.sort_values("_parsed_time", na_position="first").iloc[-1]
                    latest_time = row.get("_parsed_time")
                    if pd.notna(latest_time):
                        window_start = latest_time - pd.Timedelta(hours=3)
                        window = work[
                            work["_parsed_time"].between(
                                window_start, latest_time, inclusive="both"
                            )
                        ]
                        if not window.empty:
                            max3_value = _finite_float(window["value"].max())
                else:
                    row = work.iloc[-1]
    value = _finite_float(row.get("value")) if row is not None else None
    status = classify_auroral_absorption(value) if value is not None else "UNAVAILABLE"
    max3_status = (
        classify_auroral_absorption(max3_value)
        if max3_value is not None else "UNAVAILABLE"
    )
    horizon_frame = _as_frame(kp_horizons)
    horizon_values = {
        horizon: _kp_horizon_summary(horizon_frame, minutes)
        for horizon, minutes in FORECAST_HORIZONS.items()
    }
    source_notes = [
        (
            _source_value(row.get("source")) + "; global Kp proxy, not regional"
            if row is not None else
            "GFZ Kp/ap unavailable; global proxy, not regional"
        )
    ]
    source_notes.extend(
        item["note"] for item in horizon_values.values() if item["note"]
    )
    summary = {
        "Domain": "HF COM",
        "Indicator": "Auroral Absorption",
        "Moderate threshold": "Kp >= 8 global proxy",
        "Severe threshold": "Kp >= 9 global proxy",
        "Time UTC": _format_utc(row.get("time")) if row is not None else "N/A",
        "Latest value": _na(value),
        "Latest status": status,
        "Status": status,
        "Alert": _alert_icon(status),
        "Max-3h value": _na(max3_value),
        "Max-3h status": max3_status,
        "Source / Availability": "; ".join(source_notes),
    }
    for horizon, evidence in horizon_values.items():
        summary[f"{horizon} forecast"] = _na(evidence["value"])
        summary[f"{horizon} status"] = evidence["status"]
        summary[f"{horizon} source"] = evidence["source"]
    return summary


def _kp_horizon_summary(frame, horizon_minutes):
    result = {
        "value": None,
        "status": "UNAVAILABLE",
        "source": "Unavailable",
        "note": "",
    }
    if frame.empty or "horizon_minutes" not in frame.columns:
        return result
    numeric_horizon = pd.to_numeric(frame["horizon_minutes"], errors="coerce")
    selected = frame[numeric_horizon == horizon_minutes]
    if selected.empty:
        return result
    evidence = selected.iloc[-1]
    value = _finite_float(evidence.get("value"))
    role = str(evidence.get("evidence_role", "unavailable"))
    if value is None or role not in {"official_forecast", "observed_backtesting"}:
        reason = str(evidence.get("availability_reason", "")).strip()
        result["note"] = f"+{horizon_minutes} min: {reason}" if reason else ""
        return result

    result.update({
        "value": value,
        "status": classify_auroral_absorption(value),
        "source": str(evidence.get("source") or "Unavailable"),
    })
    if role == "observed_backtesting":
        data_status = str(evidence.get("data_status", "observed") or "observed")
        result["note"] = (
            f"+{horizon_minutes} min: observed outcome is backtesting only, "
            f"not a forecast ({data_status})"
        )
        return result

    maximum = _finite_float(evidence.get("ensemble_maximum"))
    probability = _finite_float(evidence.get("probability_kp_ge_8"))
    details = [f"+{horizon_minutes} min: ensemble median Kp {value:g}"]
    if maximum is not None:
        details.append(f"ensemble maximum Kp {maximum:g}")
    if probability is not None:
        details.append(f"P(Kp >= 8) {probability:.0%}")
    if maximum is not None and value < 8 <= maximum:
        details.append("low-probability high-impact tail; primary status unchanged")
    result["note"] = ", ".join(details)
    return result


def _regional_max(frame, indicator, horizon):
    if frame.empty or not {"indicator", "horizon"}.issubset(frame.columns):
        return None
    work = _rows_for_indicator_horizon(frame, indicator, horizon)
    if work.empty:
        return None
    if horizon == "Latest" and "time" in work.columns:
        parsed_time = (
            work["_canonical_time"]
            if "_canonical_time" in work.columns
            else pd.to_datetime(work["time"], errors="coerce", utc=True)
        )
        if parsed_time.notna().any():
            work = work[parsed_time == parsed_time.max()].copy()
    risk_column = (
        "_psd_risk_value"
        if indicator == "Post-Storm Depression"
        else "_tec_risk_value"
    )
    if risk_column in work.columns:
        work["_risk_value"] = work[risk_column]
    else:
        work["_risk_value"] = work.apply(
            lambda row: _indicator_value(row, indicator), axis=1
        )
        work["_risk_value"] = pd.to_numeric(work["_risk_value"], errors="coerce")
    work = work.dropna(subset=["_risk_value"])
    if work.empty:
        return None
    return work.loc[work["_risk_value"].idxmax()]


def _rows_for_indicator_horizon(frame, indicator, horizon):
    if frame.empty or not {"indicator", "horizon"}.issubset(frame.columns):
        return pd.DataFrame()
    canonical_horizon = _canonical_horizon(horizon)
    indicator_values = (
        frame["_canonical_indicator"]
        if "_canonical_indicator" in frame.columns
        else frame["indicator"].map(_canonical_indicator)
    )
    horizon_values = (
        frame["_canonical_horizon"]
        if "_canonical_horizon" in frame.columns
        else frame["horizon"].map(_canonical_horizon)
    )
    work = frame[
        (indicator_values == indicator)
        & (horizon_values == canonical_horizon)
    ].copy()
    if not work.empty:
        if canonical_horizon in FORECAST_HORIZONS:
            work["forecast_source"] = "SERENE official forecast"
        return work
    if canonical_horizon in FORECAST_HORIZONS:
        return _fallback_prediction_rows(frame, indicator, canonical_horizon)
    return work


def _fallback_prediction_rows(frame, indicator, horizon):
    if frame.empty or not {"indicator", "horizon", "lat", "lon"}.issubset(frame.columns):
        return pd.DataFrame()
    hours = FORECAST_HORIZONS[horizon] / 60.0
    indicator_values = (
        frame["_canonical_indicator"]
        if "_canonical_indicator" in frame.columns
        else frame["indicator"].map(_canonical_indicator)
    )
    horizon_values = (
        frame["_canonical_horizon"]
        if "_canonical_horizon" in frame.columns
        else frame["horizon"].map(_canonical_horizon)
    )
    work = frame[
        (indicator_values == indicator)
        & (horizon_values.isin(["Latest", "Max3h"]))
    ].copy()
    if "product_kind" in work.columns:
        work = work[work["product_kind"].astype(str).str.casefold() != "baseline"]
    if work.empty:
        return pd.DataFrame()
    work["lat"] = pd.to_numeric(work["lat"], errors="coerce")
    work["lon"] = pd.to_numeric(work["lon"], errors="coerce")
    risk_column = (
        "_psd_risk_value"
        if indicator == "Post-Storm Depression"
        else "_tec_risk_value"
    )
    if risk_column in work.columns:
        work["_risk_value"] = work[risk_column]
    else:
        work["_risk_value"] = work.apply(
            lambda row: _indicator_value(row, indicator), axis=1
        )
        work["_risk_value"] = pd.to_numeric(work["_risk_value"], errors="coerce")
    if "_canonical_time" in work.columns:
        work["_parsed_time"] = work["_canonical_time"]
    elif "time" in work.columns:
        work["_parsed_time"] = pd.to_datetime(work["time"], errors="coerce", utc=True)
    else:
        work["_parsed_time"] = pd.NaT
    work = work.dropna(subset=["lat", "lon", "_risk_value"])
    if work.empty:
        return pd.DataFrame()

    rows = []
    for _, group in work.groupby(["lat", "lon"], sort=False):
        if group["_parsed_time"].notna().any():
            latest_time = group["_parsed_time"].max()
            latest_candidates = group[group["_parsed_time"] == latest_time]
            latest = latest_candidates.iloc[-1].copy()
            window_start = latest_time - pd.Timedelta(hours=3)
            window = group[
                group["_parsed_time"].between(window_start, latest_time, inclusive="both")
            ].copy()
            earlier = window[window["_parsed_time"] < latest_time]
            if not earlier.empty:
                earliest = earlier.sort_values("_parsed_time").iloc[0]
                trend_per_hour = (
                    float(latest["_risk_value"]) - float(earliest["_risk_value"])
                ) / 3.0
                predicted = float(latest["_risk_value"]) + trend_per_hour * hours
                forecast_source = "Dashboard-generated trend-based forecast"
            else:
                predicted = float(latest["_risk_value"])
                forecast_source = "Dashboard-generated persistence forecast"
            latest["time"] = latest_time + pd.Timedelta(hours=hours)
            latest["_canonical_time"] = latest["time"]
        else:
            latest = group.iloc[-1].copy()
            predicted = float(latest["_risk_value"])
            forecast_source = "Dashboard-generated persistence forecast"

        latest["horizon"] = horizon
        latest["forecast_source"] = forecast_source
        latest["product_kind"] = (
            f"fallback_trend_{int(hours * 60)}"
            if "trend-based" in forecast_source
            else f"fallback_persistence_{int(hours * 60)}"
        )
        latest["source"] = (
            f"{forecast_source} from SERENE analysis"
        )
        if indicator == "Post-Storm Depression":
            latest["psd_percent"] = predicted
            latest["_psd_risk_value"] = predicted
        else:
            latest["value"] = predicted
            latest["_tec_risk_value"] = predicted
        rows.append(latest.drop(labels=["_risk_value", "_parsed_time"], errors="ignore"))
    return pd.DataFrame(rows)


def _indicator_value(row, indicator):
    if row is None:
        return None
    if indicator == "Post-Storm Depression":
        return _psd_value(row)
    return _finite_float(row.get("value"))


def _psd_value(row):
    if "psd_percent" in row:
        return _finite_float(row.get("psd_percent"))
    for column in ("display_value", "value"):
        if column in row and pd.notna(row.get(column)):
            return _finite_float(row.get(column))
    if "current" in row and "reference" in row:
        return calculate_psd_percent(row.get("current"), row.get("reference"))
    return None


def _normalise_product_columns(frame):
    if frame.empty:
        return frame.copy()
    work = frame.copy()
    if "indicator" not in work.columns and "variable" in work.columns:
        variable_names = work["variable"].astype(str)
        work["indicator"] = variable_names.map({
            "TEC": "Vertical TEC",
            "vTEC": "Vertical TEC",
            "MUF3000F2": "Post-storm depression",
            "MUF3000": "Post-storm depression",
        }).fillna(variable_names)
    if "horizon" not in work.columns:
        if "product_kind" in work.columns:
            product_kinds = work["product_kind"].astype(str)
            work["horizon"] = product_kinds.map({
                "analysis": "Latest",
                "rolling": "Max3h",
                "forecast_30": "+30 min",
                "forecast_90": "+90 min",
                "forecast_180": "+3h",
                "forecast_360": "+6h",
            }).fillna(product_kinds)
        else:
            work["horizon"] = "Latest"
    work["_canonical_indicator"] = work["indicator"].map(_canonical_indicator)
    work["_canonical_horizon"] = work["horizon"].map(_canonical_horizon)
    if "time" in work.columns:
        work["_canonical_time"] = pd.to_datetime(
            work["time"], errors="coerce", utc=True
        )
    if "value" in work.columns:
        work["_tec_risk_value"] = pd.to_numeric(work["value"], errors="coerce")
    else:
        work["_tec_risk_value"] = float("nan")
    if "psd_percent" in work.columns:
        psd_values = work["psd_percent"]
    elif "display_value" in work.columns:
        psd_values = work["display_value"]
        if "value" in work.columns:
            psd_values = psd_values.where(psd_values.notna(), work["value"])
    elif "value" in work.columns:
        psd_values = work["value"]
    elif {"current", "reference"}.issubset(work.columns):
        current = pd.to_numeric(work["current"], errors="coerce")
        reference = pd.to_numeric(work["reference"], errors="coerce")
        psd_values = ((reference - current) / reference * 100.0).clip(lower=0.0)
        psd_values = psd_values.where(reference > 0)
    else:
        psd_values = float("nan")
    work["_psd_risk_value"] = pd.to_numeric(psd_values, errors="coerce")
    return work


def _canonical_indicator(value):
    text = str(value).strip().casefold()
    if text in {"vertical tec", "tec", "vtec"}:
        return "Vertical TEC"
    if text in {"post-storm depression", "post storm depression", "psd"}:
        return "Post-Storm Depression"
    return str(value).strip()


def _canonical_horizon(value):
    text = "".join(str(value).casefold().split())
    aliases = {
        "latest": "Latest",
        "now": "Latest",
        "max3h": "Max3h",
        "+30min": "+30 min",
        "+30m": "+30 min",
        "30min": "+30 min",
        "+0.5h": "+30 min",
        "+90min": "+90 min",
        "+90m": "+90 min",
        "90min": "+90 min",
        "+1.5h": "+90 min",
        "forecast_180": "+3h",
        "180": "+3h",
        "+3h": "+3h",
        "forecast_360": "+6h",
        "360": "+6h",
        "+6h": "+6h",
    }
    return aliases.get(text, str(value).strip())


def _as_frame(value):
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if value is None:
        return pd.DataFrame()
    try:
        return pd.DataFrame(value)
    except (TypeError, ValueError):
        return pd.DataFrame()


def _finite_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _format_utc(value):
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return "N/A"
    return parsed.strftime("%Y-%m-%d %H:%M UTC")


def _source_value(value):
    if value is None or pd.isna(value) or not str(value).strip():
        return "SERENE"
    return str(value)


def _threshold_explanation(indicator, kp_storm_eligible):
    if indicator == "Vertical TEC":
        return "OK <125 TECU; MODERATE 125 to <175 TECU; SEVERE >=175 TECU"
    if kp_storm_eligible is None:
        gate = "eligibility unavailable"
    else:
        gate = "eligible" if kp_storm_eligible else "not eligible"
    return (
        "Requires Kp >=6 in prior 96h "
        f"({gate}); OK <30%; MODERATE 30 to <50%; SEVERE >=50%"
    )


def _product_state(item, horizon):
    forecast_source = item.get("forecast_source")
    if forecast_source is not None and not pd.isna(forecast_source):
        return str(forecast_source).casefold()
    product_kind = str(item.get("product_kind", "")).strip().casefold()
    if product_kind.startswith("forecast_") or horizon in FORECAST_HORIZONS:
        return "official forecast"
    return "analysis"


def _na(value):
    return "N/A" if value is None else value


def _moderate_threshold(indicator):
    if indicator == "Vertical TEC":
        return "TEC >= 125 TECU"
    if indicator == "Post-Storm Depression":
        return "PSD >= 30%"
    return "N/A"


def _severe_threshold(indicator):
    if indicator == "Vertical TEC":
        return "TEC >= 175 TECU"
    if indicator == "Post-Storm Depression":
        return "PSD >= 50%"
    return "N/A"


def _availability_note(indicator, eligible):
    if indicator == "Post-Storm Depression":
        if eligible is None:
            return "Requires AIDA MUF3000F2 baseline and complete prior-96h Kp history"
        if not eligible:
            return "AIDA MUF3000F2-derived PSD; Kp storm gate inactive"
        return "AIDA MUF3000F2-derived PSD; Kp storm gate active"
    if indicator == "Vertical TEC":
        return "SERENE AIDA TEC unavailable for selected product"
    return "N/A"


def _alert_icon(status):
    return {
        "OK": "✓",
        "MODERATE": "⚠",
        "SEVERE": "!",
        "UNAVAILABLE": "—",
        "N/A": "—",
    }.get(str(status), "—")


def _row_forecast_source(row):
    if row is None:
        return "Unavailable"
    source = row.get("forecast_source")
    if source is None or pd.isna(source) or not str(source).strip():
        return "Unavailable"
    return str(source)


def _worst_available_or_unavailable(values):
    priority = {"OK": 0, "MODERATE": 1, "SEVERE": 2}
    available = [str(value) for value in values if str(value) in priority]
    if not available:
        return "UNAVAILABLE"
    return max(available, key=priority.get)
