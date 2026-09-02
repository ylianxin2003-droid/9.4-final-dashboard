from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from aida_adapter import (
    UPSTREAM_AIDA_VERSION,
    calculate_aida_grid,
)
from aida_grid import AidaGridError, estimate_target_points
from app_utils import AIDA_ARCHIVE_START_UTC
from serene_client import SereneClient, normalise_aida_request_time

logger = logging.getLogger(__name__)


@dataclass
class LoadStatus:

    source: str = "unknown"
    ok: bool = False
    message: str = ""
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IcaoProductBundle:

    products: pd.DataFrame = field(default_factory=pd.DataFrame)
    indices: pd.DataFrame = field(default_factory=pd.DataFrame)
    status: LoadStatus = field(default_factory=LoadStatus)
    kp_storm_eligible: bool | None = None
    kp_horizons: pd.DataFrame = field(default_factory=pd.DataFrame)


PSD_REFERENCE_EXPECTED_STATES = 30
PSD_REFERENCE_MIN_STATES = 27
PRIMARY_FORECAST_PERIODS = (30, 90, 180, 360)
FORECAST_PERIODS = PRIMARY_FORECAST_PERIODS
KP_PUBLICATION_DELAY_TOLERANCE = pd.Timedelta(minutes=15)
KP_FORECAST_MAX_AGE = pd.Timedelta(hours=3)
KP_HORIZON_MINUTES = (30, 90, 180, 360)
KP_HORIZON_COLUMNS = (
    "horizon_minutes",
    "target_time",
    "interval_start",
    "value",
    "evidence_role",
    "source",
    "ensemble_maximum",
    "probability_kp_ge_8",
    "data_status",
    "issue_time",
    "availability_reason",
)


def three_hour_aida_times(analysis_time: str) -> list[pd.Timestamp]:
    end = normalise_aida_request_time(analysis_time)
    return list(pd.date_range(end=end, periods=37, freq="5min", tz="UTC"))


def psd_reference_times(analysis_time: str) -> list[pd.Timestamp]:
    end = normalise_aida_request_time(analysis_time)
    return [end - pd.Timedelta(days=days) for days in range(30, 0, -1)]


def load_icao_products(
    analysis_time: str,
    variables: list[str],
    region: dict[str, float],
    grid_step: float,
    include_three_hour_window: bool = True,
    include_psd_baseline: bool = True,
    progress_callback: Any | None = None,
) -> IcaoProductBundle:
    status = LoadStatus(source="none", ok=False)
    try:
        requested_analysis = normalise_aida_request_time(analysis_time)
        local_map_points = estimate_target_points(region, grid_step)
    except (AidaGridError, KeyError, TypeError, ValueError) as exc:
        status.message = f"Invalid ICAO product request: {exc}"
        return IcaoProductBundle(status=status)

    publication_safe_now = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=15)
    if requested_analysis < AIDA_ARCHIVE_START_UTC:
        status.message = "AIDA analysis time must not be before 2024-09-28 00:00 UTC."
        return IcaoProductBundle(status=status)
    if requested_analysis > publication_safe_now:
        status.message = "AIDA analysis time must not be in the unpublished future window."
        return IcaoProductBundle(status=status)

    client = SereneClient()
    analysis = requested_analysis
    analysis_anchor_source = "user_selected_time"

    selected_variables = list(dict.fromkeys(variables or ["TEC"]))
    requested_rolling_times = (
        three_hour_aida_times(analysis.isoformat())
        if include_three_hour_window
        else [analysis]
    )
    rolling_times = [
        requested for requested in requested_rolling_times
        if requested >= AIDA_ARCHIVE_START_UTC
    ]
    rolling_truncated = len(rolling_times) < len(requested_rolling_times)
    requested_baseline_times = (
        psd_reference_times(analysis.isoformat())
        if include_psd_baseline and "MUF3000F2" in selected_variables
        else []
    )
    baseline_times = [
        requested for requested in requested_baseline_times
        if requested >= AIDA_ARCHIVE_START_UTC
    ]
    baseline_truncated = len(baseline_times) < len(requested_baseline_times)
    if baseline_truncated:
        baseline_times = []
    total_requests = len(rolling_times) + len(baseline_times) + len(FORECAST_PERIODS)
    completed = 0
    analysis_downloads = 0
    rolling_analysis_downloads = 0
    baseline_downloads = 0
    forecast_downloads = 0
    primary_forecast_states = 0
    forecast_request_audit: list[dict[str, Any]] = []
    warnings: list[str] = []
    if rolling_truncated:
        warnings.append(
            "The three-hour window crosses the 2024-09-28 AIDA archive boundary; "
            "pre-archive states were not requested."
        )
    if baseline_truncated:
        warnings.append(
            "PSD unavailable: a complete 30-day AIDA reference is unavailable "
            "before the 2024-09-28 archive boundary."
        )
    product_frames: list[pd.DataFrame] = []
    baseline_value_series: list[pd.Series] = []
    baseline_download_failures = 0

    def report_progress(label: str) -> None:
        if progress_callback:
            progress_callback(completed, total_requests, label)

    for requested in rolling_times:
        latency = _aida_latency(requested)
        ok, message, payload = client.download_aida_raw_output(
            requested.isoformat(), latency
        )
        completed += 1
        report_progress("3-hour AIDA observations")
        if not ok or payload is None:
            warnings.append(message)
            continue
        analysis_downloads += 1
        rolling_analysis_downloads += 1
        try:
            frame = _calculate_product_frame(
                payload, region, grid_step, selected_variables,
                product_kind="rolling", requested_time=requested,
            )
        except AidaGridError as exc:
            warnings.append(str(exc))
            continue
        if frame.empty:
            continue
        product_frames.append(frame)
        if requested == analysis:
            latest = frame.copy()
            latest["product_kind"] = "analysis"
            product_frames.append(latest)

    baseline_variables = ["MUF3000F2"]
    for requested in baseline_times:
        latency = _aida_latency(requested)
        ok, message, payload = client.download_aida_raw_output(
            requested.isoformat(), latency
        )
        completed += 1
        report_progress("30-day PSD reference")
        if not ok or payload is None:
            baseline_download_failures += 1
            logger.info("PSD baseline AIDA state unavailable: %s", message)
            continue
        analysis_downloads += 1
        baseline_downloads += 1
        try:
            frame = _calculate_product_frame(
                payload, region, grid_step, baseline_variables,
                product_kind="baseline", requested_time=requested,
            )
        except AidaGridError as exc:
            warnings.append(str(exc))
            continue
        if not frame.empty:
            values = _baseline_value_series(frame, requested)
            if not values.empty:
                baseline_value_series.append(values)

    for period in FORECAST_PERIODS:
        latency = _aida_latency(analysis)
        forecast_time = analysis + pd.Timedelta(minutes=period)
        ok, message, payload = client.download_aida_forecast(
            analysis.isoformat(), latency, period
        )
        completed += 1
        forecast_label = _forecast_label(period)
        report_progress(f"AIDA {forecast_label} forecast")
        forecast_request_audit.append({
            "analysis_time": analysis.isoformat(),
            "valid_time": forecast_time.isoformat(),
            "forecast_parameter": period,
            "latency": latency,
            "display_role": (
                "primary" if period in PRIMARY_FORECAST_PERIODS else "audit_only"
            ),
            "downloaded_from_serene": bool(ok and payload is not None),
            "outcome": _forecast_request_outcome(ok, message),
            "message": message,
        })
        if not ok or payload is None:
            warnings.append(_forecast_unavailable_message(period, message))
            continue
        forecast_downloads += 1
        try:
            frame = _calculate_product_frame(
                payload, region, grid_step, selected_variables,
                product_kind=f"forecast_{period}", requested_time=analysis,
                forecast_minutes=period,
            )
        except AidaGridError as exc:
            warnings.append(f"Official AIDA {forecast_label} forecast unavailable: {exc}")
            forecast_request_audit[-1]["downloaded_from_serene"] = False
            forecast_request_audit[-1]["outcome"] = "decode_failed"
            forecast_request_audit[-1]["message"] = str(exc)
            continue
        if not frame.empty:
            product_frames.append(frame)
            primary_forecast_states += 1

    index_start = (
        analysis - pd.Timedelta(hours=96)
    ).floor("3h").isoformat()
    ok_indices, indices_message, indices = client.fetch_kp_ap_indices(
        start_time=index_start,
        end_time=analysis.isoformat(),
    )
    kp_ap_source_latest_time = getattr(client, "kp_ap_source_latest_time", None)
    kp_ap_data_statuses = list(getattr(client, "kp_ap_data_statuses", []) or [])
    kp_ap_missing_indices = list(
        getattr(client, "kp_ap_missing_indices", []) or []
    )
    kp_ap_source_latest_iso = (
        pd.Timestamp(kp_ap_source_latest_time).isoformat()
        if kp_ap_source_latest_time is not None
        and not pd.isna(kp_ap_source_latest_time)
        else None
    )
    if not ok_indices:
        warnings.append(indices_message)
        indices = pd.DataFrame()
    kp_ap_status = "loaded" if ok_indices and not indices.empty else "unavailable"
    kp_horizons, kp_horizon_message = _load_kp_horizons(
        client,
        analysis,
        indices,
    )
    if (
        not kp_horizons.empty
        and (kp_horizons["evidence_role"] == "unavailable").any()
    ):
        warnings.append(kp_horizon_message)

    products = (
        pd.concat(product_frames, ignore_index=True)
        if product_frames else pd.DataFrame()
    )
    reference_state_count = len({
        series.name for series in baseline_value_series if not series.empty
    })
    if baseline_times:
        if reference_state_count >= PSD_REFERENCE_MIN_STATES:
            missing = PSD_REFERENCE_EXPECTED_STATES - reference_state_count
            if missing > 0:
                warnings.append(
                    f"PSD reference used {reference_state_count}/"
                    f"{PSD_REFERENCE_EXPECTED_STATES} available SERENE AIDA "
                    "states; missing reference files were skipped."
                )
        else:
            warnings.append(
                f"PSD unavailable: only {reference_state_count}/"
                f"{PSD_REFERENCE_EXPECTED_STATES} SERENE AIDA reference "
                "states were available."
            )
    reference = _build_psd_reference(baseline_value_series)
    products = _attach_psd_reference(products, reference)
    kp_values = pd.Series(dtype=float)
    if not indices.empty and "variable" in indices.columns:
        kp_values = pd.to_numeric(
            indices.loc[indices["variable"] == "Kp", "value"],
            errors="coerce",
        ).dropna()
    kp_history_complete = _kp_history_is_complete(indices, analysis)
    kp_storm_eligible = (
        bool(kp_values.max() >= 6) if kp_history_complete else None
    )
    if not kp_history_complete:
        warnings.append(
            "Complete 96-hour GFZ Kp history is unavailable; PSD status is unavailable."
        )

    has_analysis = (
        not products.empty
        and "product_kind" in products.columns
        and (products["product_kind"] == "analysis").any()
    )
    actual_analysis_output_time = None
    if has_analysis and "actual_output_time" in products.columns:
        actual_values = pd.to_datetime(
            products.loc[
                products["product_kind"] == "analysis",
                "actual_output_time",
            ],
            errors="coerce",
            utc=True,
        ).dropna()
        if not actual_values.empty:
            actual_analysis_output_time = pd.Timestamp(actual_values.max()).isoformat()
    status.source = "api" if has_analysis else "none"
    status.ok = bool(has_analysis)
    status.message = (
        "Loaded SERENE AIDA observations and available prediction inputs."
        if has_analysis else
        "SERENE returned no usable AIDA analysis state."
    )
    status.warnings = warnings
    status.metadata = {
        "requested_analysis_time": requested_analysis.isoformat(),
        "analysis_time": analysis.isoformat(),
        "analysis_anchor_source": analysis_anchor_source,
        "actual_analysis_output_time": actual_analysis_output_time,
        "analysis_downloads": analysis_downloads,
        "rolling_analysis_downloads": rolling_analysis_downloads,
        "baseline_downloads": baseline_downloads,
        "forecast_downloads": forecast_downloads,
        "primary_forecast_states": primary_forecast_states,
        "available_primary_forecast_periods": [
            period for period in PRIMARY_FORECAST_PERIODS
            if any(
                item["forecast_parameter"] == period
                and item["outcome"] == "available"
                for item in forecast_request_audit
            )
        ],
        "forecast_request_audit": forecast_request_audit,
        "local_map_points": local_map_points,
        "grid_step_degrees": float(grid_step),
        "loaded_region": dict(region),
        "archive_start": AIDA_ARCHIVE_START_UTC.isoformat(),
        "rolling_state_count": len(rolling_times),
        "baseline_state_count": len(baseline_times),
        "baseline_reference_states_used": reference_state_count,
        "baseline_download_failures": baseline_download_failures,
        "kp_ap_index_status": kp_ap_status,
        "kp_ap_index_message": indices_message,
        "kp_ap_source": "GFZ Helmholtz Centre for Geosciences",
        "kp_ap_source_latest_time": kp_ap_source_latest_iso,
        "kp_ap_data_statuses": kp_ap_data_statuses,
        "kp_ap_missing_indices": kp_ap_missing_indices,
        "kp_horizon_message": kp_horizon_message,
        "total_official_aida_downloads": analysis_downloads + forecast_downloads,
        "upstream_interpreter": (
            f"breid-phys/aida-ionosphere {UPSTREAM_AIDA_VERSION}"
        ),
    }
    return IcaoProductBundle(
        products=products,
        indices=indices,
        status=status,
        kp_storm_eligible=kp_storm_eligible,
        kp_horizons=kp_horizons,
    )


def _aida_latency(requested: pd.Timestamp) -> str:
    return "ultra" if requested.year == pd.Timestamp.now(tz="UTC").year else "rapid"


def _calculate_product_frame(
    payload: bytes,
    region: dict[str, float],
    grid_step: float,
    variables: list[str],
    product_kind: str,
    requested_time: pd.Timestamp,
    forecast_minutes: int = 0,
) -> pd.DataFrame:
    frame = calculate_aida_grid(payload, region, grid_step, variables)
    if frame.empty:
        return frame
    frame = frame.copy()
    if "actual_output_time" not in frame.columns and "time" in frame.columns:
        frame["actual_output_time"] = frame["time"]
    valid_time = requested_time + pd.Timedelta(minutes=int(forecast_minutes))
    frame["time"] = valid_time
    frame["valid_time"] = valid_time
    frame["product_kind"] = product_kind
    frame["requested_time"] = requested_time
    frame["forecast_minutes"] = int(forecast_minutes)
    return frame


def _forecast_unavailable_message(period_minutes: int, detail: str) -> str:
    label = _forecast_label(period_minutes)
    reason = "SERENE did not provide a downloadable forecast file for this analysis time."
    lower_detail = detail.lower()
    if "401" in detail or "403" in detail or "token" in lower_detail:
        reason = "SERENE rejected the API token for the forecast file."
    elif "not available for download" in lower_detail or "404" in detail:
        reason = "SERENE did not provide a downloadable forecast file for this analysis time."
    elif detail:
        reason = detail
    return f"Official AIDA {label} forecast unavailable: {reason}"


def _forecast_request_outcome(ok: bool, message: str) -> str:
    text = str(message).casefold()
    if ok:
        return "available"
    if "401" in text or "403" in text or "token" in text:
        return "authentication_failed"
    if "404" in text or "not available" in text or "not provide" in text:
        return "not_published"
    if "timeout" in text or "connection" in text or "network" in text:
        return "network_failed"
    return "decode_failed"


def _forecast_label(period_minutes: int) -> str:
    if period_minutes == 30:
        return "+30 min"
    if period_minutes == 90:
        return "+90 min"
    hours = period_minutes // 60
    return f"+{hours}h"


def _baseline_value_series(
    frame: pd.DataFrame,
    requested_time: pd.Timestamp,
) -> pd.Series:
    required = {"lat", "lon", "variable", "value"}
    if frame.empty or not required.issubset(frame.columns):
        return pd.Series(dtype=float, name=requested_time)
    work = frame.loc[
        frame["variable"] == "MUF3000F2", ["lat", "lon", "value"]
    ].copy()
    work["lat"] = pd.to_numeric(work["lat"], errors="coerce")
    work["lon"] = pd.to_numeric(work["lon"], errors="coerce")
    work["value"] = pd.to_numeric(work["value"], errors="coerce")
    work = work.dropna(subset=["lat", "lon"])
    if work.empty:
        return pd.Series(dtype=float, name=requested_time)
    values = work.groupby(["lat", "lon"], sort=False)["value"].median()
    values.name = requested_time
    return values


def _build_psd_reference(
    baseline_values: list[pd.Series],
    expected_states: int = PSD_REFERENCE_EXPECTED_STATES,
    min_states: int = PSD_REFERENCE_MIN_STATES,
) -> pd.DataFrame:
    non_empty = [series for series in baseline_values if not series.empty]
    required_states = min(expected_states, min_states)
    if len({series.name for series in non_empty}) < required_states:
        return pd.DataFrame(columns=["lat", "lon", "reference_value"])
    matrix = pd.concat(non_empty, axis=1)
    complete = matrix.notna().sum(axis=1) >= required_states
    if not complete.any():
        return pd.DataFrame(columns=["lat", "lon", "reference_value"])
    reference = matrix.loc[complete].median(axis=1).rename("reference_value")
    return reference.reset_index()


def _attach_psd_reference(
    products: pd.DataFrame,
    reference: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if products.empty or "product_kind" not in products.columns:
        return products
    non_baseline = products[products["product_kind"] != "baseline"].copy()
    if reference is None:
        baseline = products[
            (products["product_kind"] == "baseline")
            & (products["variable"] == "MUF3000F2")
        ]
        if "requested_time" not in baseline.columns:
            reference = pd.DataFrame()
        else:
            series = [
                _baseline_value_series(group, requested)
                for requested, group in baseline.groupby("requested_time")
            ]
            reference = _build_psd_reference(series)
    if reference.empty:
        non_baseline["reference_value"] = pd.NA
        non_baseline["psd_percent"] = pd.NA
        return non_baseline
    result = non_baseline.drop(
        columns=["reference_value", "psd_percent"], errors="ignore"
    )
    merged = result.merge(reference, on=["lat", "lon"], how="left")
    is_muf = merged["variable"] == "MUF3000F2"
    current = pd.to_numeric(merged["value"], errors="coerce")
    reference_value = pd.to_numeric(merged["reference_value"], errors="coerce")
    valid = is_muf & reference_value.gt(0) & current.notna()
    merged["psd_percent"] = pd.NA
    merged.loc[valid, "psd_percent"] = (
        ((reference_value[valid] - current[valid]) / reference_value[valid])
        .clip(lower=0)
        * 100.0
    )
    return merged


def _kp_history_is_complete(indices: pd.DataFrame, analysis: pd.Timestamp) -> bool:
    if indices.empty or not {"variable", "time", "value"}.issubset(indices.columns):
        return False
    kp = indices[indices["variable"] == "Kp"].copy()
    kp["time"] = pd.to_datetime(kp["time"], errors="coerce", utc=True)
    kp["value"] = pd.to_numeric(kp["value"], errors="coerce")
    kp = kp.dropna(subset=["time", "value"]).sort_values("time")
    if len(kp) < 32 or kp["time"].nunique() < 32:
        return False
    if (
        kp["time"].max() + pd.Timedelta(hours=3)
        + KP_PUBLICATION_DELAY_TOLERANCE < analysis
    ):
        return False
    if kp["time"].min() > analysis - pd.Timedelta(hours=93):
        return False
    gaps = kp["time"].drop_duplicates().sort_values().diff().dropna()
    return bool(gaps.empty or gaps.max() <= pd.Timedelta(hours=3, minutes=5))


def _load_kp_horizons(
    client: SereneClient,
    analysis: pd.Timestamp,
    prior_indices: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    now = pd.Timestamp.now(tz="UTC")
    target_specs = [
        (
            minutes,
            analysis + pd.Timedelta(minutes=minutes),
            (analysis + pd.Timedelta(minutes=minutes)).floor("3h"),
        )
        for minutes in KP_HORIZON_MINUTES
    ]
    observed_frames = [prior_indices] if not prior_indices.empty else []
    existing_intervals: set[pd.Timestamp] = set()
    if not prior_indices.empty and {"variable", "time"}.issubset(prior_indices):
        existing = prior_indices[prior_indices["variable"] == "Kp"].copy()
        existing_times = pd.to_datetime(
            existing["time"], errors="coerce", utc=True
        ).dropna()
        existing_intervals = {pd.Timestamp(value) for value in existing_times}

    missing_observed = sorted({
        interval_start
        for _minutes, target_time, interval_start in target_specs
        if target_time <= now and interval_start not in existing_intervals
    })
    messages: list[str] = []
    if missing_observed:
        ok, message, frame = client.fetch_kp_ap_indices(
            start_time=missing_observed[0].isoformat(),
            end_time=missing_observed[-1].isoformat(),
        )
        messages.append(message)
        if ok and not frame.empty:
            observed_frames.append(frame)

    observed = (
        pd.concat(observed_frames, ignore_index=True)
        if observed_frames else pd.DataFrame()
    )
    forecast = pd.DataFrame()
    if any(target_time > now for _minutes, target_time, _interval in target_specs):
        ok, message, frame = client.fetch_gfz_kp_forecast()
        messages.append(message)
        if ok and not frame.empty:
            forecast = frame

    resolved = _resolve_kp_horizons(
        analysis,
        observed,
        forecast,
        now=now,
    )
    unavailable = int((resolved["evidence_role"] == "unavailable").sum())
    if unavailable:
        detail = "; ".join(message for message in messages if message)
        suffix = f" ({detail})" if detail else ""
        message = f"{unavailable} Kp horizon assessment(s) unavailable.{suffix}"
    else:
        message = "Kp +30/+90/+180/+360 minute horizon evidence resolved."
    return resolved, message


def _resolve_kp_horizons(
    analysis: pd.Timestamp,
    observed: pd.DataFrame,
    forecast: pd.DataFrame,
    *,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    analysis = pd.Timestamp(analysis)
    if analysis.tzinfo is None:
        analysis = analysis.tz_localize("UTC")
    else:
        analysis = analysis.tz_convert("UTC")
    current = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if current.tzinfo is None:
        current = current.tz_localize("UTC")
    else:
        current = current.tz_convert("UTC")

    observed_kp = pd.DataFrame()
    if not observed.empty and {"variable", "time", "value"}.issubset(observed):
        observed_kp = observed[observed["variable"] == "Kp"].copy()
        observed_kp["time"] = pd.to_datetime(
            observed_kp["time"], errors="coerce", utc=True
        )
        observed_kp["value"] = pd.to_numeric(
            observed_kp["value"], errors="coerce"
        )
        observed_kp = observed_kp.dropna(subset=["time", "value"])

    forecast_rows = forecast.copy()
    if not forecast_rows.empty:
        for column in ("interval_start", "issue_time"):
            if column in forecast_rows:
                forecast_rows[column] = pd.to_datetime(
                    forecast_rows[column], errors="coerce", utc=True
                )

    rows: list[dict[str, Any]] = []
    for horizon_minutes in KP_HORIZON_MINUTES:
        target_time = analysis + pd.Timedelta(minutes=horizon_minutes)
        interval_start = target_time.floor("3h")
        row: dict[str, Any] = {
            "horizon_minutes": horizon_minutes,
            "target_time": target_time,
            "interval_start": interval_start,
            "value": float("nan"),
            "evidence_role": "unavailable",
            "source": "Unavailable",
            "ensemble_maximum": float("nan"),
            "probability_kp_ge_8": float("nan"),
            "data_status": "unavailable",
            "issue_time": pd.NaT,
            "availability_reason": "",
        }

        if target_time <= current:
            match = (
                observed_kp[observed_kp["time"] == interval_start]
                if not observed_kp.empty
                else pd.DataFrame()
            )
            if not match.empty:
                evidence = match.iloc[-1]
                row.update({
                    "value": float(evidence["value"]),
                    "evidence_role": "observed_backtesting",
                    "source": "GFZ observed outcome — backtesting only",
                    "data_status": str(
                        evidence.get("data_status", "observed") or "observed"
                    ),
                    "availability_reason": "Observed target interval is available.",
                })
            else:
                row["availability_reason"] = (
                    "GFZ observed outcome is not available for this past target interval."
                )
            rows.append(row)
            continue

        required = {
            "interval_start", "median", "maximum",
            "probability_kp_ge_8", "issue_time",
        }
        if required.issubset(forecast_rows):
            match = forecast_rows[
                forecast_rows["interval_start"] == interval_start
            ]
        else:
            match = pd.DataFrame()
        if not match.empty:
            evidence = match.iloc[-1]
            issue_time = evidence["issue_time"]
            median = pd.to_numeric(pd.Series([evidence["median"]]), errors="coerce").iloc[0]
            maximum = pd.to_numeric(pd.Series([evidence["maximum"]]), errors="coerce").iloc[0]
            probability = pd.to_numeric(
                pd.Series([evidence["probability_kp_ge_8"]]), errors="coerce"
            ).iloc[0]
            fresh = (
                pd.notna(issue_time)
                and issue_time <= target_time
                and issue_time <= current + KP_PUBLICATION_DELAY_TOLERANCE
                and current - issue_time <= KP_FORECAST_MAX_AGE
                and issue_time >= analysis - KP_FORECAST_MAX_AGE
            )
            valid_values = (
                pd.notna(median) and pd.notna(maximum) and pd.notna(probability)
            )
            if fresh and valid_values:
                row.update({
                    "value": float(median),
                    "evidence_role": "official_forecast",
                    "source": "GFZ official PAGER/SWIFT ensemble forecast",
                    "ensemble_maximum": float(maximum),
                    "probability_kp_ge_8": float(probability),
                    "data_status": "forecast",
                    "issue_time": pd.Timestamp(issue_time),
                    "availability_reason": "Fresh aligned official ensemble row is available.",
                })
            else:
                row["availability_reason"] = (
                    "No fresh, aligned official GFZ ensemble forecast is available."
                )
        else:
            row["availability_reason"] = (
                "No fresh, aligned official GFZ ensemble forecast is available."
            )
        rows.append(row)

    return pd.DataFrame(rows, columns=KP_HORIZON_COLUMNS)


def load_data(
    source: str = "api",
    model: str = "AIDA",
    start_time: str | None = None,
    end_time: str | None = None,
    variables: list[str] | None = None,
    region: dict[str, float] | None = None,
    local_file: str | None = None,
    grid_step: float = 10.0,
    progress_callback: Any | None = None,
) -> tuple[pd.DataFrame, LoadStatus]:
    del local_file
    status = LoadStatus()
    warnings: list[str] = []

    if source != "api":
        status.source = "none"
        status.message = "Only SERENE API data mode is supported."
        return pd.DataFrame(), status
    if model != "AIDA":
        status.source = "none"
        status.message = "Only the verified SERENE AIDA model is supported."
        return pd.DataFrame(), status

    selected_region = region or {
        "lat_min": -90.0,
        "lat_max": 90.0,
        "lon_min": -180.0,
        "lon_max": 180.0,
    }
    try:
        local_map_points = estimate_target_points(selected_region, grid_step)
    except (AidaGridError, KeyError, TypeError, ValueError) as exc:
        status.source = "none"
        status.message = f"Invalid regional grid: {exc}"
        return pd.DataFrame(), status

    requested_times = list(dict.fromkeys(
        value for value in (start_time, end_time) if value
    ))
    request_specs: list[tuple[str | None, str]] = []
    if not requested_times:
        request_specs = [(None, "ultra")]
    else:
        for value in requested_times:
            try:
                parsed = normalise_aida_request_time(value)
            except ValueError:
                warnings.append(f"Invalid requested AIDA time: {value}")
                continue
            latency = _aida_latency(parsed)
            request_specs.append((parsed.isoformat(), latency))
        request_specs = list(dict.fromkeys(request_specs))

    client = SereneClient()
    aida_frames: list[pd.DataFrame] = []
    downloaded_count = 0
    download_messages: list[str] = []
    total_requests = max(len(request_specs), 1)
    for index, (requested_time, latency) in enumerate(request_specs, start=1):
        if progress_callback:
            progress_callback(index, total_requests)
        ok_download, download_message, payload = client.download_aida_raw_output(
            requested_time,
            latency,
        )
        download_messages.append(download_message)
        if not ok_download or payload is None:
            warnings.append(download_message)
            continue
        downloaded_count += 1
        try:
            frame = calculate_aida_grid(
                payload,
                selected_region,
                grid_step,
                variables,
            )
        except AidaGridError as exc:
            warnings.append(str(exc))
            continue
        if not frame.empty:
            aida_frames.append(frame)

    ok_indices, indices_message, indices_frame = client.fetch_kp_ap_indices(
        start_time=start_time,
        end_time=end_time,
    )
    if not ok_indices or indices_frame.empty:
        warnings.append(indices_message)
        indices_frame = pd.DataFrame()

    actual_output_times: list[str] = []
    for frame in aida_frames:
        if "actual_output_time" not in frame.columns:
            continue
        for value in pd.to_datetime(
            frame["actual_output_time"], errors="coerce", utc=True
        ).dropna().unique():
            iso = pd.Timestamp(value).isoformat()
            if iso not in actual_output_times:
                actual_output_times.append(iso)

    metadata = {
        "model": model,
        "cadences": list(dict.fromkeys(latency for _time, latency in request_specs)),
        "indices_message": indices_message,
        "kp_ap_source": "GFZ Helmholtz Centre for Geosciences",
        "kp_ap_source_latest_time": (
            pd.Timestamp(client.kp_ap_source_latest_time).isoformat()
            if getattr(client, "kp_ap_source_latest_time", None) is not None
            else None
        ),
        "kp_ap_data_statuses": list(
            getattr(client, "kp_ap_data_statuses", []) or []
        ),
        "requested_times": requested_times,
        "request_specs": [
            {"time": requested or "latest", "latency": latency}
            for requested, latency in request_specs
        ],
        "download_messages": download_messages,
        "actual_output_times": actual_output_times,
        "aida_dataset_downloads": downloaded_count,
        "local_map_points": local_map_points,
        "grid_step_degrees": float(grid_step),
        "upstream_interpreter": (
            f"breid-phys/aida-ionosphere {UPSTREAM_AIDA_VERSION}"
        ),
    }

    if aida_frames:
        frames = list(aida_frames)
        if not indices_frame.empty:
            frames.append(indices_frame)
        combined = pd.concat(frames, ignore_index=True)
        status.source = "api"
        status.ok = True
        status.message = (
            f"Loaded {len(combined)} rows from {downloaded_count} raw AIDA "
            "state(s), with regional values calculated locally."
        )
        status.warnings.extend(warnings)
        status.metadata = metadata
        return combined, status

    if not indices_frame.empty:
        status.source = "indices"
        status.ok = False
        status.message = (
            "Global Kp/ap indices loaded, but regional AIDA data could not be calculated."
        )
        status.warnings.extend(warnings)
        status.metadata = metadata
        return indices_frame.reset_index(drop=True), status

    status.source = "none"
    status.ok = False
    status.message = "SERENE API returned no usable regional AIDA data."
    status.warnings.extend(warnings)
    status.metadata = metadata
    return pd.DataFrame(), status


def _filter_selected_variables(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    selected = set(variables)
    if "vTEC" in selected:
        selected.add("TEC")
    if "TEC" in selected:
        selected.add("vTEC")
    return df[df["variable"].isin(selected)]
