from __future__ import annotations

import re
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

from aida_grid import estimate_target_points
from app_utils import (
    AIDA_ARCHIVE_START,
    AIDA_ARCHIVE_START_UTC,
    advisory_metadata_for_load,
    build_provenance_metadata,
    build_data_preview,
    combine_date_time_iso,
    default_time_range,
    historical_risk_windows,
    loaded_api_state,
    make_streamlit_safe_dataframe,
    mappable_variable_options,
    parse_select_range_to_widgets,
    validate_requested_window,
)
from config import SERENE_API_TOKEN, reload_config, validate_config
from data_loader import IcaoProductBundle, LoadStatus, load_icao_products
from hf_coverage_ui import render_hf_propagation_case_study
from icao_message import generate_icao_message
from icao_risk import (
    FORECAST_HORIZONS,
    ICAO_COLORS,
    build_categorical_cells,
    build_evidence_completeness,
    build_icao_summary,
    build_overall_risk_cards,
    classify_auroral_absorption,
)
from icao_visualisation import create_icao_category_map
from serene_client import SereneClient
from trial_cache import (
    build_trial_bundle_zip,
    load_trial_bundle,
    make_trial_cache_key,
    save_trial_bundle,
    trial_cache_path,
)
from validation_ui import render_validation_section
from visualisation import (
    create_map_plot,
    create_time_series_plot,
)


st.set_page_config(
    page_title="Aviation Space Weather Dashboard",
    page_icon="SW",
    layout="wide",
    initial_sidebar_state="expanded",
)

reload_config()


def _init_state() -> None:
    defaults = {
        "data": pd.DataFrame(),
        "status": LoadStatus(),
        "alerts": pd.DataFrame(),
        "icao_bundle": IcaoProductBundle(),
        "icao_summary": pd.DataFrame(),
        "advisory_sequence": 0,
        "advisory_generated_time": None,
        "advisory_number": None,
        "api_connected": None,
        "api_message": "Not tested yet.",
        "config_warnings": validate_config(),
        "trial_cache_key": None,
        "last_successful_refresh": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _render_cloud_api_hint() -> None:
    if SERENE_API_TOKEN:
        return
    st.info(
        "SERENE API is not configured. This app is API-only and does not load "
        "local sample datasets. Add SERENE_API_BASE_URL, SERENE_API_TOKEN, "
        "SERENE_API_TIMEOUT, and SERENE_AUTH_SCHEME in Streamlit Cloud Secrets, "
        "then reboot the app."
    )


def _inject_dashboard_css() -> None:
    st.markdown(
        """
        <style>
        .risk-card {
            border: 1px solid rgba(255,255,255,0.18);
            border-radius: 10px;
            padding: 0.85rem 1rem;
            background: #111827;
            min-height: 94px;
        }
        .risk-card-label {
            color: #cbd5e1;
            font-size: 0.84rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .risk-card-status {
            font-size: 1.65rem;
            font-weight: 700;
            line-height: 1.25;
        }
        .risk-card-ok {border-left: 7px solid #2E7D32;}
        .risk-card-moderate {border-left: 7px solid #F9A825;}
        .risk-card-severe {border-left: 7px solid #C62828;}
        .risk-card-unavailable {border-left: 7px solid #95A5A6;}
        .risk-card-partial, .risk-card-partial-data,
        .risk-card-moderate-partial-data, .risk-card-severe-partial-data {
            border-left: 7px solid #F9A825;
        }
        .risk-card-severe-partial-data {border-left-color: #C62828;}
        .risk-card-detail {
            margin-top: 0.4rem;
            color: #94a3b8;
            font-size: 0.78rem;
            line-height: 1.3;
        }
        .study-card {
            border: 1px solid #315175;
            border-radius: 12px;
            padding: 1rem 1.1rem;
            background: linear-gradient(135deg, #10223a, #0c1728);
            margin-bottom: 0.8rem;
        }
        .study-card strong {color: #e2e8f0;}
        .study-card p {color: #a8bad0; margin: 0.35rem 0 0;}
        .provenance-strip {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 1px;
            overflow: hidden;
            border: 1px solid #334155;
            border-radius: 10px;
            background: #334155;
            margin: 0.75rem 0;
        }
        .provenance-item {
            min-width: 0;
            padding: 0.7rem 0.8rem;
            background: #0f172a;
        }
        .provenance-item span {
            display: block;
            color: #94a3b8;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.035em;
        }
        .provenance-item strong {
            display: block;
            color: #f8fafc;
            font-size: 0.9rem;
            white-space: normal;
            overflow-wrap: anywhere;
        }
        @media (max-width: 900px) {
            .provenance-strip {grid-template-columns: 1fr 1fr;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar() -> dict:
    st.sidebar.markdown("# SERENE AIDA")
    st.sidebar.markdown("*Aviation Space Weather Monitor*")
    st.sidebar.markdown("---")

    params: dict = {"source": "api"}

    if st.session_state.config_warnings:
        with st.sidebar.expander("Configuration issues", expanded=True):
            for msg in st.session_state.config_warnings:
                st.warning(msg)

    data_loading_mode = st.sidebar.radio(
        "Data loading mode",
        ["Cached trial output", "Live SERENE API"],
        index=0,
        help=(
            "Cached trial output loads processed demo/validation results from "
            "the Git repository when available. Live SERENE API fetches new data."
        ),
    )
    params["data_loading_mode"] = data_loading_mode
    st.sidebar.info(f"Data loading mode: {data_loading_mode}")
    params["model"] = "AIDA"
    st.sidebar.caption("Verified model: AIDA")

    params["mode"] = "Full ICAO-style mode"
    params["include_three_hour_window"] = True
    params["include_psd_baseline"] = True
    st.sidebar.caption(
        "Full ICAO-style processing is always used: the dashboard attempts the "
        "Max-3h observation window and 30-day PSD baseline."
    )

    _default_start, default_end = default_time_range()
    if "end_date" not in st.session_state:
        st.session_state.end_date = default_end.date()
    if "end_time_clock" not in st.session_state:
        st.session_state.end_time_clock = default_end.time()
    selected_date = st.session_state.get("end_date")
    if selected_date is not None and selected_date < AIDA_ARCHIVE_START:
        st.session_state.end_date = AIDA_ARCHIVE_START
    st.sidebar.markdown("#### Analysis time")
    st.sidebar.caption(
        "The default selection is 15 minutes behind UTC. Choose any supported "
        "historical analysis time manually."
    )
    st.sidebar.caption(
        "The selected analysis time anchors the product; its preceding "
        "three-hour window is loaded automatically."
    )
    st.sidebar.caption(f"AIDA archive start: {AIDA_ARCHIVE_START_UTC.strftime('%Y-%m-%d %H:%M UTC')}.")

    analysis_date_col, analysis_time_col = st.sidebar.columns(2)
    with analysis_date_col:
        end_date = st.date_input(
            "Analysis date",
            min_value=AIDA_ARCHIVE_START,
            key="end_date",
        )
    with analysis_time_col:
        end_clock = st.time_input(
            "Analysis time UTC",
            step=timedelta(minutes=1),
            key="end_time_clock",
        )

    params["end_time"] = combine_date_time_iso(end_date, end_clock)
    params["start_time"] = (
        pd.Timestamp(params["end_time"]) - pd.Timedelta(hours=3)
    ).isoformat()
    params["variables"] = ["TEC", "MUF3000F2"]
    st.sidebar.caption(f"Analysis ISO time: {params['end_time']}")
    st.sidebar.caption("Fixed ICAO inputs: TEC and MUF3000F2 from SERENE AIDA.")

    with st.sidebar.expander("Demo / validation storm windows", expanded=False):
        st.caption(
            "These windows are only shortcuts for testing historical storm-like "
            "periods and for pretending the dashboard is running in the past."
        )
        st.caption(
            "Custom analysis time can be entered manually above; event rows only "
            "change the time after you press the shortcut button."
        )
        windows = historical_risk_windows()
        selection = st.dataframe(
            windows,
            width="stretch",
            hide_index=True,
            height=220,
            selection_mode="single-row",
            on_select="rerun",
            key="event_windows_sidebar",
        )
        if isinstance(selection, dict):
            selected_rows = selection.get("selection", {}).get("rows", [])
        else:
            selected_rows = getattr(getattr(selection, "selection", None), "rows", [])
        if st.button("Use selected event time", key="apply_event_time_sidebar"):
            _apply_selected_historical_range(selected_rows, windows)

    st.sidebar.markdown("#### Region selection")
    with st.sidebar.expander("Bounding box and grid step", expanded=True):
        lat_min = st.number_input("Lat min", value=-90.0, min_value=-90.0, max_value=90.0)
        lat_max = st.number_input("Lat max", value=90.0, min_value=-90.0, max_value=90.0)
        lon_min = st.number_input("Lon min", value=-180.0, min_value=-180.0, max_value=180.0)
        lon_max = st.number_input("Lon max", value=180.0, min_value=-180.0, max_value=180.0)
        params["grid_step"] = st.slider("Grid step (degrees)", 2.0, 30.0, 15.0, 1.0)
        local_points = estimate_target_points(
            {
                "lat_min": lat_min,
                "lat_max": lat_max,
                "lon_min": lon_min,
                "lon_max": lon_max,
            },
            params["grid_step"],
        )
        st.caption(
            "The default grid is global for aviation-scale awareness. Use a "
            "smaller bounding box or finer grid step for regional analysis."
        )
        st.caption(
            f"Local map points: {local_points:,}. One raw AIDA state is downloaded "
            "per output time; this grid is calculated locally."
        )

    params["region"] = {
        "lat_min": lat_min,
        "lat_max": lat_max,
        "lon_min": lon_min,
        "lon_max": lon_max,
    }

    st.sidebar.markdown("---")
    if st.sidebar.button("Test SERENE API connection", width="stretch"):
        with st.spinner("Testing connection..."):
            ok, msg = SereneClient().test_connection()
            st.session_state.api_connected = ok
            st.session_state.api_message = msg
        if ok:
            st.sidebar.success(msg)
        else:
            st.sidebar.warning(msg)

    if st.sidebar.button("Load / Refresh data", type="primary", width="stretch"):
        _do_load(params)

    st.sidebar.caption("Prototype research system, not for operational aviation decisions.")
    return params


@st.cache_data(show_spinner=False)
def _load_trial_bundle_cached(cache_key: str):
    return load_trial_bundle(cache_key)


def _do_load(params: dict) -> None:
    progress_bar = st.progress(0.0, text="Preparing...")
    progress_state = {"done": 0, "total": 1}

    def _on_api_progress(done: int, total: int, label: str = "AIDA data") -> None:
        progress_state["done"] = done
        progress_state["total"] = max(total, 1)
        progress_bar.progress(
            done / progress_state["total"],
            text=f"{label}: {done}/{total}...",
        )

    cleared = advisory_metadata_for_load(
        False,
        st.session_state.advisory_sequence,
        pd.Timestamp.now(tz="UTC"),
    )
    st.session_state.advisory_generated_time = cleared["generated_time"]
    st.session_state.advisory_number = cleared["number"]

    try:
        validation_error = validate_requested_window(
            params["start_time"], params["end_time"]
        )
        if validation_error:
            failed_status = LoadStatus(
                source="none", ok=False, message=validation_error
            )
            st.session_state.data = pd.DataFrame()
            st.session_state.status = failed_status
            st.session_state.icao_bundle = IcaoProductBundle(status=failed_status)
            st.session_state.icao_summary = pd.DataFrame()
            return
        cache_key = make_trial_cache_key(
            params["end_time"],
            params["region"],
            params.get("grid_step", 15.0),
            params["mode"],
        )
        st.session_state.trial_cache_key = cache_key
        if params.get("data_loading_mode") == "Cached trial output":
            try:
                progress_bar.progress(0.2, text="Checking cached trial output...")
                bundle, summary, data = _load_trial_bundle_cached(cache_key)
                _set_loaded_result(bundle, summary, data)
                return
            except FileNotFoundError:
                fallback_warning = (
                    "Cached trial output not found for this selection; loading "
                    "from SERENE API instead."
                )
            except Exception as exc:
                fallback_warning = (
                    "Cached trial output could not be loaded; loading from "
                    f"SERENE API instead. Cache error: {exc}"
                )
        else:
            fallback_warning = None
        bundle = load_icao_products(
            analysis_time=params["end_time"],
            variables=["TEC", "MUF3000F2"],
            region=params.get("region"),
            grid_step=params.get("grid_step", 15.0),
            include_three_hour_window=params.get("include_three_hour_window", True),
            include_psd_baseline=params.get("include_psd_baseline", True),
            progress_callback=_on_api_progress,
        )
        progress_bar.progress(1.0, text="Generating ICAO-style research products...")
        if fallback_warning:
            bundle.status.warnings = [fallback_warning, *bundle.status.warnings]
            bundle.status.metadata["cache_key"] = cache_key
            bundle.status.metadata["cache_fallback"] = True
        data = _build_display_data(bundle)
        summary = build_icao_summary(
            bundle.products,
            bundle.indices,
            eligible=bundle.kp_storm_eligible,
            kp_horizons=bundle.kp_horizons,
        )
        _set_loaded_result(bundle, summary, data)
        st.session_state.alerts = pd.DataFrame()
    finally:
        progress_bar.empty()


def _set_loaded_result(
    bundle: IcaoProductBundle,
    summary: pd.DataFrame,
    data: pd.DataFrame,
) -> None:
    st.session_state.data = data
    st.session_state.status = bundle.status
    st.session_state.icao_bundle = bundle
    st.session_state.icao_summary = summary
    if bundle.status.ok:
        generated = pd.Timestamp.now(tz="UTC")
        st.session_state.last_successful_refresh = generated.isoformat()
        advisory = advisory_metadata_for_load(
            True, st.session_state.advisory_sequence, generated
        )
        st.session_state.advisory_sequence = advisory["sequence"]
        st.session_state.advisory_generated_time = advisory["generated_time"]
        st.session_state.advisory_number = advisory["number"]


def _build_display_data(bundle: IcaoProductBundle) -> pd.DataFrame:
    frames = [
        frame for frame in (bundle.products, bundle.indices)
        if frame is not None and not frame.empty
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _source_label(status: LoadStatus) -> str:
    return {
        "api": "Live SERENE API",
        "trial_cache": "Cached trial output",
        "indices": "SERENE global indices only",
        "none": "No data",
    }.get(status.source, status.source)


def _apply_selected_historical_range(selected_rows: list[int], windows: pd.DataFrame) -> None:
    if not selected_rows:
        return
    row_index = selected_rows[0]
    if row_index >= len(windows):
        return
    parsed = parse_select_range_to_widgets(str(windows.iloc[row_index]["Select range"]))
    if parsed is None:
        return
    if any(st.session_state.get(key) != value for key, value in parsed.items()):
        st.session_state.pending_time_range_widgets = parsed
        st.rerun()


def _apply_pending_time_range() -> None:
    pending = st.session_state.pop("pending_time_range_widgets", None)
    if not pending:
        return
    for key, value in pending.items():
        st.session_state[key] = value


def _format_refresh_time(value: object) -> str:
    if value in (None, ""):
        return "N/A"
    try:
        if pd.isna(value):
            return "N/A"
    except (TypeError, ValueError):
        pass
    try:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
    except (TypeError, ValueError):
        return str(value)
    return timestamp.strftime("%Y-%m-%d %H:%M UTC")


def _kp_ap_source_freshness_caption(status: LoadStatus) -> str | None:
    value = status.metadata.get("kp_ap_source_latest_time")
    if not isinstance(value, str) or re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|\+00:00)",
        value,
    ) is None:
        return None
    try:
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp):
            return None
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
    except (TypeError, ValueError):
        return None
    latest = timestamp.strftime("%Y-%m-%d %H:%M UTC")
    if status.metadata.get("kp_ap_index_status") == "unavailable":
        return f"GFZ Kp/ap unavailable — latest source timestamp: {latest}"
    statuses = [
        str(item) for item in status.metadata.get("kp_ap_data_statuses", [])
        if str(item) in {"preliminary", "definitive"}
    ]
    missing_indices = [
        str(item) for item in status.metadata.get("kp_ap_missing_indices", [])
        if str(item) in {"Kp", "ap"}
    ]
    status_text = ", ".join(dict.fromkeys(statuses))
    suffix = f"; loaded status: {status_text}" if status_text else ""
    if "ap" in missing_indices and "Kp" not in missing_indices:
        return (
            "GFZ Kp loaded; ap unavailable — latest source timestamp: "
            f"{latest}{suffix}"
        )
    return f"GFZ Kp/ap — latest source timestamp: {latest}{suffix}"


def _actual_analysis_output_time() -> pd.Timestamp | None:
    products = st.session_state.icao_bundle.products
    if products is not None and not products.empty:
        analysis_rows = products
        if "product_kind" in products.columns:
            filtered = products[products["product_kind"] == "analysis"]
            if not filtered.empty:
                analysis_rows = filtered
        if "actual_output_time" in analysis_rows.columns:
            values = pd.to_datetime(
                analysis_rows["actual_output_time"], errors="coerce", utc=True
            ).dropna()
            if not values.empty:
                return pd.Timestamp(values.max())
    metadata_value = st.session_state.status.metadata.get(
        "actual_analysis_output_time"
    )
    if metadata_value:
        values = pd.to_datetime(
            pd.Series([metadata_value]), errors="coerce", utc=True
        ).dropna()
        if not values.empty:
            return pd.Timestamp(values.max())
    return None


def _render_connection_panel(params: dict) -> None:
    st.subheader("SERENE API and data status")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        status: LoadStatus = st.session_state.status
        api_level, api_text = loaded_api_state(
            status,
            st.session_state.api_connected,
            st.session_state.api_message,
        )
        if api_level == "success":
            st.success(f"API: {api_text}")
        elif api_level == "warning":
            st.warning(f"API: {api_text}")
        else:
            st.info(f"API: {api_text}")

    with c2:
        st.metric("Current data source", _source_label(status))
    with c3:
        st.metric("Rows loaded", f"{len(st.session_state.data):,}")
    with c4:
        st.metric(
            "Total official AIDA downloads",
            int(status.metadata.get(
                "total_official_aida_downloads",
                int(status.metadata.get("aida_dataset_downloads", 0))
                + int(status.metadata.get("analysis_downloads", 0))
                + int(status.metadata.get("forecast_downloads", 0)),
            )),
        )

    s1, s2, s3, s4, s5 = st.columns(5)
    with s1:
        st.metric("Rolling/analysis states", int(status.metadata.get("rolling_analysis_downloads", 0)))
    with s2:
        st.metric("Official forecast states", int(status.metadata.get("forecast_downloads", 0)))
    with s3:
        st.metric("PSD baseline states", int(status.metadata.get("baseline_downloads", 0)))
    with s4:
        st.metric("Kp/ap index status", str(status.metadata.get("kp_ap_index_status", "not requested")))
    with s5:
        st.metric("Local map points", f"{int(status.metadata.get('local_map_points', 0)):,}")

    kp_ap_freshness = _kp_ap_source_freshness_caption(status)
    if kp_ap_freshness:
        st.caption(kp_ap_freshness)

    if status.message:
        if status.ok:
            st.info(status.message)
        else:
            st.error(status.message)
    if status.metadata.get("forecast_request_audit"):
        st.info(_forecast_availability_message(status))
    for warn in status.warnings:
        st.warning(warn)

    if status.metadata:
        st.caption(
            "Each AIDA raw state is downloaded once per output time, then all "
            "selected regional grid points are calculated locally."
        )

    requested_time = status.metadata.get("analysis_time", params["end_time"])
    actual_time = _actual_analysis_output_time()
    provenance = build_provenance_metadata(
        requested_time,
        actual_time,
        st.session_state.last_successful_refresh,
        pd.Timestamp.now(tz="UTC"),
        int(status.metadata.get("forecast_downloads", 0)),
    )
    provenance_html = "".join(
        '<div class="provenance-item">'
        f'<span>{item["label"]}</span><strong>{item["value"]}</strong></div>'
        for item in provenance
    )
    st.markdown(
        f'<div class="provenance-strip">{provenance_html}</div>',
        unsafe_allow_html=True,
    )


def _render_empty_state() -> None:
    st.info(
        "Click Load / Refresh data in the sidebar to load cached trial output "
        "when available, or fetch Live SERENE API data for new analysis times. "
        "Each raw AIDA state is downloaded once per output time and interpreted by "
        "the official AIDA package; all requested map points are calculated locally. "
        "Cached trial outputs store processed research results for demonstration."
    )
    st.subheader("ICAO-style SERENE-only products")
    st.info(
        "The category map, summary table, and research messages appear after "
        "SERENE analysis and prediction products are loaded."
    )
    with st.expander("Quick start"):
        st.markdown(
            """
            1. Configure SERENE_API_BASE_URL and SERENE_API_TOKEN.
            2. Test the SERENE API connection.
            3. Load API data for an analysis time and selected region.
            4. Inspect Latest, historical Max-3h, and all four visible forecast groups.
            5. Open Forecast request audit for request details across all four horizons.
            """
        )


def _render_overall_risk_cards(summary: pd.DataFrame) -> None:
    st.subheader("Current aviation risk and evidence status")
    cards = build_overall_risk_cards(summary)
    completeness = build_evidence_completeness(summary)
    missing = ", ".join(completeness["missing"]) or "None"
    details = {
        "GNSS Risk": "Vertical TEC evidence",
        "HF COM Risk": "PSD and global Kp proxy evidence",
        "Overall Risk": "Worst supported severity with missing-data guard",
        "Data Completeness": (
            f'{completeness["available"]}/{completeness["required"]} required '
            f'inputs ({completeness["percent"]}%)'
        ),
    }
    columns = st.columns(4)
    for column, (label, status) in zip(columns, cards.items()):
        css_status = re.sub(r"[^a-z0-9]+", "-", str(status).casefold()).strip("-")
        with column:
            st.markdown(
                f"""
                <div class="risk-card risk-card-{css_status}">
                    <div class="risk-card-label">{label}</div>
                    <div class="risk-card-status">{status}</div>
                    <div class="risk-card-detail">{details[label]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.subheader("Evidence completeness")
    if completeness["status"] == "COMPLETE":
        st.success(
            f'All {completeness["required"]} required risk inputs are available.'
        )
    elif completeness["status"] == "PARTIAL":
        st.warning(
            f'Available: {completeness["available"]}/{completeness["required"]} '
            f'({completeness["percent"]}%). Missing: {missing}.'
        )
    else:
        st.error("Required risk inputs are unavailable; no overall risk is asserted.")
    st.caption(
        "Severity and evidence completeness are separate. Missing inputs cannot "
        "silently produce an unqualified OK result."
    )


def _render_standalone_hf_study(df: pd.DataFrame) -> None:
    st.subheader("Standalone HF Communication Engineering Study")
    st.markdown(
        """
        <div class="study-card">
            <strong>Quiet-versus-disturbed HF coverage case study</strong>
            <p>
                A separate quantitative engineering investigation complements the
                live SERENE/AIDA risk monitor. It is research evidence, not an
                integrated operational warning or validated flight-planning tool.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_hf_propagation_case_study(df)


def _style_pecasus_table(summary: pd.DataFrame):
    summary = make_streamlit_safe_dataframe(summary).astype(str)
    status_columns = [
        column for column in [
            "Status",
            "Latest status",
            "Max-3h status",
            "+30 min status",
            "+90 min status",
            "+3h status",
            "+6h status",
        ] if column in summary.columns
    ]

    def _status_cell(value: object) -> str:
        status = str(value)
        color = ICAO_COLORS.get(status, "#95A5A6")
        text = "#ffffff" if status != "MODERATE" else "#111827"
        return f"background-color: {color}; color: {text}; font-weight: 700;"

    def _cell_style(_: object) -> str:
        return (
            "background-color: #0b1220; color: #e5e7eb; "
            "border: 1px solid #334155;"
        )

    styler = summary.style.applymap(_cell_style)
    for column in status_columns:
        styler = styler.applymap(_status_cell, subset=[column])
    return styler


def _available_primary_periods(status: LoadStatus) -> list[int]:
    values = status.metadata.get("available_primary_forecast_periods", [])
    available = {int(value) for value in values if str(value).isdigit()}
    return [period for period in (30, 90, 180, 360) if period in available]


def _visible_summary_columns(
    summary: pd.DataFrame,
    status: LoadStatus,
) -> list[str]:
    del status
    return list(summary.columns)


_FORECAST_SOURCE_COLUMNS = {
    "+30 min": "+30 min source",
    "+90 min": "+90 min source",
    "+3 h": "+3h source",
    "+6 h": "+6h source",
}


def _forecast_provenance_label(source: object) -> str:
    if source is None or pd.isna(source):
        return "UNAVAILABLE"
    text = str(source).strip().casefold()
    if not text or text in {"unavailable", "n/a", "none"}:
        return "UNAVAILABLE"
    if "serene official forecast" in text:
        return "OFFICIAL SERENE API — direct HDF5"
    if "gfz official" in text and "forecast" in text:
        return "OFFICIAL GFZ API — PAGER/SWIFT"
    if "gfz observed outcome" in text or "backtesting only" in text:
        return "OBSERVED GFZ — backtesting only"
    if "dashboard-generated trend-based" in text:
        return "DASHBOARD ESTIMATE — trend extrapolation"
    if "dashboard-generated persistence" in text:
        return "DASHBOARD ESTIMATE — persistence"
    return "UNAVAILABLE"


def _summary_with_provenance(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    for label, source_column in _FORECAST_SOURCE_COLUMNS.items():
        provenance_column = f"{label.replace(' h', 'h')} provenance"
        values = (
            result[source_column].map(_forecast_provenance_label)
            if source_column in result.columns
            else pd.Series("UNAVAILABLE", index=result.index)
        )
        if provenance_column in result.columns:
            result[provenance_column] = values
            continue
        insert_at = (
            result.columns.get_loc(source_column) + 1
            if source_column in result.columns else len(result.columns)
        )
        result.insert(insert_at, provenance_column, values)
    return result


def _forecast_provenance_table(summary: pd.DataFrame) -> pd.DataFrame:
    columns = ["Indicator", "+30 min", "+90 min", "+3 h", "+6 h"]
    if not isinstance(summary, pd.DataFrame) or summary.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for _, item in summary.iterrows():
        row = {"Indicator": str(item.get("Indicator", "N/A"))}
        for label, source_column in _FORECAST_SOURCE_COLUMNS.items():
            row[label] = _forecast_provenance_label(item.get(source_column))
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _kp_horizon_evidence_table(kp_horizons: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Horizon", "Target UTC", "Evidence role", "Primary Kp",
        "Primary status", "Ensemble maximum", "P(Kp >= 8)",
        "Issue UTC", "Data status", "Source",
    ]
    if not isinstance(kp_horizons, pd.DataFrame) or kp_horizons.empty:
        return pd.DataFrame(columns=columns)
    role_labels = {
        "official_forecast": "Official forecast",
        "observed_backtesting": "Observed outcome (backtesting only)",
        "unavailable": "Unavailable",
    }
    rows = []
    work = kp_horizons.copy()
    work["horizon_minutes"] = pd.to_numeric(
        work.get("horizon_minutes"), errors="coerce"
    )
    work = work.sort_values("horizon_minutes")
    horizon_labels = {
        30: "+30 min",
        90: "+90 min",
        180: "+3 h",
        360: "+6 h",
    }
    for _, item in work.iterrows():
        value = pd.to_numeric(pd.Series([item.get("value")]), errors="coerce").iloc[0]
        maximum = pd.to_numeric(
            pd.Series([item.get("ensemble_maximum")]), errors="coerce"
        ).iloc[0]
        probability = pd.to_numeric(
            pd.Series([item.get("probability_kp_ge_8")]), errors="coerce"
        ).iloc[0]
        role = str(item.get("evidence_role", "unavailable"))
        rows.append({
            "Horizon": horizon_labels.get(
                int(item["horizon_minutes"]), f'+{int(item["horizon_minutes"])} min'
            ),
            "Target UTC": _format_refresh_time(item.get("target_time")),
            "Evidence role": role_labels.get(role, "Unavailable"),
            "Primary Kp": float(value) if pd.notna(value) else "N/A",
            "Primary status": (
                classify_auroral_absorption(value)
                if pd.notna(value) else "UNAVAILABLE"
            ),
            "Ensemble maximum": float(maximum) if pd.notna(maximum) else "N/A",
            "P(Kp >= 8)": f"{float(probability):.0%}" if pd.notna(probability) else "N/A",
            "Issue UTC": _format_refresh_time(item.get("issue_time")),
            "Data status": str(item.get("data_status", "unavailable")),
            "Source": str(item.get("source", "Unavailable")),
        })
    return pd.DataFrame(rows, columns=columns)


def _forecast_availability_message(status: LoadStatus) -> str:
    available = set(_available_primary_periods(status))
    audit = status.metadata.get("forecast_request_audit", [])
    outcomes = {
        int(item.get("forecast_parameter", 0)): str(item.get("outcome", ""))
        for item in audit
    }
    labels = {
        30: "+30 min",
        90: "+90 min",
        180: "+3 h",
        360: "+6 h",
    }
    periods = tuple(labels)
    if available == set(periods):
        return (
            "Official SERENE forecast availability this analysis cycle: "
            "+30 min, +90 min, +3 h and +6 h retrieved."
        )

    unavailable_reason = {
        "not_published": "not published",
        "authentication_failed": "authentication failed",
        "network_failed": "temporary network failure",
        "decode_failed": "could not be decoded",
    }
    availability = []
    for period in periods:
        label = labels[period]
        if period in available:
            availability.append(f"{label} retrieved")
        else:
            availability.append(
                f"{label} {unavailable_reason.get(outcomes.get(period), 'unavailable')}"
            )
    return (
        "Official SERENE forecast availability this analysis cycle: "
        + "; ".join(availability)
        + "."
    )


def _render_pecasus_summary_table() -> None:
    summary = st.session_state.icao_summary
    st.subheader("ICAO/PECASUS-style summary table")
    st.caption(
        "This table includes only SERENE-supported, derived, or proxy indicators. "
        "The risk status describes the value, while the provenance label separately "
        "states whether that value was downloaded from an official API, calculated "
        "by the Dashboard, observed later for backtesting, or unavailable."
    )
    if summary.empty:
        st.info("Load SERENE data to create the PECASUS-style table.")
        return
    provenance = _forecast_provenance_table(summary)
    st.markdown("**Forecast provenance (direct, generated, backtesting or unavailable)**")
    st.dataframe(provenance, width="stretch", hide_index=True)
    st.caption(
        "OFFICIAL SERENE/GFZ API means the forecast was downloaded directly. "
        "DASHBOARD ESTIMATE means the value was calculated locally from SERENE "
        "analysis data and is not an official SERENE forecast. OBSERVED GFZ is a "
        "known historical outcome used only for backtesting."
    )
    exportable_summary = _summary_with_provenance(summary)
    visible = _visible_summary_columns(exportable_summary, st.session_state.status)
    st.dataframe(
        _style_pecasus_table(exportable_summary.loc[:, visible]),
        width="stretch",
        hide_index=True,
    )
    kp_evidence = _kp_horizon_evidence_table(
        st.session_state.icao_bundle.kp_horizons
    )
    if not kp_evidence.empty:
        st.markdown("**Kp +30/+90/+3h/+6h horizon evidence**")
        st.dataframe(kp_evidence, width="stretch", hide_index=True)
        st.caption(
            "Historical target times use GFZ observed outcomes for backtesting only; "
            "they are not archived forecasts. Future targets retain the official "
            "GFZ PAGER/SWIFT ensemble forecast provenance for the primary category. Ensemble "
            "maximum and P(Kp >= 8) show uncertainty without automatically raising "
            "the primary status."
        )


def _render_categorical_risk_map() -> None:
    bundle: IcaoProductBundle = st.session_state.icao_bundle
    st.subheader("Categorical risk map")
    st.caption(
        "Regional category maps are only created for spatial AIDA products. "
        "Global Kp/ap are excluded from regional map cells because they are "
        "planetary indices."
    )
    map_col, horizon_col = st.columns([2, 2])
    with map_col:
        indicator = st.selectbox(
            "Risk category map",
            ["Vertical TEC", "Post-Storm Depression"],
            key="risk_category_map_indicator",
        )
    with horizon_col:
        available_labels = list(FORECAST_HORIZONS)
        horizon = st.radio(
            "Prediction horizon",
            ["Latest", *available_labels],
            horizontal=True,
            key="risk_category_map_horizon",
        )
    cells = build_categorical_cells(
        bundle.products,
        indicator,
        horizon,
        kp_storm_eligible=bundle.kp_storm_eligible,
    )
    st.plotly_chart(
        create_icao_category_map(cells, f"{indicator} risk category — {horizon}"),
    )
    if indicator == "Post-Storm Depression":
        if bundle.kp_storm_eligible is None:
            st.warning("PSD map unavailable: complete 96-hour Kp history is missing.")
        elif bundle.kp_storm_eligible:
            st.info("PSD storm gate active: Kp reached at least 6 in the prior 96 hours.")
        else:
            st.info("PSD storm gate inactive: PSD risk is reported as OK until a Kp≥6 storm gate is met.")


def _render_raw_value_maps(df: pd.DataFrame) -> None:
    st.subheader("Raw variable maps")
    st.caption(
        "Raw AIDA maps use latest analysis values and continuous colour scales. "
        "These are data-value maps, not warning category maps."
    )
    map_df = df
    if "product_kind" in df.columns:
        analysis_rows = df[df["product_kind"] == "analysis"].copy()
        if not analysis_rows.empty:
            map_df = analysis_rows
    options = [
        variable for variable in ["TEC", "vTEC", "MUF3000F2", "MUF3000"]
        if variable in set(map_df.get("variable", pd.Series(dtype=str)).astype(str))
    ]
    if not options:
        options = mappable_variable_options(map_df)
    if not options:
        st.info("No SERENE AIDA variables with latitude/longitude are available for raw maps.")
        return
    selected_map_var = st.selectbox("Raw value map", options, key="raw_value_map_variable")
    st.plotly_chart(create_map_plot(map_df, variable=selected_map_var))
    if selected_map_var in {"MUF3000F2", "MUF3000"}:
        st.caption(
            "MUF3000F2 is shown only as a raw value. PSD risk is derived from its "
            "percentage depression relative to the 30-day same-UTC baseline."
        )


def _render_research_messages(summary: pd.DataFrame, params: dict) -> None:
    st.subheader("Automated text-based SPWX research messages")
    st.caption(
        "Messages are generated with STATUS: TEST and RESEARCH PROTOTYPE wording. "
        "They are not official ICAO advisories."
    )
    analysis_time = st.session_state.status.metadata.get(
        "analysis_time", params["end_time"]
    )
    generated_time = (
        st.session_state.advisory_generated_time or pd.Timestamp.now(tz="UTC")
    )
    advisory_number = st.session_state.advisory_number or f"{generated_time.year}/001"
    loaded_region = st.session_state.status.metadata.get(
        "loaded_region", params["region"]
    )
    tec = _summary_row(summary, "Vertical TEC")
    psd = _summary_row(summary, "Post-Storm Depression")
    kp = _summary_row(summary, "Auroral Absorption")
    available_periods = set(_available_primary_periods(st.session_state.status))

    def official_forecasts(row: pd.Series | None) -> dict[int, str | None]:
        if row is None:
            return {}
        labels = {30: "+30 min", 90: "+90 min"}
        return {
            period: _available_category(row[f"{label} status"])
            for period, label in labels.items()
            if period in available_periods
        }

    if tec is not None and tec["Status"] in {"OK", "MODERATE", "SEVERE"}:
        gnss = generate_icao_message(
            effect="GNSS",
            observed_time=analysis_time,
            observed_category=tec["Status"],
            region=loaded_region,
            forecasts=official_forecasts(tec),
            generated_time=generated_time,
            advisory_number=advisory_number,
        )
        st.code(gnss, language="text")
        st.download_button(
            "Download GNSS research message",
            data=gnss,
            file_name="serene_gnss_research_advisory.txt",
            mime="text/plain",
        )
        st.caption(
            "GNSS message is currently generated from Vertical TEC only because "
            "SERENE does not provide amplitude or phase scintillation inputs."
        )
    else:
        st.info("GNSS research message unavailable because SERENE TEC is unavailable.")

    hf_observed = _worst_available_category([
        psd["Status"] if psd is not None else None,
        kp["Status"] if kp is not None else None,
    ])
    if hf_observed is None:
        st.info("HF COM research message unavailable because SERENE inputs are unavailable.")
        return
    hf = generate_icao_message(
        effect="HF COM",
        observed_time=analysis_time,
        observed_category=hf_observed,
        region=loaded_region,
        forecasts=official_forecasts(psd),
        generated_time=generated_time,
        advisory_number=advisory_number,
    )
    st.code(hf, language="text")
    st.download_button(
        "Download HF COM research message",
        data=hf,
        file_name="serene_hf_com_research_advisory.txt",
        mime="text/plain",
    )
    st.caption(
        "HF COM message is generated from Post-Storm Depression and the global "
        "Kp auroral-absorption proxy only because PCA and SWF inputs are not "
        "available from SERENE."
    )


def _summary_row(summary: pd.DataFrame, indicator: str) -> pd.Series | None:
    rows = summary[summary["Indicator"] == indicator]
    return None if rows.empty else rows.iloc[0]


def _available_category(value: object) -> str | None:
    return str(value) if value in {"OK", "MODERATE", "SEVERE"} else None


def _worst_available_category(values: list[object]) -> str | None:
    priority = {"OK": 0, "MODERATE": 1, "SEVERE": 2}
    available = [str(value) for value in values if value in priority]
    return max(available, key=priority.get) if available else None


def _render_data_views(df: pd.DataFrame, alerts: pd.DataFrame) -> None:
    st.subheader("API/data metadata and raw data preview")

    var_options = sorted(df["variable"].dropna().unique()) if "variable" in df.columns else []
    if var_options:
        selected_time_var = st.selectbox(
            "Variable for bottom time-series preview", var_options, key="bottom_time_series_variable"
        )
        st.subheader("Time series")
        st.plotly_chart(create_time_series_plot(df, variable=selected_time_var))
    else:
        st.info("No variables are available for time-series preview.")

    st.dataframe(build_data_preview(df, alerts).head(100), width="stretch")

    with st.expander("Raw load metadata"):
        st.json(
            {
                "source": st.session_state.status.source,
                "message": st.session_state.status.message,
                "warnings": st.session_state.status.warnings,
                "metadata": st.session_state.status.metadata,
            }
        )
        st.download_button(
            "Download current API response as CSV",
            data=df.to_csv(index=False),
            file_name=f"space_weather_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )


def _render_trial_cache_export(params: dict) -> None:
    status: LoadStatus = st.session_state.status
    bundle: IcaoProductBundle = st.session_state.icao_bundle
    if not status.ok or bundle.products.empty:
        return
    cache_key = st.session_state.trial_cache_key or make_trial_cache_key(
        params["end_time"],
        params["region"],
        params.get("grid_step", 15.0),
        params["mode"],
    )
    with st.expander("Cached trial output tools", expanded=False):
        st.caption(
            "Use this locally after a successful Live SERENE API load to write "
            "processed trial outputs into the repository. Streamlit Cloud runtime "
            "writes are temporary; download the ZIP there, extract it under "
            "streamlit_cloud_github/data/trial_outputs/, and commit the files."
        )
        st.code(str(trial_cache_path(cache_key)), language="text")
        try:
            cache_zip = build_trial_bundle_zip(
                cache_key,
                bundle,
                st.session_state.icao_summary,
                st.session_state.data,
            )
        except Exception as exc:
            st.warning(f"Could not prepare cached trial output ZIP: {exc}")
        else:
            st.download_button(
                "Download cached trial output ZIP",
                data=cache_zip,
                file_name=f"{cache_key}.zip",
                mime="application/zip",
                key="download_trial_cache_zip",
            )
        if st.button("Save current result as cached trial output", key="save_trial_cache"):
            try:
                saved_path = save_trial_bundle(
                    cache_key,
                    bundle,
                    st.session_state.icao_summary,
                    st.session_state.data,
                )
            except Exception as exc:
                st.error(f"Could not save cached trial output: {exc}")
            else:
                _load_trial_bundle_cached.clear()
                st.success(f"Saved cached trial output to {saved_path}")


def _forecast_audit_source(summary: pd.DataFrame, source_column: str) -> str:
    if summary.empty or source_column not in summary.columns:
        return "Unavailable"
    sources = [
        str(value) for value in summary[source_column].dropna().tolist()
        if str(value) and str(value) != "Unavailable"
    ]
    if any(value == "SERENE official forecast" for value in sources):
        return "SERENE official forecast"
    if any(value == "Dashboard-generated trend-based forecast" for value in sources):
        return "Dashboard-generated trend-based forecast"
    if any(value == "Dashboard-generated persistence forecast" for value in sources):
        return "Dashboard-generated persistence forecast"
    return "Unavailable"


def _render_forecast_request_audit(summary: pd.DataFrame) -> None:
    status: LoadStatus = st.session_state.status
    audit_rows = status.metadata.get("forecast_request_audit", [])
    if not audit_rows:
        return
    source_columns = {
        30: "+30 min source",
        90: "+90 min source",
        180: "+3h source",
        360: "+6h source",
    }
    outcome_labels = {
        "available": "Official HDF5 retrieved",
        "not_published": "Not published for this analysis cycle",
        "authentication_failed": "Authentication rejected",
        "network_failed": "Temporary network failure",
        "decode_failed": "Downloaded file could not be interpreted",
    }
    rows = []
    for item in audit_rows:
        period = int(item.get("forecast_parameter", 0))
        rows.append({
            "Selected analysis time": item.get("analysis_time", "N/A"),
            "Forecast valid time": item.get("valid_time", "N/A"),
            "SERENE forecast parameter": period,
            "Latency": item.get("latency", "N/A"),
            "Display role": item.get("display_role", "N/A"),
            "Downloaded from SERENE": bool(item.get("downloaded_from_serene", False)),
            "Outcome": outcome_labels.get(
                item.get("outcome"), item.get("outcome", "Unknown")
            ),
            "Forecast source": (
                _forecast_audit_source(summary, source_columns.get(period, ""))
                if summary is not None and not summary.empty
                else (
                    "SERENE official forecast"
                    if item.get("downloaded_from_serene")
                    else "Unavailable"
                )
            ),
            "Request message": item.get("message", ""),
        })
    with st.expander("Forecast request audit", expanded=False):
        st.dataframe(
            make_streamlit_safe_dataframe(pd.DataFrame(rows)),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "The SERENE API request sends the analysis time as file_time and the "
            "horizon as period (30, 90, 180, or 360 minutes). The forecast valid time "
            "is derived locally as analysis time plus period. When an official spatial "
            "file is unavailable, this 8.13 demonstration version may show a clearly "
            "labelled Dashboard trend or persistence estimate; it must not be read as "
            "an official SERENE forecast."
        )


def _render_global_indices(df: pd.DataFrame) -> None:
    if "variable" not in df.columns:
        return
    global_indices = df[df["variable"].isin(["Kp", "ap"])].copy()
    if global_indices.empty:
        return

    st.subheader("Global geomagnetic context")
    st.caption(
        "Kp and ap are planetary indices. They provide global storm context and are "
        "not assigned to regional map cells."
    )
    columns = st.columns(2)
    for column, variable in zip(columns, ("Kp", "ap")):
        values = pd.to_numeric(
            global_indices.loc[global_indices["variable"] == variable, "value"],
            errors="coerce",
        ).dropna()
        with column:
            st.metric(f"Peak {variable}", f"{values.max():.1f}" if not values.empty else "N/A")
    st.plotly_chart(create_time_series_plot(global_indices))


def _render_main(params: dict) -> None:
    st.title("Aviation Space Weather Dashboard")
    st.caption(
        "SERENE-only ICAO-style research monitoring. Four forecast horizon groups "
        "remain visible; official AIDA files are decoded independently when available."
    )

    _render_cloud_api_hint()

    bundle: IcaoProductBundle = st.session_state.icao_bundle
    if st.session_state.data.empty and bundle.products.empty:
        _render_empty_state()
        st.markdown("---")
        _render_connection_panel(params)
        return

    df = st.session_state.data
    alerts = st.session_state.alerts
    summary = st.session_state.icao_summary

    _render_overall_risk_cards(summary)
    st.markdown("---")
    _render_categorical_risk_map()
    st.markdown("---")
    _render_pecasus_summary_table()
    st.markdown("---")
    if bundle.status.ok:
        _render_research_messages(summary, params)
    else:
        st.info("Research messages require a successful SERENE AIDA analysis state.")
    st.markdown("---")
    if not df.empty:
        _render_raw_value_maps(df)
        st.markdown("---")
        _render_standalone_hf_study(df)
        st.markdown("---")
    _render_global_indices(df)
    st.markdown("---")
    render_validation_section()
    st.markdown("---")
    _render_connection_panel(params)
    st.markdown("---")
    _render_trial_cache_export(params)
    st.markdown("---")
    _render_forecast_request_audit(summary)
    st.markdown("---")
    if not df.empty:
        _render_data_views(df, alerts)


def main() -> None:
    _init_state()
    _apply_pending_time_range()
    _inject_dashboard_css()
    params = _render_sidebar()
    _render_main(params)


if __name__ == "__main__":
    main()
