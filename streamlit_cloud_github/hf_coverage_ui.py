from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from data_loader import LoadStatus
from hf_coverage import (
    DEFAULT_SWEEP_FREQUENCIES,
    build_frequency_sweep,
    build_hf_engineering_case,
    create_hf_coverage_map,
    create_hf_route_profile_plot,
)
from hf_locations import (
    DEFAULT_HF_ROUTE_SCENARIO,
    HF_ROUTE_SCENARIOS,
    location_names,
    resolve_location,
    resolve_route_scenario,
)


def _source_label(status: LoadStatus) -> str:
    return {
        "api": "Live SERENE API",
        "trial_cache": "Cached trial output",
        "indices": "SERENE global indices only",
        "none": "No data",
    }.get(status.source, status.source)


def _has_positive_aida_reference(df: pd.DataFrame) -> bool:
    if "reference_value" not in df.columns:
        return False
    reference = pd.to_numeric(df["reference_value"], errors="coerce")
    return bool(reference.gt(0).any())


def _add_route_direction_arrows(figure: object, route: pd.DataFrame | None) -> object:
    if route is None or not isinstance(route, pd.DataFrame) or route.empty:
        return figure
    if not {"lat", "lon"}.issubset(route.columns):
        return figure

    work = route[["lat", "lon"]].copy()
    work["lat"] = pd.to_numeric(work["lat"], errors="coerce")
    work["lon"] = pd.to_numeric(work["lon"], errors="coerce")
    work = work.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    if len(work) < 8:
        return figure

    arrow_indices = sorted({
        len(work) // 4,
        len(work) // 2,
        3 * len(work) // 4,
    })
    arrow_lats: list[float] = []
    arrow_lons: list[float] = []
    arrow_symbols: list[str] = []
    arrow_angles: list[float] = []

    for idx in arrow_indices:
        if idx <= 0 or idx >= len(work) - 1:
            continue
        current = work.iloc[idx]
        next_point = work.iloc[idx + 1]
        lat1 = math.radians(float(current["lat"]))
        lon1 = math.radians(float(current["lon"]))
        lat2 = math.radians(float(next_point["lat"]))
        dlon = math.radians(
            (
                float(next_point["lon"])
                - float(current["lon"])
                + 180.0
            )
            % 360.0
            - 180.0
        )
        if math.isclose(lat1, lat2, abs_tol=1e-12) and math.isclose(
            dlon, 0.0, abs_tol=1e-12
        ):
            continue
        y = math.sin(dlon) * math.cos(lat2)
        x = (
            math.cos(lat1) * math.sin(lat2)
            - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        )
        angle = math.degrees(math.atan2(y, x))

        midpoint_x = math.cos(lat1) + math.cos(lat2) * math.cos(dlon)
        midpoint_y = math.cos(lat2) * math.sin(dlon)
        midpoint_z = math.sin(lat1) + math.sin(lat2)
        midpoint_norm = math.sqrt(
            midpoint_x * midpoint_x
            + midpoint_y * midpoint_y
            + midpoint_z * midpoint_z
        )
        if math.isclose(midpoint_norm, 0.0, abs_tol=1e-12):
            continue
        midpoint_lat = math.degrees(
            math.atan2(midpoint_z, math.hypot(midpoint_x, midpoint_y))
        )
        midpoint_lon = (
            math.degrees(lon1 + math.atan2(midpoint_y, midpoint_x))
            + 540.0
        ) % 360.0 - 180.0

        arrow_lats.append(midpoint_lat)
        arrow_lons.append(midpoint_lon)
        arrow_symbols.append("triangle-up")
        arrow_angles.append(angle)

    if not arrow_lats:
        return figure

    import plotly.graph_objects as go

    figure.add_trace(
        go.Scattergeo(
            lat=arrow_lats,
            lon=arrow_lons,
            mode="markers",
            name="Route direction",
            marker={
                "size": 11,
                "color": "#0D47A1",
                "symbol": arrow_symbols,
                "angle": arrow_angles,
                "angleref": "up",
            },
            showlegend=False,
            hoverinfo="skip",
        )
    )
    return figure


def _render_plotly_figure(
    figure: object,
    route: pd.DataFrame | None = None,
) -> None:
    if route is not None:
        figure = _add_route_direction_arrows(figure, route)
    st.plotly_chart(figure, width="stretch")


def render_hf_propagation_case_study(df: pd.DataFrame) -> None:
    has_aida_reference = _has_positive_aida_reference(df)
    st.subheader("Engineering Impact: HF Communication Coverage")
    st.caption(
        "Phase 1 uses a simplified MUF-based coverage proxy to connect PSD/MUF "
        "changes to possible HF communication impact. Phase 2 is an experimental "
        "Trace HF ray-tracing integration path; no ray paths are fabricated here. "
        "This is not an operational HF communication coverage product."
    )

    mode_col, source_col = st.columns([1.3, 1])
    with mode_col:
        st.info(
            "Phase 1: MUF-based coverage proxy / MUF-threshold demonstration. "
            "Phase 2: experimental Trace ray-tracing integration status is documented below."
        )
    with source_col:
        status: LoadStatus = st.session_state.status
        st.metric("Data source", _source_label(status))
        if has_aida_reference:
            st.caption(
                "30-day AIDA reference is active; the assumption slider is ignored."
            )
        else:
            st.caption(
                "Assumed PSD demonstration is active because no positive AIDA "
                "reference is available."
            )
    st.info(
        "Engineering workflow: Input = transmitter, target route and frequency; "
        "Processing = quiet/background MUF compared with storm MUF; "
        "Engineering meaning = PSD lowers MUF and reduces usable HF coverage; "
        "Decision support = route status and model-based frequency comparison."
    )

    control_col, psd_col = st.columns(2)
    with control_col:
        frequency_mhz = st.slider(
            "HF frequency for coverage demo (MHz)",
            3.0,
            30.0,
            10.0,
            0.5,
            key="hf_case_frequency_mhz",
        )
    with psd_col:
        psd_percent = st.slider(
            (
                "Assumed PSD demonstration (%) — inactive; AIDA reference available"
                if has_aida_reference else
                "Assumed PSD demonstration (%)"
            ),
            0.0,
            70.0,
            30.0,
            5.0,
            key="hf_case_psd_percent",
            help=(
                "This is a user-controlled engineering assumption for showing "
                "the communication impact of Post-Storm Depression."
            ),
        )
    st.markdown("**Illustrative communication route**")
    st.caption(
        "The selected locations are assumed geographic communication endpoints. "
        "They are not confirmed HF transmitter sites, airport departure/arrival "
        "points, or validated aircraft trajectories."
    )
    route_mode = st.selectbox(
        "Route setup",
        ["Preset scenario", "Custom city-to-city", "Advanced coordinates"],
        key="hf_route_mode",
    )

    if route_mode == "Preset scenario":
        scenario = st.selectbox(
            "Representative route scenario",
            list(HF_ROUTE_SCENARIOS),
            index=list(HF_ROUTE_SCENARIOS).index(DEFAULT_HF_ROUTE_SCENARIO),
            key="hf_route_scenario",
        )
        transmitter, target = resolve_route_scenario(scenario)
    elif route_mode == "Custom city-to-city":
        origin_col, target_col = st.columns(2)
        names = location_names()
        with origin_col:
            origin_name = st.selectbox(
                "Origin city or region",
                names,
                index=names.index("Birmingham, United Kingdom"),
                key="hf_origin_location",
            )
        with target_col:
            target_name = st.selectbox(
                "Target city or region",
                names,
                index=names.index("New York, United States"),
                key="hf_target_location",
            )
        transmitter = resolve_location(origin_name)
        target = resolve_location(target_name)
    else:
        name_col1, name_col2 = st.columns(2)
        with name_col1:
            origin_name = st.text_input(
                "Origin label", "Custom origin", key="hf_custom_tx_name"
            )
        with name_col2:
            target_name = st.text_input(
                "Target label", "Custom target", key="hf_custom_target_name"
            )
        tx_lat_col, tx_lon_col, target_lat_col, target_lon_col = st.columns(4)
        with tx_lat_col:
            tx_lat = st.number_input(
                "Origin latitude", -90.0, 90.0, 52.4862, 0.1,
                key="hf_custom_tx_lat",
            )
        with tx_lon_col:
            tx_lon = st.number_input(
                "Origin longitude", -180.0, 180.0, -1.8904, 0.1,
                key="hf_custom_tx_lon",
            )
        with target_lat_col:
            target_lat = st.number_input(
                "Target latitude", -90.0, 90.0, 40.7128, 0.1,
                key="hf_custom_target_lat",
            )
        with target_lon_col:
            target_lon = st.number_input(
                "Target longitude", -180.0, 180.0, -74.0060, 0.1,
                key="hf_custom_target_lon",
            )
        transmitter = {"name": origin_name, "lat": tx_lat, "lon": tx_lon}
        target = {"name": target_name, "lat": target_lat, "lon": target_lon}

    st.caption(
        f'Resolved route: {transmitter["name"]} '
        f'({transmitter["lat"]:.4f}, {transmitter["lon"]:.4f}) → '
        f'{target["name"]} ({target["lat"]:.4f}, {target["lon"]:.4f})'
    )

    time_cols = st.columns(3)
    status = st.session_state.status
    with time_cols[0]:
        st.text_input(
            "Quiet/background analysis time",
            value=(
                "AIDA 30-day same-UTC reference"
                if has_aida_reference else
                "Assumed PSD demonstration"
            ),
            disabled=True,
            key="hf_quiet_time_display",
        )
    with time_cols[1]:
        st.text_input(
            "Storm analysis time",
            value=str(status.metadata.get("analysis_time", "loaded analysis time")),
            disabled=True,
            key="hf_storm_time_display",
        )
    with time_cols[2]:
        st.text_input(
            "Coverage data mode",
            value=_source_label(status),
            disabled=True,
            key="hf_data_mode_display",
        )

    engineering_case = build_hf_engineering_case(
        df,
        frequency_mhz=frequency_mhz,
        transmitter=transmitter,
        target=target,
        route_samples=33,
        assumed_psd_percent=psd_percent,
    )
    if engineering_case.grid.empty:
        st.info("No spatial MUF3000F2 grid is available for the HF communication case study.")
        return

    summary = engineering_case.summary
    metric_cols = st.columns(4)
    metric_cols[0].metric("Selected frequency", f"{summary['frequency_mhz']:.1f} MHz")
    metric_cols[1].metric("Quiet coverage", f"{summary['quiet_usable_grid_pct']:.0f}%")
    metric_cols[2].metric("Storm coverage", f"{summary['storm_usable_grid_pct']:.0f}%")
    metric_cols[3].metric("Coverage loss", f"{summary['regional_coverage_loss_pct_points']:.0f} pp")
    route_cols = st.columns(6)
    route_cols[0].metric("Quiet route availability", f"{summary['quiet_route_available_pct']:.0f}%")
    route_cols[1].metric("Storm route availability", f"{summary['storm_route_available_pct']:.0f}%")
    route_cols[2].metric("Route coverage reduction", f"{summary['route_coverage_loss_pct_points']:.0f} pp")
    route_cols[3].metric("Degraded route", f"{summary['degraded_route_pct']:.0f}%")
    route_cols[4].metric("Unavailable route", f"{summary['storm_route_unavailable_pct']:.0f}%")
    route_cols[5].metric("Longest degraded segment", f"{summary['longest_degraded_segment_km']:.0f} km")
    st.caption(
        f"Propagation model: {summary['propagation_model']} | "
        f"Comparison mode: {summary['comparison_mode']}"
    )

    st.markdown("**Engineering interpretation**")
    st.info(summary["interpretation"])
    st.markdown("**Route decision support**")
    st.warning(summary["route_recommendation"])

    quiet_tab, storm_tab, change_tab, profile_tab, route_tab, sweep_tab = st.tabs([
        "Quiet map",
        "Storm map",
        "Coverage change",
        "Route profile",
        "Route samples",
        "Frequency sweep",
    ])
    with quiet_tab:
        _render_plotly_figure(
            create_hf_coverage_map(
                engineering_case.grid,
                transmitter,
                target,
                route=engineering_case.route.to_dict("records"),
                title=f"Quiet/background potential HF coverage at {summary['frequency_mhz']:.1f} MHz",
                map_mode="quiet",
            ),
            route=engineering_case.route,
        )
    with storm_tab:
        _render_plotly_figure(
            create_hf_coverage_map(
                engineering_case.grid,
                transmitter,
                target,
                route=engineering_case.route.to_dict("records"),
                title=f"Storm-time potential HF coverage at {summary['frequency_mhz']:.1f} MHz",
                map_mode="storm",
            ),
            route=engineering_case.route,
        )
    with change_tab:
        _render_plotly_figure(
            create_hf_coverage_map(
                engineering_case.grid,
                transmitter,
                target,
                route=engineering_case.route.to_dict("records"),
                title=f"Coverage change at {summary['frequency_mhz']:.1f} MHz",
                map_mode="change",
            ),
            route=engineering_case.route,
        )
    with profile_tab:
        st.caption(
            "Validation profile: this view shows how "
            "quiet/background MUF compares with storm MUF along the same route. "
            "Where the storm MUF falls below the selected frequency, the route "
            "sample is treated as degraded in the MUF-threshold approximation."
        )
        _render_plotly_figure(
            create_hf_route_profile_plot(
                engineering_case.route,
                summary["frequency_mhz"],
            ),
        )
    with route_tab:
        st.dataframe(engineering_case.route, width="stretch", hide_index=True)
    with sweep_tab:
        sweep = build_frequency_sweep(engineering_case, DEFAULT_SWEEP_FREQUENCIES)
        st.caption(
            "Research comparison only. The highlighted frequency is a model-based "
            "storm-case recommendation inside the MUF-threshold approximation, "
            "not operational frequency advice."
        )
        st.dataframe(sweep, width="stretch", hide_index=True)
        best = sweep[sweep["model_recommended_for_storm_case"]]
        if not best.empty:
            row = best.iloc[0]
            st.info(
                f"Within this MUF-threshold approximation, {row['frequency_mhz']:.1f} MHz "
                "is the model-preferred storm frequency for this route because it "
                "has the strongest storm-case route availability in the comparison. "
                "This is decision support for the research prototype, not operational "
                "frequency advice."
            )
