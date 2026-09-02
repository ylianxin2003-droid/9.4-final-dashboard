from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots



RISK_COLORS: dict[str, str] = {
    "Normal": "#2ecc71",
    "Watch": "#f1c40f",
    "Warning": "#e67e22",
    "Severe": "#e74c3c",
    "G1 Minor": "#f1c40f",
    "G2 Moderate": "#f39c12",
    "G3 Strong": "#e67e22",
    "G4 Severe": "#e74c3c",
    "G5 Extreme": "#8e0000",
}

RISK_DISPLAY_ORDER = [
    "Normal", "Watch", "G1 Minor", "G2 Moderate", "Warning",
    "G3 Strong", "Severe", "G4 Severe", "G5 Extreme",
]

ALERT_TYPE_COLORS: dict[str, str] = {
    "GNSS positioning risk": "#3498db",
    "HF communication risk": "#9b59b6",
    "General ionospheric disturbance": "#e74c3c",
}




def create_time_series_plot(
    df: pd.DataFrame,
    variable: str | None = None,
    title: str | None = None,
) -> go.Figure:
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for time-series plot.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )
        return fig

    work = df.copy()
    if variable:
        work = work[work["variable"] == variable]

    if work.empty:
        fig = go.Figure()
        fig.add_annotation(
            text=f"No data for variable '{variable}'.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )
        return fig

    if "time" in work.columns:
        work["time"] = pd.to_datetime(work["time"], errors="coerce")
    else:
        work["time"] = pd.NaT
    if work["time"].notna().any():
        x_col = "time"
        x_label = "Time"
    else:
        work = work.reset_index(drop=True)
        work["sample"] = work.index + 1
        x_col = "sample"
        x_label = "API sample"

    grouped = work.groupby([x_col, "variable"], as_index=False)["value"].mean()

    if grouped["variable"].nunique() <= 1:
        fig = px.line(
            grouped, x=x_col, y="value", color="variable",
            title=title or "Ionospheric parameter over time",
            labels={"value": "Value", x_col: x_label, "variable": "Variable"},
            markers=True,
        )
    else:
        fig = make_subplots(
            rows=grouped["variable"].nunique(),
            cols=1,
            shared_xaxes=True,
            subplot_titles=list(grouped["variable"].unique()),
        )
        for i, var in enumerate(grouped["variable"].unique()):
            sub = grouped[grouped["variable"] == var]
            fig.add_trace(
                go.Scatter(x=sub[x_col], y=sub["value"], mode="lines+markers", name=var),
                row=i + 1, col=1,
            )
        fig.update_xaxes(title_text=x_label, row=grouped["variable"].nunique(), col=1)
        fig.update_layout(
            title_text=title or "Ionospheric parameters over time",
            height=250 * grouped["variable"].nunique(),
        )

    fig.update_layout(template="plotly_white", hovermode="x unified")
    return fig



def create_map_plot(
    df: pd.DataFrame,
    variable: str | None = None,
    title: str | None = None,
) -> go.Figure:
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for map plot.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )
        return fig

    work = df.copy()
    if variable:
        work = work[work["variable"] == variable]

    if "lat" not in work.columns or "lon" not in work.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="Data does not contain lat/lon columns for map display.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )
        return fig

    for col in ("lat", "lon", "value"):
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work.dropna(subset=["lat", "lon", "value"])
    if work.empty:
        fig = go.Figure()
        fig.add_annotation(
            text=f"No mappable lat/lon data for {variable or 'selected data'}.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )
        return fig

    if len(work) > 3000:
        work = work.sample(n=3000, random_state=42)

    if "time" in work.columns:
        work["time"] = pd.to_datetime(work["time"], errors="coerce")
        valid_times = work["time"].dropna()
        if not valid_times.empty:
            work = work[work["time"] == valid_times.max()]

    fig = px.scatter_geo(
        work,
        lat="lat",
        lon="lon",
        color="value",
        size="value",
        hover_name="variable" if "variable" in work.columns else None,
        hover_data=["value", "variable"] if "variable" in work.columns else ["value"],
        title=title or f"Global {variable or 'ionospheric'} map",
        color_continuous_scale="Plasma",
        projection="natural earth",
    )
    fig.update_geos(
        showcoastlines=True,
        coastlinecolor="gray",
        showland=True,
        landcolor="lightgray",
        showocean=True,
        oceancolor="aliceblue",
    )
    fig.update_layout(template="plotly_white", height=500)
    return fig




def create_alert_timeline(alerts: pd.DataFrame) -> go.Figure:
    if alerts.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No alerts to display — all parameters within normal range.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )
        return fig

    needed_cols = {"timestamp", "alert_type", "risk_level"}
    if not needed_cols.issubset(alerts.columns):
        fig = go.Figure()
        fig.add_annotation(
            text="Alert data missing required columns for timeline.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )
        return fig

    work = alerts.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce", utc=True)
    work = work.sort_values("timestamp")

    alert_types = work["alert_type"].unique()
    y_map = {t: i for i, t in enumerate(alert_types)}

    fig = go.Figure()
    for _, row in work.iterrows():
        risk = row.get("risk_level", "Normal")
        fig.add_trace(go.Scatter(
            x=[row["timestamp"]],
            y=[y_map.get(row.get("alert_type", "Unknown"), 0)],
            mode="markers",
            marker=dict(
                size=14,
                color=RISK_COLORS.get(risk, "#95a5a6"),
                symbol="diamond",
                line=dict(width=1, color="black"),
            ),
            name=f"{row.get('alert_type', '?')} — {risk}",
            text=f"{row.get('region', '?')}<br>{row.get('reason', '')}",
            hoverinfo="text+name",
        ))

    fig.update_yaxes(
        tickvals=list(y_map.values()),
        ticktext=list(y_map.keys()),
    )
    fig.update_layout(
        title="ICAO-style prototype alert timeline",
        xaxis_title="Time",
        yaxis_title="Alert type",
        template="plotly_white",
        height=300 + 60 * len(alert_types),
        showlegend=False,
    )
    valid_times = work["timestamp"].dropna()
    if not valid_times.empty:
        start = valid_times.min()
        end = valid_times.max()
        if end - start < pd.Timedelta(hours=1):
            centre = start + (end - start) / 2
            start = centre - pd.Timedelta(minutes=30)
            end = centre + pd.Timedelta(minutes=30)
        fig.update_xaxes(
            range=[start, end],
            tickformat="%Y-%m-%d<br>%H:%M UTC",
        )
    return fig




def create_alert_summary(alerts: pd.DataFrame) -> go.Figure:
    if alerts.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No alerts — all parameters within normal range.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )
        return fig

    if "alert_type" not in alerts.columns or "risk_level" not in alerts.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="Alert data missing required columns.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )
        return fig

    counts = alerts.groupby(["alert_type", "risk_level"]).size().reset_index(name="count")

    risk_order = RISK_DISPLAY_ORDER
    counts["risk_level"] = pd.Categorical(
        counts["risk_level"], categories=risk_order, ordered=True
    )
    counts = counts.sort_values(["alert_type", "risk_level"])

    fig = px.bar(
        counts,
        x="alert_type",
        y="count",
        color="risk_level",
        color_discrete_map=RISK_COLORS,
        title="Alert summary by type and risk level",
        labels={"count": "Number of advisories", "alert_type": "Alert type", "risk_level": "Risk level"},
        barmode="stack",
        category_orders={"risk_level": risk_order},
    )
    fig.update_layout(template="plotly_white", height=400)
    return fig
