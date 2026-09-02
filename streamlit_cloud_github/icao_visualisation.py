from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from icao_risk import ICAO_COLORS


def create_icao_category_map(cells: pd.DataFrame, title: str) -> go.Figure:
    if cells.empty:
        return _empty_map("No regional ICAO category data are available.", title)
    if "variable" in cells.columns and cells["variable"].isin(["Kp", "ap"]).any():
        return _empty_map(
            "Kp/ap are global indices, not a regional map product.", title
        )

    required = {"lat", "lon", "category", "display_value"}
    if not required.issubset(cells.columns):
        return _empty_map("No regional ICAO category data are available.", title)

    work = cells.copy()
    work["lat"] = pd.to_numeric(work["lat"], errors="coerce")
    work["lon"] = pd.to_numeric(work["lon"], errors="coerce")
    work = work.dropna(subset=["lat", "lon"])
    if work.empty:
        return _empty_map("No regional ICAO category data are available.", title)

    hover_data = {
        column: True
        for column in (
            "time",
            "display_value",
            "unit",
            "category",
            "source",
            "threshold_explanation",
            "product_state",
        )
        if column in work.columns
    }
    fig = px.scatter_geo(
        work,
        lat="lat",
        lon="lon",
        color="category",
        hover_name="indicator" if "indicator" in work.columns else None,
        hover_data=hover_data,
        color_discrete_map=ICAO_COLORS,
        category_orders={
            "category": ["OK", "MODERATE", "SEVERE", "UNAVAILABLE"]
        },
        title=title,
        projection="natural earth",
    )
    fig.update_traces(
        marker={
            "size": 10,
            "opacity": 0.82,
            "line": {"width": 0.7, "color": "rgba(30,30,30,0.55)"},
        }
    )
    fig.update_geos(
        fitbounds="locations",
        showcoastlines=True,
        coastlinecolor="gray",
        showland=True,
        landcolor="rgb(236,238,240)",
        showocean=True,
        oceancolor="rgb(226,238,248)",
    )
    fig.update_layout(
        template="plotly_white",
        height=560,
        legend_title_text="ICAO-style category",
        margin={"l": 0, "r": 0, "t": 55, "b": 0},
    )
    return fig


def _empty_map(message: str, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
    )
    fig.update_layout(template="plotly_white", height=560, title=title)
    return fig
