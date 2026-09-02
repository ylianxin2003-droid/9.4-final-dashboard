from __future__ import annotations

import tempfile
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from aida_grid import (
    AidaGridError,
    normalise_aida_variables,
    target_axis,
)


UPSTREAM_AIDA_VERSION = "v0.1.3"


def _official_state_factory() -> Any:
    try:
        import aida
    except ImportError as exc:
        raise AidaGridError(
            "The official breid-phys/aida-ionosphere package is not installed."
        ) from exc
    return aida.AIDAState()


def read_aida_state_time(
    payload: bytes,
    state_factory: Callable[[], Any] | None = None,
) -> pd.Timestamp:
    factory = state_factory or _official_state_factory
    state = factory()
    try:
        with tempfile.NamedTemporaryFile(suffix=".h5") as handle:
            handle.write(payload)
            handle.flush()
            state.readFile(handle.name)
    except Exception as exc:
        raise AidaGridError(
            f"Official AIDA interpreter could not read the raw state time: {exc}"
        ) from exc
    return _normalise_state_time(state.Time)


def calculate_aida_grid(
    payload: bytes,
    region: dict[str, float],
    step: float,
    variables: list[str] | None,
    state_factory: Callable[[], Any] | None = None,
) -> pd.DataFrame:
    target_lats = target_axis(region["lat_min"], region["lat_max"], step)
    target_lons = target_axis(region["lon_min"], region["lon_max"], step)
    if target_lats[0] < -90 or target_lats[-1] > 90:
        raise AidaGridError("Latitude bounds must be within -90 and 90 degrees.")
    if target_lons[0] < -180 or target_lons[-1] > 180:
        raise AidaGridError("Longitude bounds must be within -180 and 180 degrees.")
    selected = normalise_aida_variables(variables)

    factory = state_factory or _official_state_factory
    state = factory()
    try:
        with tempfile.NamedTemporaryFile(suffix=".h5") as handle:
            handle.write(payload)
            handle.flush()
            state.readFile(handle.name)
    except AidaGridError:
        raise
    except Exception as exc:
        raise AidaGridError(
            f"Official AIDA interpreter could not read the raw state: {exc}"
        ) from exc

    try:
        output = state.calc(
            lat=target_lats,
            lon=target_lons,
            grid="3D",
            TEC="TEC" in selected,
            MUF3000="MUF3000F2" in selected,
            collapse_particles=True,
            as_dict=True,
        )
    except Exception as exc:
        raise AidaGridError(f"Official AIDA grid calculation failed: {exc}") from exc

    output_time = _normalise_state_time(state.Time)
    upstream_names = {"MUF3000F2": "MUF3000"}
    frames: list[pd.DataFrame] = []
    expected_shape = (len(target_lons), len(target_lats))
    for variable in selected:
        field = upstream_names.get(variable, variable)
        if field not in output:
            raise AidaGridError(f"Official AIDA output is missing requested field: {field}")
        values = np.asarray(output[field], dtype=float)
        if values.shape != expected_shape:
            raise AidaGridError(
                f"AIDA field {field} has shape {values.shape}; expected {expected_shape}."
            )
        frames.append(
            _field_frame(
                values=values,
                target_lats=target_lats,
                target_lons=target_lons,
                variable=variable,
                output_time=output_time,
            )
        )

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _field_frame(
    *,
    values: np.ndarray,
    target_lats: np.ndarray,
    target_lons: np.ndarray,
    variable: str,
    output_time: object,
) -> pd.DataFrame:
    lat_count = len(target_lats)
    lon_count = len(target_lons)
    return pd.DataFrame({
        "time": output_time,
        "actual_output_time": output_time,
        "lat": np.tile(np.asarray(target_lats, dtype=float), lon_count),
        "lon": np.repeat(np.asarray(target_lons, dtype=float), lat_count),
        "variable": variable,
        "value": np.asarray(values, dtype=float).reshape(-1, order="C"),
        "model": "AIDA",
        "source": (
            "SERENE raw API + breid-phys/aida-ionosphere "
            f"{UPSTREAM_AIDA_VERSION}"
        ),
    })


def _normalise_state_time(value: object) -> pd.Timestamp:
    scalar = np.asarray(value).squeeze()
    if np.asarray(scalar).size != 1:
        raise AidaGridError("Official AIDA state time must be a scalar.")
    try:
        if np.issubdtype(np.asarray(scalar).dtype, np.datetime64):
            parsed = pd.to_datetime(scalar, errors="coerce", utc=True)
        else:
            parsed = pd.to_datetime(float(scalar), unit="s", errors="coerce", utc=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AidaGridError(f"Invalid official AIDA state time: {value}") from exc
    if pd.isna(parsed):
        raise AidaGridError(f"Invalid official AIDA state time: {value}")
    return parsed
