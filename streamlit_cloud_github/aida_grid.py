from __future__ import annotations

import numpy as np


AIDA_VARIABLES = ("TEC", "foF2", "MUF3000F2", "NmF2", "hmF2")
_VARIABLE_ALIASES = {"vTEC": "TEC", "MUF3000": "MUF3000F2"}


class AidaGridError(ValueError):
    pass


def target_axis(lower: float, upper: float, step: float) -> np.ndarray:
    if not np.isfinite(step) or step <= 0:
        raise AidaGridError("Grid step must be positive.")
    lo, hi = sorted((float(lower), float(upper)))
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise AidaGridError("Grid bounds must be finite.")
    count = int(np.floor((hi - lo) / float(step) + 1e-9)) + 1
    return lo + np.arange(count, dtype=float) * float(step)


def estimate_target_points(region: dict[str, float], step: float) -> int:
    latitudes = target_axis(region["lat_min"], region["lat_max"], step)
    longitudes = target_axis(region["lon_min"], region["lon_max"], step)
    return int(len(latitudes) * len(longitudes))


def normalise_aida_variables(variables: list[str] | None) -> list[str]:
    requested = variables or list(AIDA_VARIABLES)
    selected = list(dict.fromkeys(_VARIABLE_ALIASES.get(name, name) for name in requested))
    unknown = [name for name in selected if name not in AIDA_VARIABLES]
    if unknown:
        raise AidaGridError(f"Unsupported AIDA variable(s): {', '.join(unknown)}")
    return selected
