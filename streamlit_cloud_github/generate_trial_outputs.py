from __future__ import annotations

import argparse
import sys

import pandas as pd

from app_utils import historical_risk_windows
from data_loader import IcaoProductBundle, load_icao_products
from icao_risk import build_icao_summary
from trial_cache import make_trial_cache_key, save_trial_bundle


GLOBAL_REGION = {
    "lat_min": -90.0,
    "lat_max": 90.0,
    "lon_min": -180.0,
    "lon_max": 180.0,
}
FULL_ICAO_MODE = "Full ICAO-style mode"


def _analysis_times() -> list[str]:
    times: list[str] = []
    for value in historical_risk_windows()["Select range"]:
        parts = str(value).split(" to ")
        if len(parts) == 2:
            times.append(parts[1])
    return times


def _display_data(bundle: IcaoProductBundle) -> pd.DataFrame:
    frames = [
        frame for frame in (bundle.products, bundle.indices)
        if frame is not None and not frame.empty
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def generate_trial_outputs(
    grid_step: float = 15.0,
    stop_on_error: bool = False,
) -> int:
    saved = 0
    for analysis_time in _analysis_times():
        cache_key = make_trial_cache_key(
            analysis_time, GLOBAL_REGION, grid_step, FULL_ICAO_MODE
        )
        print(f"Generating {cache_key} ...", flush=True)
        bundle = load_icao_products(
            analysis_time=analysis_time,
            variables=["TEC", "MUF3000F2"],
            region=GLOBAL_REGION,
            grid_step=grid_step,
            include_three_hour_window=True,
            include_psd_baseline=True,
        )
        if not bundle.status.ok or bundle.products.empty:
            print(f"  skipped: {bundle.status.message}", file=sys.stderr)
            for warning in bundle.status.warnings[:5]:
                print(f"  warning: {warning}", file=sys.stderr)
            if stop_on_error:
                raise RuntimeError(bundle.status.message)
            continue
        summary = build_icao_summary(
            bundle.products,
            bundle.indices,
            eligible=bundle.kp_storm_eligible,
            kp_horizons=bundle.kp_horizons,
        )
        output_path = save_trial_bundle(
            cache_key,
            bundle,
            summary,
            _display_data(bundle),
        )
        print(f"  saved: {output_path}", flush=True)
        saved += 1
    return saved


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate processed cached trial outputs. Run locally with a valid "
            "SERENE API token, then commit streamlit_cloud_github/data/trial_outputs/."
        )
    )
    parser.add_argument("--grid-step", type=float, default=15.0)
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()
    saved = generate_trial_outputs(
        grid_step=args.grid_step,
        stop_on_error=args.stop_on_error,
    )
    print(f"Saved {saved} cached trial output folder(s).")
    return 0 if saved else 1


if __name__ == "__main__":
    raise SystemExit(main())
