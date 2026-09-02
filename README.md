# Aviation Space Weather Dashboard

A Streamlit application for exploring aviation-relevant ionospheric conditions
from SERENE AIDA products and GFZ geomagnetic-index data.

The dashboard combines spatial model output, transparent risk rules, forecast
provenance and an HF communication study in one reproducible research tool. It
is intended for technical analysis and education. It is not an operational
aviation warning service.

## Live application

The deployed dashboard is available at:
https://sep-4th-final-dashboard.streamlit.app/

## Features

- Live loading of authenticated SERENE AIDA analysis and forecast products
- Four cached Full ICAO-style demonstration cases that run without an API token
- Vertical TEC maps and GNSS risk categories
- MUF3000F2-based post-storm-depression analysis and HF communication risk
- GFZ Kp/ap history and GFZ PAGER/SWIFT Kp forecast context
- Four forecast horizons: +30 minutes, +90 minutes, +3 hours and +6 hours
- Explicit separation of official forecasts, local estimates and unavailable data
- Summary tables, categorical maps, CSV export and TEST research messages
- Configurable HF routes using presets, named locations or coordinates
- Quiet/storm coverage comparison, route metrics and frequency sensitivity

## Data sources and provenance

SERENE supplies AIDA ionospheric analysis and spatial forecast files. The app
uses the versioned
[`aida-ionosphere`](https://github.com/breid-phys/aida-ionosphere)
interpreter to calculate local grids from each downloaded HDF5 state.

GFZ supplies global Kp/ap observations and the current PAGER/SWIFT ensemble Kp
forecast. Kp and ap are planetary indices and are not plotted as regional map
cells.

Every forecast value carries a source label:

- **OFFICIAL SERENE API**: decoded from the matching SERENE forecast file.
- **DASHBOARD ESTIMATE — trend extrapolation**: calculated locally from recent
  SERENE analysis states when the official spatial forecast is absent.
- **DASHBOARD ESTIMATE — persistence**: holds the latest analysis value when a
  trend cannot be calculated.
- **GFZ observed outcome — backtesting only**: an observed historical Kp target,
  not an archived forecast.
- **UNAVAILABLE**: the required evidence was not returned or failed validation.

Local estimates are presented as experimental scenarios. They are not official
SERENE products and have not been validated as independent scientific
forecasts.

## Repository layout

```text
.
├── README.md
└── streamlit_cloud_github/
    ├── app.py
    ├── *_loader.py, *_risk.py, *_coverage.py, ...
    ├── requirements.txt
    ├── .env.example
    └── data/trial_outputs/
```

The Streamlit entrypoint is `streamlit_cloud_github/app.py`.

## Quick start

### Requirements

- Python 3.11
- Git
- Network access for dependency installation

From the repository root, create an isolated environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r streamlit_cloud_github/requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Start the application:

```bash
streamlit run streamlit_cloud_github/app.py
```

The app opens in **Cached trial output** mode. Select a cached date and click
**Load / Refresh data** to explore the dashboard without a SERENE token.

## Optional live SERENE access

Live AIDA downloads require a valid SERENE API token. Copy the example file and
replace only the local value:

```bash
cp streamlit_cloud_github/.env.example streamlit_cloud_github/.env
```

```dotenv
SERENE_API_BASE_URL=https://spaceweather.bham.ac.uk
SERENE_API_TOKEN=your_serene_api_token_here
SERENE_API_TIMEOUT=30
SERENE_AUTH_SCHEME=Token
SERENE_AIDA_ARCHIVE_START=2024-09-28T00:00:00Z
```

The real `.env` file is ignored by Git. Do not place tokens in source files,
issues, screenshots or commits.

In the dashboard, select **Live SERENE API**, use **Test SERENE API connection**,
choose an analysis time, and then click **Load / Refresh data**. Forecast
availability is cycle-dependent; a missing horizon remains visible as
`UNAVAILABLE` or as a clearly labelled Dashboard estimate.

## Cached demonstration data

Version-controlled cached outputs are stored in
`streamlit_cloud_github/data/trial_outputs/`. They contain processed products,
indices, summaries and source metadata only. They do not contain API tokens or
Streamlit secrets.

With a valid local token, additional caches can be generated using:

```bash
python streamlit_cloud_github/generate_trial_outputs.py
```

The generator and dashboard always use the full ICAO-style processing path,
including the preceding three-hour observation window and the 30-day
same-UTC MUF baseline used for PSD evidence.

Review generated files before committing them. Streamlit Cloud runtime storage
is temporary.

## Deploy to Streamlit Community Cloud

1. Fork or copy this repository to GitHub.
2. In Streamlit Community Cloud, create an app from that repository.
3. Select Python **3.11**.
4. Set the main file to `streamlit_cloud_github/app.py`.
5. Deploy. Cached demonstration mode works without private configuration.
6. For live SERENE access, add the following names in **Advanced settings →
   Secrets** and insert your own private token:

```toml
SERENE_API_BASE_URL = "https://spaceweather.bham.ac.uk"
SERENE_API_TOKEN = "your_private_token"
SERENE_API_TIMEOUT = "30"
SERENE_AUTH_SCHEME = "Token"
SERENE_AIDA_ARCHIVE_START = "2024-09-28T00:00:00Z"
```

After deployment, verify the API connection, load one cached case and one small
live region, confirm all four horizon groups and download a CSV to check the
source labels. Streamlit Community Cloud may need the app to be recreated if an
existing deployment uses a different Python version.

## Risk rules

The application applies deterministic, inspectable rules:

- Vertical TEC: `OK` below 125 TECU, `MODERATE` from 125 to below 175 TECU,
  and `SEVERE` at 175 TECU or above.
- Kp auroral-absorption proxy: `MODERATE` at Kp 8 and `SEVERE` at Kp 9.
- Post-Storm Depression: `MODERATE` from 30% to below 50% and `SEVERE` at
  50% or above, subject to the required Kp storm-history gate.
- Missing required evidence remains `UNAVAILABLE`; it is never converted to OK.

Risk severity and data completeness are shown separately. These categories are
research interpretations of published aviation space-weather thresholds, not
official advisories.

## HF communication study

The standalone HF section turns the loaded MUF3000F2 grid into an inspectable
route-level engineering example:

```text
Risk Assessment
  → Communication Impact
  → Engineering Interpretation
  → Decision Support
```

For a selected frequency, a grid cell or route sample is treated as potentially
usable when its local MUF is at least that frequency. Quiet/background and
disturbed states are compared to calculate regional coverage, route
availability, degraded samples and the longest degraded segment.

Named locations are assumed geographic endpoints. They are not verified HF
stations, airport pairs or aircraft tracks. The current implementation is a
MUF-threshold engineering proxy and does not perform physical ray tracing or
recommend operational frequencies.

## Limitations

- Research prototype; not for operational aviation decisions
- Not an official ICAO, SERENE or GFZ advisory product
- Local trend and persistence estimates are not validated forecast models
- No direct radiation-dose, S4/sigma-phi scintillation, PCA or SWF input
- HF coverage is a MUF-threshold proxy, not a propagation solver
- Live results depend on upstream publication, credentials and network access
- Threshold implementation and software tests do not establish operational
  accuracy, certification or forecast skill

## References

- B. Reid et al., “The Real-Time Advanced Ionospheric Data Assimilation (AIDA)
  Model,” *Space Weather*, 24(2), e2025SW004712, 2026.
  https://doi.org/10.1029/2025SW004712
- J. Matzka et al., “The geomagnetic Kp index and derived indices of
  geomagnetic activity,” *Space Weather*, 2021.
  https://doi.org/10.1029/2020SW002641
- GFZ Kp dataset: https://doi.org/10.5880/Kp.0001
- SERENE products: https://serene.bham.ac.uk/output/
