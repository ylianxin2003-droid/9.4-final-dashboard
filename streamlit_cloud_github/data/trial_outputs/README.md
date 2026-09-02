# Cached Trial Outputs

This folder stores cached research outputs for selected demo / validation
periods in the Aviation Space Weather Dashboard.

These files are used to speed up presentation and validation by loading
processed dashboard outputs from the Git repository instead of repeating every
SERENE download.

Live SERENE API loading is still available for new analysis times and for
regenerating cached outputs locally.

Cached outputs must contain only processed dashboard data such as products,
indices, summary tables, and status metadata. Do not store SERENE API tokens,
Streamlit secrets, raw credentials, or personal data in this folder.

This dashboard is a research prototype and is not for operational aviation use.
