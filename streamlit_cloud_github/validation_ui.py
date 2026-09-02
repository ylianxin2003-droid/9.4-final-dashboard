from __future__ import annotations

import streamlit as st


def render_validation_section() -> None:
    st.subheader("Validation and engineering assumptions")
    with st.expander("Validation checklist for the engineering decision-support prototype", expanded=False):
        st.markdown(
            """
            The validation work is organised around the project risks that matter
            for a decision-support prototype:

            - Historical event replay: use cached trial outputs or the Live SERENE
              API source to replay selected storm-like analysis windows.
            - Quiet vs storm comparison: prefer AIDA `reference_value` from the
              30-day same-UTC MUF3000F2 baseline when it is available.
            - PSD sensitivity: use the fallback PSD slider only when a historical
              comparison is unavailable, and label the result as an assumption.
            - Frequency sensitivity: compare 5, 7.5, 10, 12.5, 15, 17.5 and
              20 MHz as research cases, not operational frequency advice.
            - Route assessment verification: check quiet route availability,
              storm route availability, route coverage reduction, degraded route
              points and longest degraded route segment for the UK to North
              Atlantic to New York JFK case study.

            Main assumptions and limitations:

            - HF Communication Coverage is a MUF-threshold engineering proxy, not
              Trace ray tracing.
            - The dashboard depends on SERENE/AIDA inputs and available archive
              states.
            - Kp/ap are global context only and are not regional map cells.
            - This is a research prototype only, not operational aviation
              guidance.
            """
        )
