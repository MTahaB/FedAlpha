from __future__ import annotations

import pandas as pd
import streamlit as st

from verification.checks import collect_status, status_counts


st.set_page_config(page_title="FedAlpha v2", layout="wide")
st.title("FedAlpha v2")

def render_verification() -> None:
    if "verify_pytest" not in st.session_state:
        st.session_state.verify_pytest = False
    if "verify_docker" not in st.session_state:
        st.session_state.verify_docker = False
    if "verify_hardhat" not in st.session_state:
        st.session_state.verify_hardhat = False

    action_cols = st.columns([1, 1, 1, 1, 4])
    if action_cols[0].button("Refresh"):
        st.session_state.verify_pytest = False
        st.session_state.verify_docker = False
        st.session_state.verify_hardhat = False
    if action_cols[1].button("Run pytest"):
        st.session_state.verify_pytest = True
    if action_cols[2].button("Probe Docker"):
        st.session_state.verify_docker = True
    if action_cols[3].button("Probe Hardhat"):
        st.session_state.verify_hardhat = True

    results = collect_status(
        run_pytest=st.session_state.verify_pytest,
        run_docker=st.session_state.verify_docker,
        run_hardhat=st.session_state.verify_hardhat,
    )
    counts = status_counts(results)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Passed", counts.get("pass", 0))
    metric_cols[1].metric("Blocked", counts.get("blocked", 0))
    metric_cols[2].metric("Failed", counts.get("fail", 0))
    metric_cols[3].metric("Total", len(results))

    frame = pd.DataFrame([result.to_dict() for result in results])
    area = st.segmented_control("Area", ["all", *sorted(frame["area"].unique())], default="all")
    visible = frame if area == "all" else frame[frame["area"] == area]
    st.dataframe(
        visible[["status", "area", "name", "detail"]],
        width="stretch",
        hide_index=True,
    )


col1, col2, col3, col4 = st.columns(4)
col1.metric("Round", "0")
col2.metric("Model Sharpe", "pending")
col3.metric("Oracle verdict", "pending")
col4.metric("Privacy epsilon", "pending")

tabs = st.tabs(["Verification", "Training", "Backtest", "Privacy", "Oracle", "Governance"])

with tabs[0]:
    render_verification()

with tabs[1]:
    st.subheader("Flower convergence")
    st.line_chart(pd.DataFrame({"loss": []}))

with tabs[2]:
    st.subheader("Backtest")
    st.line_chart(pd.DataFrame({"FL": [], "S&P500": [], "Equal Weight": []}))

with tabs[3]:
    st.subheader("Privacy-performance")
    st.line_chart(pd.DataFrame({"epsilon": [], "sharpe": []}))

with tabs[4]:
    st.subheader("Oracle response")
    st.json({"validated": None, "model_hash": None})

with tabs[5]:
    st.subheader("Participants")
    st.dataframe(
        pd.DataFrame(
            [
                {"institution": "A", "stake": 0, "reputation": 100, "status": "pending"},
                {"institution": "B", "stake": 0, "reputation": 100, "status": "pending"},
                {"institution": "C", "stake": 0, "reputation": 100, "status": "pending"},
            ]
        ),
        width="stretch",
    )
