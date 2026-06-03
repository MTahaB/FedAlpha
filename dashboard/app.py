from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.results import load_dashboard_results, participant_rows
from verification.checks import collect_status, status_counts


st.set_page_config(page_title="FedAlpha v2", layout="wide")
st.title("FedAlpha v2")
results = load_dashboard_results()

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
metrics = results["metrics"]
oracle = results["oracle"]
privacy = results["privacy"]
latest_epsilon = "pending"
if not privacy.empty and "epsilon" in privacy:
    latest_epsilon = f"{privacy['epsilon'].replace([float('inf')], pd.NA).dropna().min():.2f}"

col1.metric("Round", str(results["round"]))
col2.metric("Model Sharpe", f"{metrics.get('sharpe_ratio', 'pending'):.3f}" if "sharpe_ratio" in metrics else "pending")
col3.metric("Oracle verdict", str(oracle.get("validated", "pending")))
col4.metric("Privacy epsilon", latest_epsilon)

tabs = st.tabs(["Verification", "Training", "Backtest", "Privacy", "Oracle", "Governance"])

with tabs[0]:
    render_verification()

with tabs[1]:
    st.subheader("Flower convergence")
    training = results["training"]
    if training.empty:
        st.info("No federated training report found yet.")
    else:
        st.line_chart(training, x="client", y="loss")

with tabs[2]:
    st.subheader("Backtest")
    returns = results["returns"]
    if returns.empty:
        st.info("No backtest report found yet.")
    else:
        date_column = "date" if "date" in returns.columns else returns.columns[0]
        value_column = "portfolio_return" if "portfolio_return" in returns.columns else returns.columns[-1]
        returns["equity"] = (1 + returns[value_column].fillna(0.0)).cumprod()
        st.line_chart(returns, x=date_column, y="equity")
        st.json(metrics)

with tabs[3]:
    st.subheader("Privacy-performance")
    if privacy.empty:
        st.info("No privacy tradeoff report found yet.")
    else:
        st.line_chart(privacy, x="epsilon", y="sharpe", color="mode")

with tabs[4]:
    st.subheader("Oracle response")
    st.json(oracle or {"validated": None, "model_hash": None})
    if results["blockchain_events"]:
        st.subheader("Blockchain anchors")
        st.dataframe(pd.DataFrame(results["blockchain_events"]), width="stretch", hide_index=True)

with tabs[5]:
    st.subheader("Participants")
    st.dataframe(participant_rows(results), width="stretch", hide_index=True)
