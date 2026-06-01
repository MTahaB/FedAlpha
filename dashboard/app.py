from __future__ import annotations

import pandas as pd
import streamlit as st


st.set_page_config(page_title="FedAlpha v2", layout="wide")
st.title("FedAlpha v2")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Round", "0")
col2.metric("Model Sharpe", "pending")
col3.metric("Oracle verdict", "pending")
col4.metric("Privacy epsilon", "pending")

tabs = st.tabs(["Training", "Backtest", "Privacy", "Oracle", "Governance"])

with tabs[0]:
    st.subheader("Flower convergence")
    st.line_chart(pd.DataFrame({"loss": []}))

with tabs[1]:
    st.subheader("Backtest")
    st.line_chart(pd.DataFrame({"FL": [], "S&P500": [], "Equal Weight": []}))

with tabs[2]:
    st.subheader("Privacy-performance")
    st.line_chart(pd.DataFrame({"epsilon": [], "sharpe": []}))

with tabs[3]:
    st.subheader("Oracle response")
    st.json({"validated": None, "model_hash": None})

with tabs[4]:
    st.subheader("Participants")
    st.dataframe(
        pd.DataFrame(
            [
                {"institution": "A", "stake": 0, "reputation": 100, "status": "pending"},
                {"institution": "B", "stake": 0, "reputation": 100, "status": "pending"},
                {"institution": "C", "stake": 0, "reputation": 100, "status": "pending"},
            ]
        ),
        use_container_width=True,
    )
