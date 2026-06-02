"""Project verification helpers used by CLI scripts and the Streamlit dashboard."""

from verification.checks import CheckResult, collect_status

__all__ = ["CheckResult", "collect_status"]
