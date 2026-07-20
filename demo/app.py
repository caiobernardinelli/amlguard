"""Public recruiter demo for AMLGuard transaction risk scoring.

Run from the repository root with:

    python -m streamlit run demo/app.py

The interface intentionally accepts only the five model features used by the
validated AMLGuard pipeline. Users should enter synthetic or non-sensitive
values only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from src.models.predict import predict_transaction

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "artifacts" / "model.joblib"

PAYMENT_FORMATS = (
    "ACH",
    "Bitcoin",
    "Cash",
    "Cheque",
    "Credit Card",
    "Reinvestment",
    "Wire",
)

DEFAULT_AMOUNT_PAID = 13701.30
DEFAULT_PREVIOUS_TX_COUNT = 238


def build_model_features(
    *,
    payment_format: str,
    amount_paid: float,
    sender_previous_tx_count: int,
    is_business_hours: bool,
    same_account: bool,
) -> dict[str, Any]:
    """Translate recruiter-friendly form values into the frozen model contract."""
    return {
        "Payment Format": payment_format,
        "Amount Paid": float(amount_paid),
        "sender_previous_tx_count": int(sender_previous_tx_count),
        "is_business_hours": int(is_business_hours),
        "same_account": int(same_account),
    }


def render_result(result: dict[str, Any]) -> None:
    """Render one AMLGuard scoring result without overstating model certainty."""
    risk_score = float(result["risk_score"])
    threshold = float(result["threshold"])
    is_alert = bool(result["is_alert"])
    model_version = str(result["model_version"])

    st.divider()
    st.subheader("AMLGuard Risk Assessment")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Risk score", f"{risk_score:.4f}")
        st.caption("Model score in [0, 1]. It is not presented as a calibrated probability.")

    with col2:
        st.metric("Alert threshold", f"{threshold:.6f}")
        st.caption(f"Model version: {model_version}")

    st.progress(min(max(risk_score, 0.0), 1.0))

    if is_alert:
        st.error(
            "High-risk alert — the transaction scored above the frozen AMLGuard "
            "operating threshold and would be forwarded for analyst review."
        )
    else:
        st.success(
            "Below alert threshold — the transaction did not cross the frozen "
            "AMLGuard operating threshold."
        )

    with st.expander("Technical response"):
        st.json(result)


def main() -> None:
    st.set_page_config(
        page_title="AMLGuard | Transaction Risk Demo",
        page_icon="🛡️",
        layout="centered",
    )

    st.title("🛡️ AMLGuard")
    st.subheader("Anti-Money Laundering Transaction Risk Demo")
    st.write(
        "Enter a synthetic transaction profile and score it with the validated "
        "AMLGuard XGBoost pipeline."
    )

    st.info(
        "Portfolio demonstration only. Do not enter real account identifiers, "
        "personal data, credentials, or confidential banking information."
    )

    with st.expander("What this demo proves"):
        st.markdown(
            """
- The public interface calls the same persisted AMLGuard model used by the project.
- The scoring contract preserves the frozen operating threshold.
- The result returns a risk score, alert decision, threshold, and model version.
- The application is designed for portfolio demonstration, not production compliance decisions.
"""
        )

    if not MODEL_PATH.is_file():
        st.error(
            "The AMLGuard model artifact is unavailable. "
            "Expected file: artifacts/model.joblib"
        )
        st.stop()

    st.divider()
    st.subheader("Transaction inputs")

    with st.form("transaction_form"):
        payment_format = st.selectbox(
            "Payment format",
            PAYMENT_FORMATS,
            index=0,
            help="Payment channel supplied to the trained AMLGuard model.",
        )

        amount_paid = st.number_input(
            "Amount paid",
            min_value=0.0,
            value=DEFAULT_AMOUNT_PAID,
            step=100.0,
            format="%.2f",
        )

        sender_previous_tx_count = st.number_input(
            "Sender previous transaction count",
            min_value=0,
            value=DEFAULT_PREVIOUS_TX_COUNT,
            step=1,
            help=(
                "Leakage-safe historical feature: number of transactions observed "
                "for the sender before the current transaction."
            ),
        )

        is_business_hours = st.checkbox(
            "Transaction occurred during business hours",
            value=False,
        )

        same_account = st.checkbox(
            "Source and destination represent the same account",
            value=False,
        )

        submitted = st.form_submit_button(
            "Analyze transaction",
            use_container_width=True,
            type="primary",
        )

    if submitted:
        features = build_model_features(
            payment_format=payment_format,
            amount_paid=amount_paid,
            sender_previous_tx_count=sender_previous_tx_count,
            is_business_hours=is_business_hours,
            same_account=same_account,
        )

        try:
            result = predict_transaction(features)
        except Exception as exc:
            st.exception(exc)
        else:
            render_result(result)

    st.divider()
    st.caption(
        "AMLGuard is an educational portfolio project for financial-crime risk "
        "modelling under extreme class imbalance. Model outputs require human review."
    )
    st.markdown(
        "[View the AMLGuard source code on GitHub]"
        "(https://github.com/caiobernardinelli/amlguard)"
    )


if __name__ == "__main__":
    main()
