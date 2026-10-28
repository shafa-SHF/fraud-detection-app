import streamlit as st
import joblib
import pandas as pd
from huggingface_hub import hf_hub_download

REPO = "shshShafa/fraud-detection-model"

st.set_page_config(page_title="PaySim Fraud Detector", page_icon="🛡️", layout="wide")

st.markdown(
    """
    <style>
    .banner {
        background: linear-gradient(90deg, #1D4ED8 0%, #0F172A 100%);
        padding: 28px 32px; border-radius: 14px; margin-bottom: 18px;
    }
    .banner h1 { color: #FFFFFF; margin: 0; font-size: 2rem; }
    .banner p { color: #CBD5E1; margin: 6px 0 0 0; font-size: 1rem; }
    .result-card {
        padding: 22px 26px; border-radius: 14px; border: 1px solid #E2E8F0;
        margin-top: 8px;
    }
    .result-card h2 { margin: 0 0 4px 0; font-size: 1.6rem; }
    .result-card p { margin: 0; font-size: 1rem; }
    .high { background: #FEF2F2; border-color: #FCA5A5; color: #991B1B; }
    .medium { background: #FFFBEB; border-color: #FCD34D; color: #92400E; }
    .low { background: #F0FDF4; border-color: #86EFAC; color: #166534; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load():
    model = joblib.load(hf_hub_download(REPO, "fraud_model.joblib"))
    encoder = joblib.load(hf_hub_download(REPO, "type_encoder.joblib"))
    return model, encoder


model, encoder = load()

# ---------- Banner ----------
st.markdown(
    """
    <div class="banner">
        <h1>🛡️ PaySim Fraud Detector</h1>
        <p>Check whether a mobile money transaction looks fraudulent, powered by a Random Forest model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_check, tab_about = st.tabs(["Check a transaction", "About the model"])

# ---------- Tab 1: Check ----------
with tab_check:
    DEFAULTS = {"step": 1, "tx_type": "TRANSFER", "amount": 0.0,
                "old_org": 0.0, "new_org": 0.0, "old_dest": 0.0, "new_dest": 0.0}
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(k, v)

    def fill_fraud():
        st.session_state.update(step=1, tx_type="TRANSFER", amount=181.0,
                                old_org=181.0, new_org=0.0, old_dest=0.0, new_dest=0.0)

    def fill_normal():
        st.session_state.update(step=1, tx_type="PAYMENT", amount=9839.64,
                                old_org=170136.0, new_org=160296.36,
                                old_dest=0.0, new_dest=0.0)

    form_col, result_col = st.columns([3, 2], gap="large")

    with form_col:
        st.subheader("Transaction details")
        b1, b2 = st.columns(2)
        b1.button("Load a fraud example", on_click=fill_fraud, use_container_width=True)
        b2.button("Load a normal example", on_click=fill_normal, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            tx_type = st.selectbox("Transaction type", list(encoder.classes_), key="tx_type")
            amount = st.number_input("Amount", min_value=0.0, key="amount")
            step = st.number_input("Step (hour of simulation)", min_value=0, key="step")
        with c2:
            old_org = st.number_input("Sender balance before", min_value=0.0, key="old_org")
            new_org = st.number_input("Sender balance after", min_value=0.0, key="new_org")
            old_dest = st.number_input("Receiver balance before", min_value=0.0, key="old_dest")
            new_dest = st.number_input("Receiver balance after", min_value=0.0, key="new_dest")

        check = st.button("Check transaction", type="primary", use_container_width=True)

    with result_col:
        st.subheader("Result")
        if not check:
            st.info("Fill in the transaction details, or load an example, then click **Check transaction**.")
        elif amount <= 0:
            st.warning("Enter an amount greater than 0 to check a transaction.")
        else:
            row = pd.DataFrame([[
                step, encoder.transform([tx_type])[0], amount,
                old_org, new_org, old_dest, new_dest
            ]], columns=["step", "type_encoded", "amount", "oldbalanceOrg",
                         "newbalanceOrig", "oldbalanceDest", "newbalanceDest"])
            prob = float(model.predict_proba(row)[0][1])

            if prob >= 0.5:
                level, css, title = "High", "high", "⚠️ Suspected fraud"
                text = "This transaction is flagged for review."
            elif prob >= 0.2:
                level, css, title = "Medium", "medium", "🟠 Needs attention"
                text = "Some signs of risk. Consider a manual check."
            else:
                level, css, title = "Low", "low", "✅ Looks legitimate"
                text = "No strong signs of fraud were found."

            st.markdown(
                f'<div class="result-card {css}"><h2>{title}</h2><p>{text}</p></div>',
                unsafe_allow_html=True,
            )
            m1, m2 = st.columns(2)
            m1.metric("Fraud probability", f"{prob:.2%}")
            m2.metric("Risk level", level)
            st.progress(prob)

            st.markdown("**Quick checks**")
            emptied = old_org > 0 and new_org == 0
            risky_type = tx_type in ("TRANSFER", "CASH_OUT")
            expected_drop = abs((old_org - new_org) - amount) < 0.01
            st.write(("🔴" if emptied else "🟢") + " Sender account emptied: " + ("yes" if emptied else "no"))
            st.write(("🔴" if risky_type else "🟢") + " Type is TRANSFER or CASH_OUT: " + ("yes" if risky_type else "no"))
            st.write(("🟢" if expected_drop else "🟠") + " Sender balance change matches the amount: " + ("yes" if expected_drop else "no"))

# ---------- Tab 2: About ----------
with tab_about:
    st.subheader("About the model")
    a1, a2 = st.columns(2)
    with a1:
        st.write(
            "**Model:** Random Forest Classifier (100 trees)\n\n"
            "**Dataset:** PaySim synthetic mobile money transactions "
            "(about 1.3 million rows, 0.12% fraud)\n\n"
            "**Inputs:** transaction type, amount, step, and the sender and "
            "receiver balances before and after the transaction."
        )
    with a2:
        k1, k2 = st.columns(2)
        k1.metric("Accuracy", "99.97%")
        k2.metric("Precision", "92.3%")
        k3, k4 = st.columns(2)
        k3.metric("Recall", "70.6%")
        k4.metric("F1-score", "79.9%")

    st.write(
        "Fraud is only about 0.12% of the data, so accuracy alone is misleading. "
        "A recall of 70.6% means about 3 in 10 fraud cases are missed. "
        "In this dataset, fraud only appears in TRANSFER and CASH_OUT transactions."
    )
    st.caption("Big Data Final Project, Nusa Putra University, 2026.")

st.divider()
st.caption(
    "Demo trained on synthetic data. For learning purposes only, "
    "not for real financial decisions."
)
