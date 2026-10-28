import streamlit as st
import joblib
import pandas as pd
from huggingface_hub import hf_hub_download

REPO = "shshShafa/fraud-detection-model"

st.set_page_config(page_title="PaySim Fraud Detector", page_icon="🛡️")

@st.cache_resource
def load():
    model = joblib.load(hf_hub_download(REPO, "fraud_model.joblib"))
    encoder = joblib.load(hf_hub_download(REPO, "type_encoder.joblib"))
    return model, encoder

model, encoder = load()

# ---------- Sidebar ----------
with st.sidebar:
    st.header("About this project")
    st.write(
        "A machine learning app that checks a mobile money transaction "
        "and estimates the chance that it is fraud."
    )
    st.subheader("Model")
    st.write("Random Forest Classifier (100 trees), trained on the PaySim synthetic dataset.")
    st.subheader("Test results")
    m1, m2 = st.columns(2)
    m1.metric("Accuracy", "99.97%")
    m2.metric("Precision", "92.3%")
    m3, m4 = st.columns(2)
    m3.metric("Recall", "70.6%")
    m4.metric("F1-score", "79.9%")
    st.caption(
        "Fraud is only about 0.12% of the data, so accuracy alone is misleading. "
        "Recall shows that about 3 in 10 fraud cases are missed."
    )
    st.divider()
    st.caption("Big Data Final Project\n\nNusa Putra University, 2026")

# ---------- Header ----------
st.title("🛡️ PaySim Fraud Detector")
st.write("Enter the details of a mobile money transaction to check whether it looks fraudulent.")

# ---------- Example buttons ----------
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

st.caption("Quick examples")
b1, b2 = st.columns(2)
b1.button("Load a fraud example", on_click=fill_fraud, use_container_width=True)
b2.button("Load a normal example", on_click=fill_normal, use_container_width=True)

# ---------- Form ----------
st.subheader("Transaction details")

left, right = st.columns(2)
with left:
    tx_type = st.selectbox("Transaction type", list(encoder.classes_), key="tx_type")
    amount = st.number_input("Amount", min_value=0.0, key="amount")
    step = st.number_input("Step (hour of simulation)", min_value=0, key="step")
with right:
    old_org = st.number_input("Sender balance before", min_value=0.0, key="old_org")
    new_org = st.number_input("Sender balance after", min_value=0.0, key="new_org")
    old_dest = st.number_input("Receiver balance before", min_value=0.0, key="old_dest")
    new_dest = st.number_input("Receiver balance after", min_value=0.0, key="new_dest")

check = st.button("Check transaction", type="primary", use_container_width=True)

# ---------- Result ----------
if check:
    row = pd.DataFrame([[
        step, encoder.transform([tx_type])[0], amount,
        old_org, new_org, old_dest, new_dest
    ]], columns=["step", "type_encoded", "amount", "oldbalanceOrg",
                 "newbalanceOrig", "oldbalanceDest", "newbalanceDest"])
    prob = float(model.predict_proba(row)[0][1])

    if prob >= 0.5:
        risk = "High"
    elif prob >= 0.2:
        risk = "Medium"
    else:
        risk = "Low"

    st.subheader("Result")
    r1, r2 = st.columns(2)
    r1.metric("Fraud probability", f"{prob:.2%}")
    r2.metric("Risk level", risk)
    st.progress(prob)

    if prob >= 0.5:
        st.error("This transaction is flagged as suspected fraud.")
    else:
        st.success("This transaction looks legitimate.")

# ---------- How it works ----------
with st.expander("How it works"):
    st.write(
        "The model was trained on more than one million simulated mobile money "
        "transactions. For a new transaction it looks at the type, amount, and "
        "account balances before and after, and returns the probability of fraud. "
        "A probability of 50% or more is flagged as suspected fraud."
    )
    st.write(
        "In this dataset, fraud only appears in TRANSFER and CASH_OUT transactions, "
        "often when a whole account balance is emptied in one move."
    )

st.divider()
st.caption(
    "Demo trained on synthetic data. For learning purposes only, "
    "not for real financial decisions."
)
