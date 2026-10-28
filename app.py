import streamlit as st
import joblib
import pandas as pd
from huggingface_hub import hf_hub_download

REPO = "shshShafa/fraud-detection-model"

st.set_page_config(page_title="PaySim Fraud Detector", page_icon="🔍")

@st.cache_resource
def load():
    model = joblib.load(hf_hub_download(REPO, "fraud_model.joblib"))
    encoder = joblib.load(hf_hub_download(REPO, "type_encoder.joblib"))
    return model, encoder

model, encoder = load()

st.title("🔍 PaySim Fraud Detector")
st.caption("Enter a mobile money transaction to check if it looks fraudulent. "
           "Demo model trained on synthetic PaySim data.")

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

c1, c2 = st.columns(2)
c1.button("Try a fraud example", on_click=fill_fraud, use_container_width=True)
c2.button("Try a normal example", on_click=fill_normal, use_container_width=True)

st.subheader("Transaction details")
step = st.number_input("Step (hour of simulation)", min_value=0, key="step")
tx_type = st.selectbox("Transaction type", list(encoder.classes_), key="tx_type")
amount = st.number_input("Amount", min_value=0.0, key="amount")
old_org = st.number_input("Sender balance before", min_value=0.0, key="old_org")
new_org = st.number_input("Sender balance after", min_value=0.0, key="new_org")
old_dest = st.number_input("Receiver balance before", min_value=0.0, key="old_dest")
new_dest = st.number_input("Receiver balance after", min_value=0.0, key="new_dest")

if st.button("Check transaction", type="primary"):
    row = pd.DataFrame([[
        step, encoder.transform([tx_type])[0], amount,
        old_org, new_org, old_dest, new_dest
    ]], columns=["step", "type_encoded", "amount", "oldbalanceOrg",
                 "newbalanceOrig", "oldbalanceDest", "newbalanceDest"])
    prob = model.predict_proba(row)[0][1]
    st.progress(float(prob))
    if prob >= 0.5:
        st.error(f"⚠️ FRAUD SUSPECTED. Fraud probability: {prob:.2%}")
    else:
        st.success(f"✅ Looks legitimate. Fraud probability: {prob:.2%}")

st.divider()
st.caption("Random Forest model trained on the PaySim synthetic dataset. "
           "For learning purposes only, not for real financial decisions.")
