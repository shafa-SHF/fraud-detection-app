import streamlit as st
import joblib
import pandas as pd
from huggingface_hub import hf_hub_download

REPO = "shshShafa/fraud-detection-model"

@st.cache_resource
def load():
    model = joblib.load(hf_hub_download(REPO, "fraud_model.joblib"))
    encoder = joblib.load(hf_hub_download(REPO, "type_encoder.joblib"))
    return model, encoder

model, encoder = load()

st.title("Fraud Detection")

step = st.number_input("Step (hour)", value=1, min_value=0)
tx_type = st.selectbox("Transaction type", list(encoder.classes_))
amount = st.number_input("Amount", value=0.0)
old_org = st.number_input("Sender balance before", value=0.0)
new_org = st.number_input("Sender balance after", value=0.0)
old_dest = st.number_input("Receiver balance before", value=0.0)
new_dest = st.number_input("Receiver balance after", value=0.0)

if st.button("Check transaction"):
    row = pd.DataFrame([[
        step, encoder.transform([tx_type])[0], amount,
        old_org, new_org, old_dest, new_dest
    ]], columns=["step", "type_encoded", "amount", "oldbalanceOrg",
                 "newbalanceOrig", "oldbalanceDest", "newbalanceDest"])
    prob = model.predict_proba(row)[0][1]
    if prob >= 0.5:
        st.error(f"FRAUD SUSPECTED ({prob:.2%})")
    else:
        st.success(f"Looks legitimate (fraud probability {prob:.2%})")
