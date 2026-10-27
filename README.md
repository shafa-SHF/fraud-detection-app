# PaySim Fraud Detector

A machine learning app that checks a mobile money transaction and predicts whether it is fraudulent. The model is a Random Forest classifier trained on the PaySim synthetic dataset.

**Live app:** https://paysim-fraud-detector.streamlit.app/
**Model on Hugging Face:** https://huggingface.co/shshShafa/fraud-detection-model

> Big Data Final Project, Nusa Putra University, Faculty of Engineering, Informatics Engineering Program, 2026.

---

## What it does

You enter one transaction (type, amount, and account balances before and after). The app returns the probability that the transaction is fraud and flags it as suspicious when the probability is 50% or higher.

This is a demo and prototype. It is not connected to a real payment system.

## Dataset

[PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) is a public synthetic dataset that simulates mobile money transactions, built from real transaction logs of a mobile money service in an African country.

| Item | Value |
|---|---|
| Rows used | 1,344,126 |
| Columns | 11 |
| Fraudulent transactions | 1,568 (about 0.12%) |
| Transaction types | CASH_OUT, PAYMENT, CASH_IN, TRANSFER, DEBIT |

Fraud only appears in **TRANSFER** and **CASH_OUT** transactions. The other three types contain no fraud cases.

The dataset is not included in this repository. Download it from Kaggle if you want to re-run the notebook.

## Method

1. **Exploratory data analysis:** transaction types, class imbalance, correlation heatmap, fraud rate by transaction type.
2. **Feature engineering:** the transaction type is label-encoded, and a `HighAmount` flag (amount above 200,000) was explored, matching PaySim's own `isFlaggedFraud` rule.
3. **Model:** `RandomForestClassifier(n_estimators=100, random_state=42)`, with an 80/20 train/test split.
4. **Features used:** `step`, `type`, `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`.

## Results

| Metric | Value |
|---|---|
| Accuracy | 99.97% |
| Precision (fraud) | 92.3% |
| Recall (fraud) | 70.6% |
| F1-score (fraud) | 79.9% |

Confusion matrix on the test set (20,000 transactions):

| | Predicted legitimate | Predicted fraud |
|---|---|---|
| **Actual legitimate** | 19,982 | 1 |
| **Actual fraud** | 5 | 12 |

**Important:** only about 0.12% of transactions are fraud, so accuracy is misleading. A model that always answers "not fraud" would also score about 99.9%. The more useful numbers are precision and recall on the fraud class. At 70.6% recall, roughly 3 in 10 fraud cases are missed.

The most important features were `oldbalanceOrg`, `amount`, and `newbalanceDest`.

## Repository contents

| File | Purpose |
|---|---|
| `fraud_detection_paysim.ipynb` | Google Colab notebook: EDA, training, evaluation, and saving the model |
| `app.py` | Streamlit web app |
| `requirements.txt` | Python packages for the app |
| `README.md` | This file |

The trained model files (`fraud_model.joblib` and `type_encoder.joblib`) are stored on Hugging Face, and the app downloads them when it starts.

## Run it yourself

**Run the app locally**

```bash
git clone https://github.com/shafa-SHF/fraud-detection-app.git
cd fraud-detection-app
pip install -r requirements.txt
streamlit run app.py
```

If you rename the repository, change the folder name in the `cd` command.

**Re-run the training**

1. Open `fraud_detection_paysim.ipynb` in Google Colab.
2. Upload the PaySim CSV when the first cell asks for it, and update the filename in the `pd.read_csv(...)` cell to match your file.
3. Run all cells. The last cell saves the model and encoder with `joblib`.

## Limitations and future work

- The model was trained on **synthetic** data, so it should not be used for real financial decisions.
- Fraud recall can be improved with `class_weight="balanced"`, SMOTE or undersampling, and a lower decision threshold.
- Other ideas: k-fold cross-validation, comparing XGBoost, LightGBM, and Isolation Forest, and adding features such as balance differences.
- A real-time version would need the model to receive transactions automatically from a payment system instead of manual input.

## Tech stack

Python, pandas, scikit-learn, matplotlib, seaborn, joblib, Streamlit, Hugging Face Hub, Google Colab.

## Team

| Name | Student ID |
|---|---|
| Naina Mohammed Fathima Saafa | 20240040325 |
| Nyimansata Fofana | 20240040316 |
| Milenia Soares Goncalves | 20240040323 |

Nusa Putra University, 2026.
