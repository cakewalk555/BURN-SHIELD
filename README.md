
# BURN-SHIELD — SIH26170 Prototype

A demonstration prototype for:
**AI-Driven Anomaly Detection in Component Burn-In & Screening**

## What it demonstrates

1. Dynamic/lot-relative anomaly detection
2. Time-series trajectory analysis
3. AI-based outlier detection using Isolation Forest
4. 168h prediction using Random Forest
5. Engineer-facing explanation of why a component was flagged
6. What-if future drift simulation

## Important

The supplied SIH problem statement does not provide a public ISRO training dataset or mandate a particular ML algorithm.
This prototype therefore uses synthetic demonstration data and implementation choices that must be validated with real domain data.

## Run on Windows

Open Command Prompt in this folder:

```text
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python generate_data.py
streamlit run app.py
```

The browser should open the dashboard automatically.
