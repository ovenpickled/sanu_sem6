# app.py
import streamlit as st
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter

@st.cache_resource
def load_and_train():
    # 1) Load your survival dataset
    df = pd.read_csv("survival_data.csv")
    
    # 2) Encode categorical covariates
    df['gender']    = df['gender'].map({'Male': 0, 'Female': 1})
    df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})
    
    # 3) Fit Cox Proportional Hazards model
    cph = CoxPHFitter()
    cph.fit(
        df[['age','gender','diagnosis','survival_time','event']],
        duration_col='survival_time',
        event_col='event'
    )
    return cph

# Cache & load model once
surv_model = load_and_train()

st.title("Cancer Survival Predictor")
st.write("""
Provide the patient’s details below, select a time‑horizon (in days),  
and click **Predict** to see their estimated survival probability.
""")

# --- User Inputs ---
age       = st.number_input("Age (years)", min_value=0, max_value=120, value=50)
gender    = st.selectbox("Gender", ["Male","Female"])
diag      = st.selectbox("Detected Cancer", ["Benign","Malignant"])
horizon   = st.number_input(
    "Time horizon (days) for survival probability",
    min_value=1, max_value=3650, value=365
)

# --- Prepare input for model ---
X_new = pd.DataFrame({
    'age':       [age],
    'gender':    [0 if gender=="Male" else 1],
    'diagnosis': [0 if diag=="Benign" else 1]
})

# --- Predict & display ---
if st.button("🔍 Predict Survival"):
    # Obtain survival function (indexed by time)
    surv_fn = surv_model.predict_survival_function(X_new)
    times = surv_fn.index.values
    # find nearest available time to your horizon
    idx = np.abs(times - horizon).argmin()
    surv_prob = float(surv_fn.iloc[idx, 0] * 100)

    st.metric(
        label=f"Estimated Survival Probability at ~{int(times[idx])} days",
        value=f"{surv_prob:.2f}%",
        delta=None
    )