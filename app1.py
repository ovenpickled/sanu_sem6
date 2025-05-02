# app.py
import streamlit as st
import pandas as pd
import numpy as np

# imports for Random Survival Forest
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

@st.cache_resource
def load_and_train_rsf():
    df = pd.read_csv("survival_data.csv")
    # Encode categories
    df['gender']    = df['gender'].map({'Male': 0, 'Female': 1})
    df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})

    # Build structured array for survival:
    #  (event: bool, time: float)
    surv_array = Surv.from_dataframe(
        event='event',
        time='survival_time',
        data=df
    )

    # Features matrix
    X = df[['age','gender','diagnosis']].values

    # Train RSF
    rsf = RandomSurvivalForest(
        n_estimators=100,
        min_samples_split=10,
        min_samples_leaf=15,
        random_state=0
    )
    rsf.fit(X, surv_array)

    return rsf

# Load RSF once
rsf_model = load_and_train_rsf()

st.title("Cancer Survival Predictor (Random Survival Forest)")
st.write("""
Enter the patient’s **Age**, **Gender**, and **Detected Cancer** status,  
choose a time‑horizon (days), and click **Predict** for an RSF‑based estimate.
""")

# Inputs
age     = st.number_input("Age (years)", 0, 120, 50)
gender  = st.selectbox("Gender", ["Male","Female"])
diag    = st.selectbox("Detected Cancer", ["Benign","Malignant"])
horizon = st.number_input("Time horizon (days)", 1, 3650, 365)

# Prepare feature vector
X_new = np.array([[age,
                   0 if gender=="Male" else 1,
                   0 if diag=="Benign" else 1]])

if st.button("🔍 Predict Survival"):
    # RSF.predict_survival_function returns a step fn for each sample
    surv_funcs = rsf_model.predict_survival_function(X_new, return_array=True)
    times = rsf_model.event_times_
    # locate nearest time index
    idx = np.abs(times - horizon).argmin()
    prob = surv_funcs[0, idx] * 100

    st.metric(
      label=f"Estimated Survival Probability at ~{int(times[idx])} days",
      value=f"{prob:.2f}%"
    )