
import streamlit as st
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('creditcard.csv')
X = df.drop('Class', axis=1)
y = df['Class']

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model training
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_scaled, y)

st.title("🚨 Loan Fraud Detection App")

input_data = st.text_area("Enter 30 values (comma-separated):")

if input_data:
    try:
        values = np.array([float(x) for x in input_data.split(',')]).reshape(1, -1)
        values = scaler.transform(values)
        result = model.predict(values)
        st.success("Prediction: FRAUD ❌" if result[0]==1 else "Prediction: Genuine ✅")
    except:
        st.error("Please enter 30 valid numbers!")
