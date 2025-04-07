import streamlit as st
import joblib as jb
import numpy as np

# Load Model
model, label_encoders = jb.load("Circular_Model.pkl")

# Custom Styling
st.markdown(
    """
    <style>
        .title { font-size: 36px; color: #2c3e50; text-align: center; font-weight: bold; }
        .stButton>button { background-color: #1abc9c; color: white; font-size: 18px; border-radius: 10px; }
        .stButton>button:hover { background-color: #16a085; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown('<div class="title">♻️ AI-Powered Circular Economy Model</div>', unsafe_allow_html=True)
st.write("\n")

# Input Section
industry = st.selectbox("🏭 Select Industry", label_encoders["Industry"].classes_)
material = st.selectbox("🗑️ Select Waste Material", label_encoders["Material"].classes_)

# Predict Button
if st.button("🔄 Suggest Action"):
    industry_encoded = label_encoders["Industry"].transform([industry])[0]
    material_encoded = label_encoders["Material"].transform([material])[0]
    prediction = model.predict([[industry_encoded, material_encoded]])[0]
    suggested_action = label_encoders["Action"].inverse_transform([prediction])[0]
    
    st.success(f"✅ Recommended Action: **{suggested_action}**")
    st.balloons()

st.info("🌍 Reduce, Reuse, Recycle for a Sustainable Future!")
