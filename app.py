import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("NSL_KDD_cnn_best.keras")
scaler = joblib.load("NSL_KDD_scaler.pkl")

st.title("Intrusion Detection System")

input_text = st.text_input("Enter features (comma separated)")

if st.button("Predict"):
    try:
        values = list(map(float, input_text.split(",")))
        data = np.array(values).reshape(1, -1)

        data = scaler.transform(data)
        data = data[..., np.newaxis]

        pred = model.predict(data)
        result = np.argmax(pred)

        if result == 1:
            st.error("🚨 Attack Detected")
        else:
            st.success("✅ Normal Traffic")

    except Exception as e:
        st.error(f"Error: {e}")
