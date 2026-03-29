import gradio as gr
import numpy as np
import joblib
import tensorflow as tf

# Load models and scalers
nsl_model = tf.keras.models.load_model("NSL_KDD_cnn_best.keras")
nsl_scaler = joblib.load("NSL_KDD_scaler.pkl")

def predict(input_data):
    try:
        data = np.array(input_data).reshape(1, -1)
        data = nsl_scaler.transform(data)
        data = data[..., np.newaxis]

        pred = nsl_model.predict(data)
        result = np.argmax(pred)

        return "Attack 🚨" if result == 1 else "Normal ✅"
    except Exception as e:
        return str(e)

iface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Enter features (comma separated)"),
    outputs="text"
)

iface.launch(server_name="0.0.0.0", server_port=7860)
