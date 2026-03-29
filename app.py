import gradio as gr
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import os

# ── LOAD MODELS & SCALERS ────────────────────────────────────
# These files must be in the same folder as app.py
NSL_MODEL   = tf.keras.models.load_model("NSL_KDD_cnn_best.keras")
NSL_SCALER  = joblib.load("NSL_KDD_scaler.pkl")
UNSW_MODEL  = tf.keras.models.load_model("UNSW_NB15_cnn_best.keras")
UNSW_SCALER = joblib.load("UNSW_NB15_scaler.pkl")

NSL_FEATURES = 41
UNSW_FEATURES = 42

# ── PREDICTION FUNCTION ──────────────────────────────────────
def predict(file, dataset_choice):
    if file is None:
        return "❌ Please upload a CSV file.", None

    try:
        df = pd.read_csv(file.name)
    except Exception as e:
        return f"❌ Error reading file: {e}", None

    # Select model and scaler
    if dataset_choice == "NSL-KDD":
        model   = NSL_MODEL
        scaler  = NSL_SCALER
        n_feats = NSL_FEATURES
        # Drop label column if present
        for col in ['label', 'difficulty']:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)
    else:
        model   = UNSW_MODEL
        scaler  = UNSW_SCALER
        n_feats = UNSW_FEATURES
        # Drop non-feature columns if present
        for col in ['id', 'attack_cat', 'label', 'Label']:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

    # Encode any remaining object columns
    from sklearn.preprocessing import LabelEncoder
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # Validate feature count
    if df.shape[1] != n_feats:
        return (
            f"❌ Feature mismatch: expected {n_feats} features, "
            f"got {df.shape[1]}. Make sure you upload the correct dataset CSV."
        ), None

    X = df.values.astype(np.float32)
    X_scaled = scaler.transform(X)[..., np.newaxis]

    probs  = model.predict(X_scaled, verbose=0)
    preds  = np.argmax(probs, axis=1)
    labels = ["Normal" if p == 0 else "Attack" for p in preds]

    # Build results dataframe
    results_df = df.copy()
    results_df["Prediction"]       = labels
    results_df["Confidence (%)"]   = (np.max(probs, axis=1) * 100).round(2)
    results_df["Attack Prob (%)"]  = (probs[:, 1] * 100).round(2)

    # Summary
    total   = len(preds)
    attacks = int(np.sum(preds == 1))
    normal  = total - attacks

    summary = f"""## ✅ Prediction Complete

**Dataset Model:** {dataset_choice}
**Total Samples:** {total}
**Normal Traffic:** {normal} ({normal/total*100:.1f}%)
**Attack Traffic:** {attacks} ({attacks/total*100:.1f}%)

{'⚠️ **ALERT: Intrusion detected in ' + str(attacks) + ' samples!**' if attacks > 0 else '✅ **All traffic appears normal.**'}
"""
    return summary, results_df

# ── GRADIO UI ────────────────────────────────────────────────
with gr.Blocks(
    title="Network Intrusion Detection System",
    theme=gr.themes.Base(
        primary_hue="blue",
        secondary_hue="slate",
    )
) as demo:

    gr.Markdown("""
# 🛡️ Network Intrusion Detection System
### Deep Learning (1D-CNN) + CTGAN Augmentation
Detect malicious network traffic instantly. Upload a CSV file of network flows and get predictions in seconds.

---
""")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Configuration")
            dataset_choice = gr.Radio(
                choices=["NSL-KDD", "UNSW-NB15"],
                value="NSL-KDD",
                label="Select Dataset Model",
                info="Choose the model that matches your CSV format"
            )
            file_input = gr.File(
                label="Upload CSV File",
                file_types=[".csv"],
            )
            predict_btn = gr.Button("🔍 Run Detection", variant="primary", size="lg")

            gr.Markdown("""
---
### 📋 CSV Format Guide

**NSL-KDD** — 41 features, no header row required.
Example features: `duration, protocol_type, service, flag, src_bytes ...`

**UNSW-NB15** — 42 features (after dropping id/attack_cat).
Example features: `dur, proto, service, state, spkts, dpkts ...`

Label columns are automatically removed if present.
""")

        with gr.Column(scale=2):
            gr.Markdown("### 📊 Results")
            summary_output = gr.Markdown(value="*Upload a CSV and click Run Detection to see results.*")
            results_table  = gr.Dataframe(
                label="Per-Sample Predictions",
                wrap=True,
                max_rows=20
            )

    predict_btn.click(
        fn=predict,
        inputs=[file_input, dataset_choice],
        outputs=[summary_output, results_table]
    )

    gr.Markdown("""
---
**Model Performance**
| Dataset | Accuracy | F1-Score | ROC-AUC |
|---------|----------|----------|---------|
| NSL-KDD | 98.65% | 0.9865 | 0.9991 |
| UNSW-NB15 | 92.42% | 0.9250 | 0.9865 |

*Models trained with CTGAN-balanced data on Google Colab.*
""")

if __name__ == "__main__":
    demo.launch()
