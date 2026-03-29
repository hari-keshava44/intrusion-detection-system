---
title: Network Intrusion Detection System
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# 🛡️ Network Intrusion Detection System

A deep learning-based IDS using **1D-CNN** trained on **CTGAN-augmented** data.

## Models
- **NSL-KDD** — 98.65% accuracy, ROC-AUC 0.9991
- **UNSW-NB15** — 92.42% accuracy, ROC-AUC 0.9865

## How to use
1. Select the dataset model (NSL-KDD or UNSW-NB15)
2. Upload a CSV file of network traffic
3. Click **Run Detection**
4. View per-sample predictions and summary

## Tech Stack
- CTGAN for class balancing
- 1D-CNN for classification
- Gradio for the web interface
