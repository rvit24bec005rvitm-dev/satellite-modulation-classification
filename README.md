# 📡 AI-Based Satellite Modulation Classification

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI / Validation](https://img.shields.io/badge/CI-Passing-brightgreen.svg)](.github/workflows/ci.yml)

An enterprise-grade, academic software suite for satellite communication systems utilizing Deep Learning (Convolutional Neural Networks, ResNet, and CLDNN) for **Automatic Modulation Classification (AMC)** on complex baseband In-phase and Quadrature (I/Q) radio frequency signals.

---

## 🌟 Key Capabilities & Architecture

- **Multi-Page Web Dashboard:** Modern dark-theme glassmorphism UI deployed on Streamlit Community Cloud.
- **🛰️ Live Modulation Classifier (`pages/1_🛰️_Live_Classifier.py`):**
  - Instant classification of **BPSK**, **QPSK**, and **16-QAM** signals.
  - Multi-format file ingestion supporting `.npy`, `.npz`, `.csv`, `.mat` (MATLAB), and `.wav` (stereo baseband).
  - Shannon Entropy uncertainty estimation ($H(p)$ gauge) to detect out-of-distribution or corrupted frames.
- **🎛️ Synthetic Satellite Signal Studio (`pages/2_🎛️_Satellite_Studio.py`):**
  - Real-time signal synthesizer for BPSK, QPSK, 8-PSK, 16-QAM, 16-APSK, 32-APSK, GMSK, and BFSK.
  - Hardware & channel impairments: **LEO Doppler Shift**, **Carrier Frequency Offset (CFO)**, **Phase Noise**, **Rician Fading**, Root-Raised Cosine (RRC) filtering, and **Saleh TWTA non-linear amplifier saturation**.
- **📊 Model Evaluation Dashboard (`pages/3_📊_Model_Evaluation.py`):**
  - Interactive Plotly confusion matrix heatmaps, SNR vs. Accuracy performance curves (-20 dB to +18 dB), and per-class classification metrics.
- **🔬 Deep Learning Model Arena (`pages/4_🔬_Model_Arena.py`):**
  - Side-by-side benchmarking of **1D-CNN**, **1D-ResNet**, **CLDNN (CNN + Bi-LSTM)**, and **Random Forest Expert Baseline**.
  - Live latency profiling (P95 latency, FPS throughput).
- **📖 Satellite Theory Encyclopedia (`pages/5_📖_Satellite_Theory.py`):**
  - Mathematical formulations of APSK constellations, TWTA Saleh AM/AM and AM/PM transfer curves, LEO Doppler equations, and AMC literature references.
- **⚡ Zero-RAM Native NumPy Engine:**
  - Self-contained inference engine that executes the trained 1D-CNN in pure NumPy in `<1 ms` without requiring heavy C++ packages or exceeding the 1 GB memory limit on Streamlit Cloud.

---

## 🏗️ 1D-CNN Pipeline Architecture

```
Raw Baseband I/Q Signal [128, 2]
         │
         ▼
Z-Score Standardization (μ = -6.56e-5, σ = 0.00607)
         │
         ▼
Conv1D (32 filters, k=5, ReLU) ──► BatchNorm ──► MaxPool1D(2) ──► Dropout(0.20)
         │
         ▼
Conv1D (64 filters, k=5, ReLU) ──► BatchNorm ──► MaxPool1D(2) ──► Dropout(0.25)
         │
         ▼
Conv1D (96 filters, k=3, ReLU) ──► BatchNorm ──► GlobalAveragePooling1D -> (96,)
         │
         ▼
Dense (64 units, ReLU) ──► Dropout(0.40)
         │
         ▼
Dense (3 units, Softmax) ──► [P(BPSK), P(QPSK), P(16-QAM)]
```

---

## 📊 Empirical Performance Summary

| Metric | Overall (-20 dB to +18 dB) | High SNR Tier (+6 dB to +18 dB) | Low SNR Tier (-20 dB to -6 dB) |
| :--- | :--- | :--- | :--- |
| **Accuracy** | **72.44%** | **95.5% – 100%** | 20.0% – 33.3% |
| **BPSK F1-Score** | 0.75 | 0.99 | 0.35 |
| **QPSK F1-Score** | 0.70 | 0.96 | 0.30 |
| **16-QAM F1-Score** | 0.73 | 0.94 | 0.28 |

---

## 🚀 Quickstart & Local Installation

### 1. Clone the Repository
```bash
git clone https://github.com/rvit24bec005rvitm-dev/satellite-modulation-classification.git
cd satellite-modulation-classification
```

### 2. Set Up Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Streamlit Application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

### 5. Run the Automated Test Suite
```bash
python3 -c "
import tests.test_signal_processing as tsp, tests.test_model_backend as tmb
tsp.test_normalization()
tsp.test_synthetic_signal_generator_shapes()
tsp.test_channel_impairments()
tsp.test_psd_and_spectrogram()
tsp.test_shannon_entropy()
tsp.test_extract_features()
tmb.test_model_backend_loading()
tmb.test_model_inference_output()
tmb.test_dataset_samples_compatibility()
print('All tests passed successfully!')
"
```
Or with pytest:
```bash
pytest tests/ -v
```

---

## ☁️ Streamlit Community Cloud Deployment

1. Fork or push this repository to GitHub.
2. Sign in to [share.streamlit.io](https://share.streamlit.io).
3. Click **"New App"** and select:
   - **Repository:** `rvit24bec005rvitm-dev/satellite-modulation-classification`
   - **Branch:** `main` (or `master`)
   - **Main file path:** `app.py`
4. Click **"Deploy"**! The application will initialize smoothly in seconds without OOM crashes.

---

## 📂 Repository Structure

```
satellite-modulation-classification/
├── .github/
│   └── workflows/
│       └── ci.yml                          # Automated GitHub Actions CI workflow
├── .streamlit/
│   └── config.toml                         # Custom dark satellite theme & upload limits
├── app.py                                  # Landing page & navigation hub
├── pages/
│   ├── 1_🛰️_Live_Classifier.py             # Live sample & file upload classifier
│   ├── 2_🎛️_Satellite_Studio.py            # Synthetic signal generator & channel simulator
│   ├── 3_📊_Model_Evaluation.py            # Confusion matrix & SNR curve analytics
│   ├── 4_🔬_Model_Arena.py                 # Multi-model benchmarking & latency profiling
│   └── 5_📖_Satellite_Theory.py            # Technical reference & modulation physics
├── models/
│   └── cnn_weights.npz                     # Zero-RAM compressed weights for native NumPy inference
├── utils/
│   ├── __init__.py
│   ├── model_backend.py                    # Multi-backend wrapper (NumPy / ONNX / Keras)
│   ├── plotting.py                         # Interactive Plotly chart builders
│   └── signal_processing.py                # PSD, Spectrogram, Impairments, Constellations
├── tests/
│   ├── test_model_backend.py               # Backend loading & inference tests
│   └── test_signal_processing.py           # Signal generator & impairment tests
├── classifier_samples.npz                  # 300 test signal samples across classes & SNRs
├── evaluation_results.json                 # Classification report & metrics data
├── normalization_stats_all_snr.npz         # Mean & std values used during training
├── requirements.txt                        # Streamlit Cloud production requirements
├── satellite_modulation_cnn_all_snr.keras  # Original Keras model archive
├── snr_accuracy_results.csv                # Accuracy breakdown per SNR level
└── README.md                               # Project documentation
```

---

## 📄 License
This project is licensed under the MIT License.
