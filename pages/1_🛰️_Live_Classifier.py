"""
pages/1_🛰️_Live_Classifier.py
Interactive modulation classifier with:
 - Sample picker (from saved NPZ)
 - File upload (.npy, .npz, .csv, .mat, .wav)
 - Constellation, waveform, PSD visualisations
 - Shannon entropy uncertainty gauge
"""

from pathlib import Path
import sys, io, json

import numpy as np
import pandas as pd
import streamlit as st

# ── Path setup ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.signal_processing import (
    normalize_signal, compute_psd, compute_spectrogram,
    shannon_entropy, extract_features, CONSTELLATION_REFS,
)
from utils.plotting import (
    plot_constellation, plot_time_domain, plot_psd,
    plot_spectrogram, plot_class_scores, plot_entropy_gauge,
)
from utils.model_backend import ModelBackend

# ── Page config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Live Classifier · SatModClass",
    page_icon="🛰️",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html,[class*="css"]{ font-family:'Inter',sans-serif; }
.result-card{
    background:linear-gradient(135deg,#111827,#0F172A);
    border:1px solid #1F2937;
    border-radius:14px;
    padding:1.5rem 2rem;
    margin:1rem 0;
}
.pred-label{
    font-size:2.5rem;font-weight:800;
    background:linear-gradient(90deg,#00D4FF,#7C3AED);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
}
.match-success{ color:#34D399;font-weight:700;font-size:1.1rem; }
.match-fail   { color:#F87171;font-weight:700;font-size:1.1rem; }
</style>
""", unsafe_allow_html=True)

st.title("🛰️ Live Modulation Classifier")
st.markdown("Classify **BPSK**, **QPSK**, and **16-QAM** signals from RadioML samples or your own uploaded I/Q data.")

# ── Constants ───────────────────────────────────────────────────────────────────
CLASSES = ["BPSK", "QPSK", "QAM16"]
MAX_ENTROPY = np.log2(len(CLASSES))   # 1.585 bits for 3 classes

# ── Load model ──────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    # Prefer ONNX if available, else .keras
    onnx_path = ROOT / "models" / "satellite_modulation_cnn.onnx"
    keras_path = ROOT / "satellite_modulation_cnn_all_snr.keras"
    if onnx_path.exists():
        return ModelBackend.load(onnx_path)
    elif keras_path.exists():
        return ModelBackend.load(keras_path)
    return None

@st.cache_data(show_spinner="Loading normalisation stats…")
def load_stats():
    p = ROOT / "normalization_stats_all_snr.npz"
    if not p.exists():
        return None, None
    with np.load(p) as d:
        return float(d["mean"]), float(d["std"])

@st.cache_data(show_spinner="Loading signal samples…")
def load_samples():
    p = ROOT / "classifier_samples.npz"
    if not p.exists():
        return None, None, None
    with np.load(p, allow_pickle=False) as d:
        return d["signals"].copy(), d["labels"].astype(str), d["snrs"].copy()

model = load_model()
mean_val, std_val = load_stats()
signals_all, labels_all, snrs_all = load_samples()

if model is None:
    st.error("❌ No model found. Place `satellite_modulation_cnn_all_snr.keras` or `models/satellite_modulation_cnn.onnx` in the project root.")
    st.stop()
if mean_val is None:
    st.error("❌ `normalization_stats_all_snr.npz` not found.")
    st.stop()

st.success(f"✅ Model loaded via **{model.backend_name}** · Input shape: `{model.input_shape}`")

st.divider()

# ── Signal source selection ─────────────────────────────────────────────────────
tab_sample, tab_upload = st.tabs(["📦 RadioML Sample Library", "📂 Upload Your Own Signal"])

signal_iq: np.ndarray | None = None
ground_truth: str | None = None
selected_snr: int | None = None

# ┌── Tab 1: Sample library ────────────────────────────────────────────────────┐
with tab_sample:
    if signals_all is None:
        st.warning("⚠️ `classifier_samples.npz` not found.")
    else:
        available_snrs = sorted(int(v) for v in np.unique(snrs_all))
        c1, c2, c3 = st.columns(3)
        with c1:
            sel_class = st.selectbox("Modulation class", CLASSES, key="sc_class")
        with c2:
            default_idx = available_snrs.index(0) if 0 in available_snrs else 0
            selected_snr = st.selectbox("SNR (dB)", available_snrs, index=default_idx, key="sc_snr")
        
        mask = (labels_all == sel_class) & (snrs_all == selected_snr)
        matching = np.where(mask)[0]
        
        if len(matching) == 0:
            st.warning("No samples for this class + SNR combination.")
        else:
            with c3:
                sample_idx = st.number_input("Sample index", 0, len(matching) - 1, 0, key="sc_idx")
            signal_iq = signals_all[matching[int(sample_idx)]].copy()
            ground_truth = sel_class
            st.caption(f"Signal shape: `{signal_iq.shape}` — {len(matching)} samples available at {selected_snr} dB")

# ┌── Tab 2: File upload ────────────────────────────────────────────────────────┐
with tab_upload:
    st.markdown("Upload a file containing a **single I/Q signal** (shape `(128, 2)` or `(2, 128)`).")
    st.markdown("**Supported:** `.npy` `.npz` `.csv` `.mat` `.wav`")
    
    uploaded = st.file_uploader(
        "Drag and drop your signal file",
        type=["npy", "npz", "csv", "mat", "wav"],
        key="upload_file",
    )
    
    if uploaded:
        ext = Path(uploaded.name).suffix.lower()
        try:
            if ext == ".npy":
                arr = np.load(io.BytesIO(uploaded.read()))
            elif ext == ".npz":
                z = np.load(io.BytesIO(uploaded.read()), allow_pickle=False)
                arr = z[z.files[0]]
            elif ext == ".csv":
                df_up = pd.read_csv(uploaded)
                arr = df_up.values
            elif ext == ".mat":
                from scipy.io import loadmat
                mat = loadmat(io.BytesIO(uploaded.read()))
                keys = [k for k in mat if not k.startswith("_")]
                arr = mat[keys[0]] if keys else None
            elif ext == ".wav":
                from scipy.io import wavfile
                sr, wav = wavfile.read(io.BytesIO(uploaded.read()))
                if wav.ndim == 1:
                    arr = np.stack([wav, np.zeros_like(wav)], axis=1).astype(np.float32)
                else:
                    arr = wav[:, :2].astype(np.float32)
            else:
                arr = None

            if arr is None:
                st.error("Could not parse the uploaded file.")
            else:
                if arr.shape == (2, 128):
                    arr = arr.T
                if arr.shape[0] != 128:
                    # Trim or pad to 128
                    if arr.shape[0] > 128:
                        arr = arr[:128, :]
                    else:
                        arr = np.pad(arr, ((0, 128 - arr.shape[0]), (0, 0)))
                if arr.shape != (128, 2):
                    arr = arr.reshape(128, 2)
                
                # Normalise to unit power
                pwr = np.mean(arr[:, 0]**2 + arr[:, 1]**2)
                if pwr > 0:
                    arr = arr / np.sqrt(pwr)
                
                signal_iq = arr.astype(np.float32)
                st.success(f"✅ Loaded `{uploaded.name}` → shape `{signal_iq.shape}`")

        except Exception as e:
            st.error(f"Parse error: {e}")

st.divider()

# ── Classify ────────────────────────────────────────────────────────────────────
if signal_iq is not None:
    if st.button("🚀 Classify Signal", type="primary", use_container_width=True):
        st.session_state["classify"] = True

    if st.session_state.get("classify"):

        # Normalise
        iq_norm = normalize_signal(signal_iq, mean_val, std_val)

        # Inference
        with st.spinner("Running CNN inference…"):
            try:
                probs = model.predict(iq_norm)
                pred_idx = int(np.argmax(probs))
                pred_class = CLASSES[pred_idx]
                confidence = float(probs[pred_idx]) * 100
                entropy = shannon_entropy(probs)
            except Exception as e:
                st.error(f"Inference failed: {e}")
                st.stop()

        # ── Result Card ───────────────────────────────────────────────────────
        match_html = ""
        if ground_truth:
            if pred_class == ground_truth:
                match_html = f'<div class="match-success">✅ Matches dataset label: {ground_truth}</div>'
            else:
                match_html = f'<div class="match-fail">⚠️ Mismatch — dataset label: {ground_truth}</div>'

        st.markdown(f"""
<div class="result-card">
    <div style="font-size:0.9rem;color:#9CA3AF;margin-bottom:0.25rem;">PREDICTED MODULATION</div>
    <div class="pred-label">{pred_class}</div>
    <div style="color:#E2E8F0;font-size:1.1rem;margin-top:0.25rem;">
        Confidence: <strong>{confidence:.1f}%</strong>
        {f" &nbsp;·&nbsp; SNR: <strong>{selected_snr} dB</strong>" if selected_snr is not None else ""}
    </div>
    {match_html}
</div>
""", unsafe_allow_html=True)

        # ── Confidence + Entropy ──────────────────────────────────────────────
        col_score, col_ent = st.columns([1.5, 1])
        with col_score:
            st.plotly_chart(plot_class_scores(CLASSES, probs, pred_class), use_container_width=True)
        with col_ent:
            st.plotly_chart(plot_entropy_gauge(entropy, MAX_ENTROPY), use_container_width=True)
            if entropy > MAX_ENTROPY * 0.75:
                st.warning("⚠️ **High entropy** — signal may be severely corrupted or out-of-distribution.")
            elif entropy > MAX_ENTROPY * 0.4:
                st.info("ℹ️ Moderate uncertainty — check SNR level.")
            else:
                st.success("✅ High confidence classification.")

        st.divider()

        # ── Signal Visualisations ─────────────────────────────────────────────
        st.subheader("📊 Signal Analysis")
        viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs(
            ["🔭 Constellation", "📈 Time Domain", "📡 PSD / FFT", "🌊 Spectrogram"]
        )

        with viz_tab1:
            ref = CONSTELLATION_REFS.get(pred_class)
            fig_const = plot_constellation(signal_iq, pred_class, ref)
            st.plotly_chart(fig_const, use_container_width=True)
            st.caption("White crosses show theoretical ideal symbol locations. Spread indicates noise.")

        with viz_tab2:
            fig_td = plot_time_domain(signal_iq)
            st.plotly_chart(fig_td, use_container_width=True)

        with viz_tab3:
            freqs, psd_db = compute_psd(signal_iq)
            fig_psd = plot_psd(freqs, psd_db)
            st.plotly_chart(fig_psd, use_container_width=True)

        with viz_tab4:
            freqs_s, times_s, Zxx_db = compute_spectrogram(signal_iq, nperseg=16)
            fig_spec = plot_spectrogram(times_s, freqs_s, Zxx_db)
            st.plotly_chart(fig_spec, use_container_width=True)

        st.divider()

        # ── Expert Features ───────────────────────────────────────────────────
        with st.expander("🔍 Expert Signal Features", expanded=False):
            feats = extract_features(signal_iq)
            f_col1, f_col2 = st.columns(2)
            items = list(feats.items())
            half = len(items) // 2
            with f_col1:
                for k, v in items[:half]:
                    st.metric(k.replace("_", " "), f"{v:.5f}")
            with f_col2:
                for k, v in items[half:]:
                    st.metric(k.replace("_", " "), f"{v:.5f}")

        st.caption("Model outputs are softmax scores and not calibrated probabilities. "
                   "Saved samples may overlap with training data.")

        if st.button("🔄 Reset", key="reset_btn"):
            st.session_state["classify"] = False
            st.rerun()
else:
    st.info("👆 Select a sample from the library or upload a signal file, then click **Classify Signal**.")
