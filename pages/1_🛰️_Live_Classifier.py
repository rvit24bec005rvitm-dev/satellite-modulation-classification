"""
pages/1_🛰️_Live_Classifier.py
Interactive modulation classifier with:
 - RadioML Sample Library & quick filters
 - Space Mission Presets
 - File upload (.npy, .npz, .csv, .mat, .wav)
 - 2D & 3D Constellation, time-domain, PSD, STFT Spectrogram, and Eye Diagram
 - Audible RF Demodulation player
 - Shannon entropy uncertainty gauge & telemetry report export
"""

from pathlib import Path
import sys, io, json, time

import numpy as np
import pandas as pd
import streamlit as st

# ── Path setup ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.signal_processing import (
    normalize_signal, compute_psd, compute_spectrogram,
    shannon_entropy, extract_features, CONSTELLATION_REFS,
    generate_iq_signal, generate_audio_waveform, compute_eye_diagram,
)
from utils.plotting import (
    plot_constellation, plot_constellation_3d, plot_time_domain, plot_psd,
    plot_spectrogram, plot_class_scores, plot_entropy_gauge, plot_eye_diagram,
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
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;600;700;800&display=swap');

html, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; }
.mono-font { font-family: 'JetBrains Mono', monospace !important; }

.result-card {
    background: linear-gradient(135deg, rgba(17, 24, 39, 0.95), rgba(15, 23, 42, 0.95));
    border: 1px solid rgba(0, 212, 255, 0.35);
    border-radius: 16px;
    padding: 1.75rem 2rem;
    margin: 1rem 0;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.1);
}
.pred-label {
    font-size: 3rem;
    font-weight: 800;
    font-family: 'Space Grotesk', sans-serif;
    background: linear-gradient(90deg, #00D4FF, #A78BFA, #38BDF8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
}
.match-success {
    display: inline-flex;
    align-items: center;
    background: rgba(16, 185, 129, 0.15);
    color: #34D399;
    border: 1px solid rgba(16, 185, 129, 0.4);
    border-radius: 9999px;
    padding: 0.3rem 0.85rem;
    font-weight: 700;
    font-size: 0.9rem;
    margin-top: 0.5rem;
}
.match-fail {
    display: inline-flex;
    align-items: center;
    background: rgba(244, 63, 94, 0.15);
    color: #FB7185;
    border: 1px solid rgba(244, 63, 94, 0.4);
    border-radius: 9999px;
    padding: 0.3rem 0.85rem;
    font-weight: 700;
    font-size: 0.9rem;
    margin-top: 0.5rem;
}
.telemetry-pill {
    display: inline-block;
    padding: 0.25rem 0.65rem;
    border-radius: 6px;
    background: #1E293B;
    color: #94A3B8;
    font-size: 0.8rem;
    font-family: 'JetBrains Mono', monospace;
    margin-right: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

st.title("🛰️ Tactical Satellite Signal Classifier")
st.markdown("Real-time baseband I/Q signal classification powered by a trained 1D-CNN with 3D phase-space trajectory inspection, vector eye diagrams, and audible RF demodulation.")

# ── Constants ───────────────────────────────────────────────────────────────────
CLASSES = ["BPSK", "QPSK", "QAM16"]
MAX_ENTROPY = np.log2(len(CLASSES))   # 1.585 bits for 3 classes

# ── Load model ──────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model backend…")
def load_model():
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

# Status bar
c_stat1, c_stat2 = st.columns([3, 1])
with c_stat1:
    st.caption(f"🟢 **Backend Online:** `{model.backend_name}` · **Input Tensor:** `{model.input_shape}` · **Classes:** `{', '.join(CLASSES)}`")
with c_stat2:
    st.markdown('<div style="text-align:right;"><span class="telemetry-pill">FRAME: 128 SAMPLES</span><span class="telemetry-pill">CH: 2 (I/Q)</span></div>', unsafe_allow_html=True)

st.divider()

# ── Signal source selection ─────────────────────────────────────────────────────
tab_sample, tab_presets, tab_upload = st.tabs([
    "📦 RadioML Sample Library",
    "⚡ Space Mission Presets",
    "📂 Upload Custom I/Q File"
])

signal_iq: np.ndarray | None = None
ground_truth: str | None = None
selected_snr: int | None = None
source_name: str = "Unknown"

# ┌── Tab 1: RadioML Sample library ───────────────────────────────────────────┐
with tab_sample:
    if signals_all is None:
        st.warning("⚠️ `classifier_samples.npz` not found.")
    else:
        available_snrs = sorted(int(v) for v in np.unique(snrs_all))
        c1, c2, c3 = st.columns(3)
        with c1:
            sel_class = st.selectbox("Modulation Class", CLASSES, key="sc_class")
        with c2:
            default_idx = available_snrs.index(0) if 0 in available_snrs else 0
            selected_snr = st.selectbox("SNR Level (dB)", available_snrs, index=default_idx, key="sc_snr")
        
        mask = (labels_all == sel_class) & (snrs_all == selected_snr)
        matching = np.where(mask)[0]
        
        if len(matching) == 0:
            st.warning("No samples for this class + SNR combination.")
        else:
            with c3:
                sample_idx = st.number_input("Sample Index", 0, len(matching) - 1, 0, key="sc_idx")
            signal_iq = signals_all[matching[int(sample_idx)]].copy()
            ground_truth = sel_class
            source_name = f"RadioML {sel_class} [{selected_snr} dB, #{sample_idx}]"
            st.caption(f"Shape: `{signal_iq.shape}` — {len(matching)} samples available at {selected_snr} dB")

# ┌── Tab 2: Space Mission Presets ─────────────────────────────────────────────┐
with tab_presets:
    st.markdown("Quickly test realistic operational satellite link scenarios:")
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    
    preset_choice = None
    with col_p1:
        if st.button("🛰️ Starlink Ka-Band\n(16-QAM @ +14dB)", use_container_width=True):
            preset_choice = ("QAM16", 14.0, 0.005, 0.15, "Starlink Ka-Band Downlink")
    with col_p2:
        if st.button("🚀 ISS Crew Telemetry\n(QPSK @ +8dB)", use_container_width=True):
            preset_choice = ("QPSK", 8.0, 0.002, 0.0, "ISS S-Band Telemetry")
    with col_p3:
        if st.button("🪐 Deep Space Probe\n(BPSK @ -2dB)", use_container_width=True):
            preset_choice = ("BPSK", -2.0, 0.0, 0.0, "Deep Space Interplanetary Link")
    with col_p4:
        if st.button("🌪️ LEO Scintillation Pass\n(QPSK @ +2dB)", use_container_width=True):
            preset_choice = ("QPSK", 2.0, 0.025, 0.35, "LEO Ionospheric Scintillation")

    if preset_choice:
        p_mod, p_snr, p_dop, p_twta, p_name = preset_choice
        st.session_state["preset_sig"] = generate_iq_signal(
            modulation=p_mod, num_samples=128, snr_db=p_snr,
            doppler_hz=p_dop, twta_alpha_a=p_twta, rng_seed=42
        )
        st.session_state["preset_gt"] = p_mod
        st.session_state["preset_snr"] = int(p_snr)
        st.session_state["preset_name"] = p_name

    if "preset_sig" in st.session_state:
        signal_iq = st.session_state["preset_sig"]
        ground_truth = st.session_state["preset_gt"]
        selected_snr = st.session_state["preset_snr"]
        source_name = st.session_state["preset_name"]
        st.success(f"Loaded **{source_name}** (`{ground_truth}` @ `{selected_snr} dB`)")

# ┌── Tab 3: File upload ───────────────────────────────────────────────────────┐
with tab_upload:
    st.markdown("Upload a file containing a **single I/Q signal** (shape `(128, 2)` or `(2, 128)`).")
    st.markdown("**Supported file formats:** `.npy` `.npz` `.csv` `.mat` `.wav`")
    
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
                ground_truth = None
                selected_snr = None
                source_name = f"Uploaded `{uploaded.name}`"
                st.success(f"✅ Loaded `{uploaded.name}` → Normalized shape `{signal_iq.shape}`")

        except Exception as e:
            st.error(f"Parse error: {e}")

st.divider()

# ── Classify Execution ──────────────────────────────────────────────────────────
if signal_iq is not None:
    c_btn1, c_btn2 = st.columns([3, 1])
    with c_btn1:
        run_classify = st.button("🚀 Intercept & Classify Signal", type="primary", use_container_width=True)
    with c_btn2:
        auto_run = st.checkbox("Auto-classify", value=True)

    if run_classify or auto_run:
        # Normalise for CNN
        iq_norm = normalize_signal(signal_iq, mean_val, std_val)

        # Model Inference
        t0 = time.perf_counter()
        probs = model.predict(iq_norm)
        inf_latency_ms = (time.perf_counter() - t0) * 1000.0

        pred_idx = int(np.argmax(probs))
        pred_class = CLASSES[pred_idx]
        confidence = float(probs[pred_idx]) * 100.0
        entropy = shannon_entropy(probs)

        # ── Tactical Result Card ───────────────────────────────────────────────
        match_html = ""
        if ground_truth:
            gt_normalized = "QAM16" if ground_truth in ["16-QAM", "QAM16"] else ground_truth
            if pred_class == gt_normalized:
                match_html = f'<div class="match-success">✅ Ground Truth Verified: {ground_truth}</div>'
            else:
                match_html = f'<div class="match-fail">⚠️ Mismatch — Dataset Ground Truth: {ground_truth}</div>'

        st.markdown(f"""
        <div class="result-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div style="font-size:0.85rem; color:#94A3B8; text-transform:uppercase; letter-spacing:0.06em;">
                    PREDICTED MODULATION SCHEME &nbsp;|&nbsp; {source_name}
                </div>
                <div style="font-family:'JetBrains Mono'; font-size:0.85rem; color:#00D4FF;">
                    LATENCY: {inf_latency_ms:.2f} ms
                </div>
            </div>
            <div class="pred-label">{pred_class}</div>
            <div style="color:#E2E8F0; font-size:1.15rem; margin-top:0.35rem;">
                Confidence: <strong style="color:#00D4FF;">{confidence:.1f}%</strong>
                {f" &nbsp;·&nbsp; Channel SNR: <strong>{selected_snr} dB</strong>" if selected_snr is not None else ""}
                &nbsp;·&nbsp; Entropy: <strong>{entropy:.3f} bits</strong>
            </div>
            {match_html}
        </div>
        """, unsafe_allow_html=True)

        # ── Confidence Gauge + Audio Heterodyne Demodulator ────────────────────
        col_score, col_ent, col_audio = st.columns([1.2, 1, 1])
        with col_score:
            st.plotly_chart(plot_class_scores(CLASSES, probs, pred_class), use_container_width=True)
        with col_ent:
            st.plotly_chart(plot_entropy_gauge(entropy, MAX_ENTROPY), use_container_width=True)
        with col_audio:
            st.markdown("#### 🔊 RF Demodulated Audio")
            st.markdown("Listen to the audible baseband heterodyne tone:")
            audio_bytes = generate_audio_waveform(signal_iq, carrier_tone_hz=880.0, duration_sec=1.5)
            st.audio(audio_bytes, format="audio/wav")
            
            # Uncertainty indicator
            if entropy > MAX_ENTROPY * 0.75:
                st.warning("⚠️ **Severe Uncertainty** — SNR may be below -8 dB or heavy multipath fading.")
            elif entropy > MAX_ENTROPY * 0.4:
                st.info("ℹ️ **Moderate Uncertainty** — Transitional SNR band.")
            else:
                st.success("🟢 **High-Confidence Lock** — Clean constellation clustering.")

        st.divider()

        # ── Multi-Dimensional Signal Visualizations ────────────────────────────
        st.subheader("📊 Tactical Signal & Phase Space Analysis")
        viz_tab1, viz_tab2, viz_tab3, viz_tab4, viz_tab5, viz_tab6 = st.tabs([
            "🔭 2D Constellation",
            "🌀 3D Phase Space (I/Q/Time)",
            "👁️ Vector Eye Diagram",
            "📈 Time Domain (I & Q)",
            "📡 PSD Spectrum",
            "🌊 STFT Spectrogram",
        ])

        with viz_tab1:
            ref = CONSTELLATION_REFS.get(pred_class)
            fig_const = plot_constellation(signal_iq, pred_class, ref)
            st.plotly_chart(fig_const, use_container_width=True)
            st.caption("White crosses denote ideal symbol coordinate centroids. Cluster dispersion indicates channel noise variance.")

        with viz_tab2:
            fig_3d = plot_constellation_3d(signal_iq, pred_class)
            st.plotly_chart(fig_3d, use_container_width=True)
            st.caption("Rotate and zoom this 3D trajectory to inspect how symbols evolve across sample index time, revealing phase rotation (Doppler) or cluster drift.")

        with viz_tab3:
            t_axis, i_traces, q_traces = compute_eye_diagram(signal_iq, sps=8)
            fig_eye = plot_eye_diagram(t_axis, i_traces, q_traces)
            st.plotly_chart(fig_eye, use_container_width=True)
            st.caption("Eye diagram opening height correlates with noise margin; eye width opening correlates with timing jitter margin.")

        with viz_tab4:
            fig_td = plot_time_domain(signal_iq)
            st.plotly_chart(fig_td, use_container_width=True)

        with viz_tab5:
            freqs, psd_db = compute_psd(signal_iq)
            fig_psd = plot_psd(freqs, psd_db)
            st.plotly_chart(fig_psd, use_container_width=True)

        with viz_tab6:
            freqs_s, times_s, Zxx_db = compute_spectrogram(signal_iq, nperseg=16)
            fig_spec = plot_spectrogram(times_s, freqs_s, Zxx_db)
            st.plotly_chart(fig_spec, use_container_width=True)

        st.divider()

        # ── Expert Features & Telemetry Export ────────────────────────────────
        exp_col1, exp_col2 = st.columns([2, 1])
        with exp_col1:
            with st.expander("🔍 Expert Statistical RF Features", expanded=False):
                feats = extract_features(signal_iq)
                f_cols = st.columns(3)
                items = list(feats.items())
                chunk = len(items) // 3 + 1
                for i, col in enumerate(f_cols):
                    with col:
                        for k, v in items[i*chunk:(i+1)*chunk]:
                            st.metric(k.replace("_", " "), f"{v:.5f}")

        with exp_col2:
            # Export classification report
            report_data = {
                "predicted_modulation": pred_class,
                "confidence_percent": confidence,
                "entropy_bits": entropy,
                "ground_truth": ground_truth,
                "snr_db": selected_snr,
                "probabilities": {cls: float(p) for cls, p in zip(CLASSES, probs)},
                "signal_shape": list(signal_iq.shape),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            json_report = json.dumps(report_data, indent=2).encode("utf-8")
            st.download_button(
                "⬇️ Export Telemetry JSON Report",
                data=json_report,
                file_name=f"satmod_telemetry_{pred_class}_{int(time.time())}.json",
                mime="application/json",
                use_container_width=True,
            )

else:
    st.info("👆 Select a sample from the library, pick a space mission preset, or upload an I/Q file to classify.")
