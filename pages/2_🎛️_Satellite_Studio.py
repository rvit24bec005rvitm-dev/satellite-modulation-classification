"""
pages/2_🎛️_Satellite_Studio.py
Synthetic Satellite Signal Studio with real-time channel impairments:
  - Modulation selection (BPSK, QPSK, 8PSK, QAM16, 16APSK, 32APSK, GMSK, BFSK)
  - AWGN SNR slider
  - LEO Doppler shift
  - Carrier Frequency Offset (CFO)
  - Phase noise
  - Rician K-factor fading
  - TWTA Saleh AM/AM distortion
All visualisations update automatically from the sidebar sliders.
"""

from pathlib import Path
import sys

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.signal_processing import (
    generate_iq_signal, compute_psd, compute_spectrogram,
    shannon_entropy, CONSTELLATION_REFS,
    normalize_signal,
)
from utils.plotting import (
    plot_constellation, plot_time_domain, plot_psd,
    plot_spectrogram, plot_class_scores, plot_entropy_gauge,
)
from utils.model_backend import ModelBackend

# ── Page config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Satellite Studio · SatModClass",
    page_icon="🎛️",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html,[class*="css"]{ font-family:'Inter',sans-serif; }
.studio-header{
    background:linear-gradient(135deg,#0A0E1A 0%,#0C1E3E 100%);
    border:1px solid #1F3A5F;border-radius:14px;
    padding:1.5rem 2rem;margin-bottom:1.5rem;
}
.imp-badge{
    display:inline-block;padding:0.2rem 0.65rem;border-radius:999px;
    font-size:0.75rem;font-weight:600;margin:0.1rem;
    background:rgba(0,212,255,0.08);color:#00D4FF;
    border:1px solid rgba(0,212,255,0.25);
}
.imp-badge.active{
    background:rgba(0,212,255,0.2);color:#00D4FF;
    border-color:#00D4FF;
}
</style>
""", unsafe_allow_html=True)

CLASSES = ["BPSK", "QPSK", "QAM16"]
MAX_ENTROPY = np.log2(3)

# ── Load model ──────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    onnx = ROOT / "models" / "satellite_modulation_cnn.onnx"
    keras = ROOT / "satellite_modulation_cnn_all_snr.keras"
    if onnx.exists():
        return ModelBackend.load(onnx)
    elif keras.exists():
        return ModelBackend.load(keras)
    return None

@st.cache_data
def load_stats():
    p = ROOT / "normalization_stats_all_snr.npz"
    if not p.exists():
        return None, None
    with np.load(p) as d:
        return float(d["mean"]), float(d["std"])

model = load_model()
mean_val, std_val = load_stats()

# ── Header ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="studio-header">
    <h2 style="margin:0;color:#00D4FF;">🎛️ Satellite Signal Studio</h2>
    <p style="color:#94A3B8;margin:0.5rem 0 0 0;">
        Synthesize realistic satellite baseband I/Q signals with full LEO channel impairments.
        Adjust sliders to see how each effect corrupts the constellation and affects classification.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar controls ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎚️ Signal Parameters")

    MOD_DISPLAY = {
        "BPSK":   "BPSK   (Binary PSK)",
        "QPSK":   "QPSK   (Quadrature PSK)",
        "8PSK":   "8-PSK  (DVB-S2 standard)",
        "QAM16":  "16-QAM (Square QAM)",
        "16APSK": "16-APSK (DVB-S2 broadcast)",
        "32APSK": "32-APSK (DVB-S2X high throughput)",
        "GMSK":   "GMSK   (CubeSat / AIS)",
        "BFSK":   "BFSK   (CubeSat telemetry)",
    }
    mod_key = st.selectbox(
        "Modulation Scheme",
        list(MOD_DISPLAY.keys()),
        format_func=lambda k: MOD_DISPLAY[k],
    )

    snr_db = st.slider("AWGN SNR (dB)", -20, 30, 10, 2)
    rrc_alpha = st.slider("RRC Roll-off Factor α", 0.10, 0.50, 0.35, 0.05)
    sps = st.slider("Samples per Symbol", 4, 16, 8, 2)

    st.markdown("---")
    st.markdown("### 🛰️ LEO Channel Impairments")

    doppler_hz = st.slider(
        "Doppler Shift (norm.)", 0.000, 0.050, 0.000, 0.001,
        help="Normalised Doppler frequency offset from LEO satellite motion"
    )
    cfo_hz = st.slider(
        "Carrier Freq. Offset (CFO)", 0.000, 0.030, 0.000, 0.001,
        help="Receiver local oscillator drift"
    )
    phase_noise = st.slider(
        "Phase Noise Std (rad)", 0.00, 0.50, 0.00, 0.01,
        help="Additive phase noise from LO phase jitter"
    )
    rician_k = st.slider(
        "Rician K-factor", 0.0, 20.0, 0.0, 0.5,
        help="0 = Rayleigh (scattered), >10 ≈ strong LOS satellite link"
    )
    twta_gain = st.slider(
        "TWTA Saleh α_a (HPA compression)", 0.00, 1.00, 0.00, 0.05,
        help="0 = linear amp, >0.3 = strong non-linear TWTA saturation"
    )

    st.markdown("---")
    st.markdown("### ⚙️ Options")
    rng_seed = st.number_input("Random seed", 0, 9999, 42)
    auto_classify = st.toggle("Auto-classify on change", value=True)

# ── Active impairment badges ────────────────────────────────────────────────────
active_imps = []
if doppler_hz > 0:   active_imps.append("LEO Doppler")
if cfo_hz > 0:       active_imps.append("CFO")
if phase_noise > 0:  active_imps.append("Phase Noise")
if rician_k > 0:     active_imps.append("Rician Fading")
if twta_gain > 0:    active_imps.append("TWTA Distortion")

imp_html = " ".join(
    f'<span class="imp-badge{"  active" if imp in active_imps else ""}">{imp}</span>'
    for imp in ["LEO Doppler", "CFO", "Phase Noise", "Rician Fading", "TWTA Distortion"]
)
st.markdown(f"**Active Impairments:** {imp_html}", unsafe_allow_html=True)

# ── Generate signal ─────────────────────────────────────────────────────────────
with st.spinner("Generating signal…"):
    iq = generate_iq_signal(
        modulation=mod_key,
        num_samples=128,
        sps=sps,
        snr_db=snr_db,
        rrc_alpha=rrc_alpha,
        doppler_hz=doppler_hz,
        cfo_hz=cfo_hz,
        rician_k=rician_k,
        twta_alpha_a=twta_gain,
        phase_noise_std=phase_noise,
        rng_seed=int(rng_seed),
    )

st.caption(f"Generated `{mod_key}` signal — shape `{iq.shape}` — SNR `{snr_db} dB`")

# ── Classify ────────────────────────────────────────────────────────────────────
probs = None
pred_class = None

if model and mean_val is not None and auto_classify:
    try:
        iq_norm = normalize_signal(iq, mean_val, std_val)
        probs = model.predict(iq_norm)
        pred_idx = int(np.argmax(probs))
        pred_class = CLASSES[pred_idx]
        entropy = shannon_entropy(probs)
    except Exception as e:
        st.warning(f"Classification error: {e}")

# ── Two-column layout: Info + Result ────────────────────────────────────────────
col_info, col_result = st.columns([1.5, 1])

with col_info:
    # signal stats
    I, Q = iq[:, 0], iq[:, 1]
    A = np.sqrt(I**2 + Q**2)
    s_pwr = np.mean(A**2)
    papr = np.max(A**2) / (s_pwr + 1e-30)
    
    st.markdown("### Signal Statistics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Signal Power", f"{10*np.log10(s_pwr+1e-30):.1f} dBW")
    m2.metric("Mean Amplitude", f"{np.mean(A):.4f}")
    m3.metric("PAPR", f"{10*np.log10(papr):.2f} dB")
    m4.metric("Phase Std", f"{np.degrees(np.std(np.arctan2(Q,I))):.1f}°")

with col_result:
    if probs is not None:
        st.markdown("### CNN Prediction")
        color_map = {"BPSK":"#00D4FF","QPSK":"#7C3AED","QAM16":"#F59E0B"}
        c = color_map.get(pred_class, "#E2E8F0")
        st.markdown(
            f'<div style="font-size:2rem;font-weight:800;color:{c};">🎯 {pred_class}</div>'
            f'<div style="color:#9CA3AF;font-size:0.9rem;">Confidence: <strong style="color:{c};">'
            f'{float(probs[int(np.argmax(probs))])*100:.1f}%</strong></div>',
            unsafe_allow_html=True,
        )
        match_lbl = "✅ Correct" if pred_class == mod_key.upper().replace("-","") else f"⚠️ Model says {pred_class}"
        if mod_key in CLASSES or mod_key == "QAM16":
            expected = mod_key if mod_key in CLASSES else "QAM16"
            match_ok = pred_class == expected
        else:
            match_ok = None
        if match_ok is True:   st.success("✅ Matches generated modulation")
        elif match_ok is False: st.warning(f"⚠️ Mismatch — generated {mod_key}, predicted {pred_class}")
        else:
            st.info(f"ℹ️ {mod_key} is outside trained classes (BPSK/QPSK/QAM16)")
    else:
        st.info("Auto-classify is off or model unavailable.")

st.divider()

# ── Visualisation tabs ─────────────────────────────────────────────────────────
viz1, viz2, viz3, viz4 = st.tabs([
    "🔭 Constellation", "📈 Time Domain", "📡 PSD", "🌊 Spectrogram"
])

with viz1:
    ref = CONSTELLATION_REFS.get(mod_key)
    fig = plot_constellation(iq, pred_class or mod_key, ref)
    st.plotly_chart(fig, use_container_width=True)

with viz2:
    st.plotly_chart(plot_time_domain(iq), use_container_width=True)

with viz3:
    freqs, psd_db = compute_psd(iq)
    st.plotly_chart(plot_psd(freqs, psd_db), use_container_width=True)

with viz4:
    freqs_s, times_s, Zxx_db = compute_spectrogram(iq, nperseg=16)
    st.plotly_chart(plot_spectrogram(times_s, freqs_s, Zxx_db), use_container_width=True)

# ── Score + Entropy row ────────────────────────────────────────────────────────
if probs is not None:
    st.divider()
    sc_col, ent_col = st.columns([1.5, 1])
    with sc_col:
        st.plotly_chart(plot_class_scores(CLASSES, probs, pred_class), use_container_width=True)
    with ent_col:
        st.plotly_chart(plot_entropy_gauge(entropy, MAX_ENTROPY), use_container_width=True)
        if entropy > MAX_ENTROPY * 0.75:
            st.error("🔴 Very high uncertainty — try increasing SNR or disabling strong impairments.")
        elif entropy > MAX_ENTROPY * 0.4:
            st.warning("🟡 Moderate uncertainty")
        else:
            st.success("🟢 High confidence")

st.divider()
st.markdown("""
**Channel Physics Reference**

| Impairment | Effect on Constellation |
|:---|:---|
| Low SNR | Symbol cluster spreading / overlap |
| Doppler Shift | Constellation rotation at constant angular rate |
| CFO | Slow constellation spin (phase ramp) |
| Phase Noise | Fuzzy arc-shaped smearing of clusters |
| Rician Fading | Amplitude variation + scatter around ideal symbols |
| TWTA Saturation | Amplitude compression → outer symbols pulled inward |
""")
