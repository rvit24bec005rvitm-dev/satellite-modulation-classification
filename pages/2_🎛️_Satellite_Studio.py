"""
pages/2_🎛️_Satellite_Studio.py
Synthetic Satellite Signal Studio with real-time channel impairments:
  - Modulation selection (BPSK, QPSK, 8PSK, QAM16, 16APSK, 32APSK, GMSK, BFSK)
  - AWGN SNR slider & presets
  - LEO Doppler shift, Carrier Frequency Offset (CFO), Phase noise
  - Rician K-factor fading & TWTA Saleh non-linear distortion
  - 2D & 3D Constellation spirals, Eye Diagram, and Audible RF demodulator
"""

from pathlib import Path
import sys, time

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.signal_processing import (
    generate_iq_signal, compute_psd, compute_spectrogram,
    shannon_entropy, CONSTELLATION_REFS,
    normalize_signal, generate_audio_waveform, compute_eye_diagram,
)
from utils.plotting import (
    plot_constellation, plot_constellation_3d, plot_time_domain, plot_psd,
    plot_spectrogram, plot_class_scores, plot_entropy_gauge, plot_eye_diagram,
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
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;600;700;800&display=swap');

html, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; }
.mono-font { font-family: 'JetBrains Mono', monospace !important; }

.studio-header {
    background: linear-gradient(135deg, rgba(10, 14, 26, 0.95) 0%, rgba(12, 30, 62, 0.85) 100%);
    border: 1px solid rgba(0, 212, 255, 0.35);
    border-radius: 16px;
    padding: 1.75rem 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
}
.imp-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 700;
    margin: 0.15rem;
    background: rgba(30, 41, 59, 0.7);
    color: #64748B;
    border: 1px solid rgba(51, 65, 85, 0.8);
    transition: all 0.2s ease;
}
.imp-badge.active {
    background: rgba(0, 212, 255, 0.18);
    color: #00D4FF;
    border-color: #00D4FF;
    box-shadow: 0 0 10px rgba(0, 212, 255, 0.3);
}
.stat-pill {
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid #1E293B;
    border-radius: 10px;
    padding: 0.85rem 1rem;
    text-align: center;
}
.stat-val {
    font-size: 1.4rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: #38BDF8;
}
.stat-lbl {
    font-size: 0.75rem;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-top: 0.2rem;
}
</style>
""", unsafe_allow_html=True)

CLASSES = ["BPSK", "QPSK", "QAM16"]
MAX_ENTROPY = np.log2(3)

# ── Load model ──────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading classification model…")
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
    <h2 style="margin:0; color:#00D4FF;">🎛️ Satellite RF Signal Studio & Channel Laboratory</h2>
    <p style="color:#94A3B8; margin:0.5rem 0 0 0; font-size:1.02rem; line-height:1.6;">
        Synthesize precision spaceborne I/Q baseband signals under physical LEO channel impairments.
        Adjust orbital Doppler shifts, oscillator drifts, TWTA amplifier gain compression, and fading to watch constellations warp and hear audio demodulation in real time.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar Controls ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛰️ Channel Scenarios & Presets")
    
    preset_choice = st.selectbox(
        "Load Mission Scenario",
        [
            "Custom User Defined",
            "🚀 Clean LEO Pass (High SNR)",
            "⚡ TWTA Saturated Transponder (16-APSK)",
            "🌪️ Severe LEO Doppler & Scintillation",
            "🪐 Deep Space Noise Barrier (-6dB)",
            "📡 Starlink Ka-Band High Throughput",
        ],
        index=0,
    )

    # Defaults
    def_mod = "QPSK"
    def_snr = 12
    def_dop = 0.000
    def_cfo = 0.000
    def_pn = 0.00
    def_ric = 0.0
    def_twta = 0.00

    if preset_choice == "🚀 Clean LEO Pass (High SNR)":
        def_mod, def_snr, def_dop, def_cfo, def_pn, def_ric, def_twta = "QPSK", 18, 0.001, 0.000, 0.00, 15.0, 0.0
    elif preset_choice == "⚡ TWTA Saturated Transponder (16-APSK)":
        def_mod, def_snr, def_dop, def_cfo, def_pn, def_ric, def_twta = "16APSK", 15, 0.000, 0.000, 0.02, 0.0, 0.55
    elif preset_choice == "🌪️ Severe LEO Doppler & Scintillation":
        def_mod, def_snr, def_dop, def_cfo, def_pn, def_ric, def_twta = "QPSK", 4, 0.028, 0.012, 0.15, 1.0, 0.25
    elif preset_choice == "🪐 Deep Space Noise Barrier (-6dB)":
        def_mod, def_snr, def_dop, def_cfo, def_pn, def_ric, def_twta = "BPSK", -6, 0.000, 0.000, 0.00, 0.0, 0.0
    elif preset_choice == "📡 Starlink Ka-Band High Throughput":
        def_mod, def_snr, def_dop, def_cfo, def_pn, def_ric, def_twta = "QAM16", 16, 0.006, 0.002, 0.05, 12.0, 0.20

    st.markdown("---")
    st.markdown("### 🎚️ Baseband Signal Parameters")

    MOD_DISPLAY = {
        "BPSK":   "BPSK   (Binary PSK - Deep Space)",
        "QPSK":   "QPSK   (Quadrature PSK - LEO/ISS)",
        "8PSK":   "8-PSK  (DVB-S2 satellite)",
        "QAM16":  "16-QAM (Square QAM - Broadband)",
        "16APSK": "16-APSK (DVB-S2 Satellite Broadcast)",
        "32APSK": "32-APSK (DVB-S2X High Throughput)",
        "GMSK":   "GMSK   (CubeSat / Marine AIS)",
        "BFSK":   "BFSK   (Spacecraft Telemetry Beacon)",
    }
    
    mod_keys = list(MOD_DISPLAY.keys())
    mod_idx = mod_keys.index(def_mod) if def_mod in mod_keys else 1
    mod_key = st.selectbox(
        "Modulation Scheme",
        mod_keys,
        index=mod_idx,
        format_func=lambda k: MOD_DISPLAY[k],
    )

    snr_db = st.slider("AWGN SNR (dB)", -20, 30, int(def_snr), 2)
    rrc_alpha = st.slider("RRC Roll-off Factor α", 0.10, 0.50, 0.35, 0.05)
    sps = st.slider("Samples per Symbol", 4, 16, 8, 2)

    st.markdown("---")
    st.markdown("### 🛰️ Space Channel Impairments")

    doppler_hz = st.slider(
        "Doppler Shift (norm.)", 0.000, 0.050, float(def_dop), 0.001,
        help="Simulates relative orbital velocity of LEO satellite"
    )
    cfo_hz = st.slider(
        "Carrier Frequency Offset (CFO)", 0.000, 0.030, float(def_cfo), 0.001,
        help="Spacecraft receiver local oscillator drift from thermal gradient"
    )
    phase_noise = st.slider(
        "Phase Noise Std (rad)", 0.00, 0.50, float(def_pn), 0.01,
        help="Additive phase jitter from local oscillator"
    )
    rician_k = st.slider(
        "Rician K-factor", 0.0, 20.0, float(def_ric), 0.5,
        help="0 = Rayleigh (multipath), >10 = strong Line-of-Sight space link"
    )
    twta_gain = st.slider(
        "TWTA Saleh Compression (α_a)", 0.00, 1.00, float(def_twta), 0.05,
        help="0 = linear amplifier, >0.3 = strong non-linear saturation"
    )

    st.markdown("---")
    rng_seed = st.number_input("Random seed", 0, 9999, 42)
    auto_classify = st.toggle("Auto-classify with CNN", value=True)

# ── Active Impairment Badges ───────────────────────────────────────────────────
active_imps = []
if doppler_hz > 0:   active_imps.append("LEO Doppler")
if cfo_hz > 0:       active_imps.append("CFO")
if phase_noise > 0:  active_imps.append("Phase Noise")
if rician_k > 0:     active_imps.append("Rician Fading")
if twta_gain > 0:    active_imps.append("TWTA Distortion")

badge_names = ["LEO Doppler", "CFO", "Phase Noise", "Rician Fading", "TWTA Distortion"]
badges_html = " ".join(
    f'<span class="imp-badge{" active" if b in active_imps else ""}">{b}</span>'
    for b in badge_names
)
st.markdown(f"**Channel Impairments Active:** &nbsp; {badges_html}", unsafe_allow_html=True)

# ── Generate Synthetic Signal ──────────────────────────────────────────────────
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

# ── Run CNN Classification ────────────────────────────────────────────────────
probs = None
pred_class = None
entropy = 0.0

if model and mean_val is not None and auto_classify:
    try:
        iq_norm = normalize_signal(iq, mean_val, std_val)
        probs = model.predict(iq_norm)
        pred_idx = int(np.argmax(probs))
        pred_class = CLASSES[pred_idx]
        entropy = shannon_entropy(probs)
    except Exception as e:
        st.warning(f"Classification note: {e}")

# ── Stats & Audio Row ─────────────────────────────────────────────────────────
I, Q = iq[:, 0], iq[:, 1]
A = np.sqrt(I**2 + Q**2)
s_pwr = np.mean(A**2)
papr = np.max(A**2) / (s_pwr + 1e-30)

col_telemetry, col_audio_box, col_pred_box = st.columns([1.5, 1, 1.2])

with col_telemetry:
    st.markdown("##### 📡 Baseband Telemetry Metrics")
    st_c1, st_c2, st_c3, st_c4 = st.columns(4)
    with st_c1:
        st.markdown(f'<div class="stat-pill"><div class="stat-val">{10*np.log10(s_pwr+1e-30):.1f}</div><div class="stat-lbl">Power (dBW)</div></div>', unsafe_allow_html=True)
    with st_c2:
        st.markdown(f'<div class="stat-pill"><div class="stat-val">{np.mean(A):.3f}</div><div class="stat-lbl">Mean |A|</div></div>', unsafe_allow_html=True)
    with st_c3:
        st.markdown(f'<div class="stat-pill"><div class="stat-val">{10*np.log10(papr):.2f}</div><div class="stat-lbl">PAPR (dB)</div></div>', unsafe_allow_html=True)
    with st_c4:
        st.markdown(f'<div class="stat-pill"><div class="stat-val">{np.degrees(np.std(np.arctan2(Q,I))):.0f}°</div><div class="stat-lbl">Phase Std</div></div>', unsafe_allow_html=True)

with col_audio_box:
    st.markdown("##### 🔊 Audible RF Demodulation")
    audio_wav = generate_audio_waveform(iq, carrier_tone_hz=880.0, duration_sec=1.5)
    st.audio(audio_wav, format="audio/wav")
    st.caption("Drag Doppler or CFO to hear the frequency pitch shift!")

with col_pred_box:
    if probs is not None:
        st.markdown("##### 🎯 CNN Modulation Lock")
        color_map = {"BPSK": "#00D4FF", "QPSK": "#7C3AED", "QAM16": "#F59E0B"}
        c_hex = color_map.get(pred_class, "#38BDF8")
        conf_pct = float(probs[int(np.argmax(probs))]) * 100.0
        
        st.markdown(f"""
        <div style="background:rgba(15,23,42,0.85); border:1px solid {c_hex}44; border-radius:10px; padding:0.75rem 1rem;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:1.6rem; font-weight:800; color:{c_hex};">{pred_class}</span>
                <span style="font-family:'JetBrains Mono'; font-weight:700; color:#34D399;">{conf_pct:.1f}%</span>
            </div>
            <div style="font-size:0.78rem; color:#94A3B8; margin-top:0.2rem;">
                Target: <strong>{mod_key}</strong> &nbsp;·&nbsp; Entropy: <strong>{entropy:.3f} bits</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ── Visualisation Tabs ────────────────────────────────────────────────────────
st.subheader("📊 Multi-Domain RF Laboratory Visualizations")
viz1, viz2, viz3, viz4, viz5, viz6 = st.tabs([
    "🔭 2D Constellation",
    "🌀 3D Phase Space Spiral",
    "👁️ Vector Eye Diagram",
    "📈 Time Domain Waveforms",
    "📡 Power Spectral Density (PSD)",
    "🌊 STFT Spectrogram",
])

with viz1:
    ref = CONSTELLATION_REFS.get(mod_key)
    fig_const = plot_constellation(iq, pred_class or mod_key, ref)
    st.plotly_chart(fig_const, use_container_width=True)

with viz2:
    fig_3d = plot_constellation_3d(iq, pred_class or mod_key)
    st.plotly_chart(fig_3d, use_container_width=True)
    st.caption("Notice how continuous Doppler and CFO unwind into a corkscrew spiral across the temporal axis!")

with viz3:
    t_axis, i_traces, q_traces = compute_eye_diagram(iq, sps=sps)
    fig_eye = plot_eye_diagram(t_axis, i_traces, q_traces)
    st.plotly_chart(fig_eye, use_container_width=True)
    st.caption("Eye diagram opening closure indicates ISI degradation from channel filtering, noise, or amplifier compression.")

with viz4:
    st.plotly_chart(plot_time_domain(iq), use_container_width=True)

with viz5:
    freqs, psd_db = compute_psd(iq)
    st.plotly_chart(plot_psd(freqs, psd_db), use_container_width=True)

with viz6:
    freqs_s, times_s, Zxx_db = compute_spectrogram(iq, nperseg=16)
    st.plotly_chart(plot_spectrogram(times_s, freqs_s, Zxx_db), use_container_width=True)

# ── Confidence & Entropy Row ──────────────────────────────────────────────────
if probs is not None:
    st.divider()
    sc_col, ent_col = st.columns([1.5, 1])
    with sc_col:
        st.plotly_chart(plot_class_scores(CLASSES, probs, pred_class), use_container_width=True)
    with ent_col:
        st.plotly_chart(plot_entropy_gauge(entropy, MAX_ENTROPY), use_container_width=True)
        if entropy > MAX_ENTROPY * 0.75:
            st.error("🔴 **Extreme Uncertainty** — Channel corruption exceeds trained model bounds.")
        elif entropy > MAX_ENTROPY * 0.4:
            st.warning("🟡 **Moderate Uncertainty** — Transitional SNR or moderate impairment.")
        else:
            st.success("🟢 **High Confidence Classification**")

st.divider()

# ── Physical Channel Reference Table ──────────────────────────────────────────
with st.expander("📖 Satellite Channel Impairment Physical Mechanics", expanded=False):
    st.markdown("""
    | Physical Impairment | Aerospace Mechanism | Visual Impact on Constellation |
    |:---|:---|:---|
    | **AWGN Thermal Noise** | Antenna thermal noise & cosmic microwave background | Gaussian dispersion and cluster spreading around ideal coordinates |
    | **LEO Doppler Shift** | High satellite orbital velocity ($v \\approx 7.6\\text{ km/s}$) | Continuous rotation of entire constellation at frequency $f_d$ |
    | **Local Oscillator CFO** | Thermal gradients between sunlit and eclipse orbits | Slow phase rotation (linear phase ramp across time) |
    | **Oscillator Phase Noise** | Crystal phase jitter in local oscillator | Circumferential arc-shaped smearing around origin |
    | **Rician Fading** | Ground reflection multipath + direct Line-of-Sight | Amplitude fluctuation & asymmetric symbol scatter |
    | **TWTA Saturation** | Travelling Wave Tube operating at maximum RF output | Non-linear AM/AM compression (outer symbols pulled inward) |
    """)
