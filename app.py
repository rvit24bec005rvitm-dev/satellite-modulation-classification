"""
app.py — Landing page & navigation hub
Satellite Modulation Classification System
"""

from pathlib import Path
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SatModClass · AI Satellite Modulation Classifier",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inline custom CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hero gradient banner */
.hero-banner {
    background: linear-gradient(135deg, #0A0E1A 0%, #0C1E3E 40%, #091428 100%);
    border: 1px solid #1F3A5F;
    border-radius: 16px;
    padding: 3rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at 70% 40%, rgba(0,212,255,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00D4FF 0%, #7C3AED 50%, #00D4FF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.15;
    margin-bottom: 0.75rem;
}
.hero-sub {
    font-size: 1.1rem;
    color: #94A3B8;
    max-width: 680px;
    line-height: 1.7;
}
.badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-right: 0.5rem;
    margin-top: 1rem;
}
.badge-cyan  { background: rgba(0,212,255,0.12); color:#00D4FF; border:1px solid rgba(0,212,255,0.3); }
.badge-purple{ background: rgba(124,58,237,0.12); color:#A78BFA; border:1px solid rgba(124,58,237,0.3); }
.badge-green { background: rgba(16,185,129,0.12); color:#34D399; border:1px solid rgba(16,185,129,0.3); }
.badge-amber { background: rgba(245,158,11,0.12); color:#FCD34D; border:1px solid rgba(245,158,11,0.3); }

/* Feature cards */
.feat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 1.25rem;
    margin: 1.5rem 0;
}
.feat-card {
    background: linear-gradient(145deg, #111827, #0F172A);
    border: 1px solid #1F2937;
    border-radius: 12px;
    padding: 1.5rem;
    transition: border-color 0.2s, transform 0.2s;
}
.feat-card:hover {
    border-color: #00D4FF;
    transform: translateY(-2px);
}
.feat-icon { font-size: 1.8rem; margin-bottom: 0.6rem; }
.feat-title { font-weight: 700; font-size: 1.05rem; color: #E2E8F0; margin-bottom: 0.35rem; }
.feat-desc  { font-size: 0.88rem; color: #9CA3AF; line-height: 1.55; }

/* Metric highlights */
.metric-row {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin: 1rem 0;
}
.metric-box {
    flex: 1;
    min-width: 140px;
    background: #111827;
    border: 1px solid #1F2937;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    text-align: center;
}
.metric-val {
    font-size: 1.8rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}
.metric-label { font-size: 0.8rem; color: #9CA3AF; margin-top: 0.2rem; }

/* Nav cards */
.nav-card {
    background: #111827;
    border: 1px solid #1F2937;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
    transition: border-color 0.2s;
    cursor: pointer;
}
.nav-card:hover { border-color: #00D4FF; }

/* Sidebar tweak */
section[data-testid="stSidebar"] { background: #0D1117; }
</style>
""", unsafe_allow_html=True)

# ── Hero Banner ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">📡 SatModClass</div>
    <div class="hero-title" style="font-size:1.8rem; margin-top:-0.5rem;">
        AI-Based Satellite Modulation Classification
    </div>
    <div class="hero-sub">
        A deep-learning platform that classifies satellite communication modulation schemes
        (BPSK, QPSK, 16-QAM) from raw I/Q baseband signals — with full interactive signal
        analysis, synthetic channel simulation, and explainable AI.
    </div>
    <div>
        <span class="badge badge-cyan">📡 3 Modulation Classes</span>
        <span class="badge badge-purple">🧠 1D CNN · ResNet</span>
        <span class="badge badge-green">✅ 72.4% → 100% @+6dB</span>
        <span class="badge badge-amber">🛰️ DVB-S2 Satellite Focus</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Top Metrics ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="metric-row">
    <div class="metric-box">
        <div class="metric-val" style="color:#00D4FF;">72.4%</div>
        <div class="metric-label">Overall Accuracy</div>
    </div>
    <div class="metric-box">
        <div class="metric-val" style="color:#34D399;">100%</div>
        <div class="metric-label">At +6dB SNR</div>
    </div>
    <div class="metric-box">
        <div class="metric-val" style="color:#A78BFA;">3</div>
        <div class="metric-label">Modulation Classes</div>
    </div>
    <div class="metric-box">
        <div class="metric-val" style="color:#FCD34D;">20</div>
        <div class="metric-label">SNR Levels Tested</div>
    </div>
    <div class="metric-box">
        <div class="metric-val" style="color:#F87171;">~20K</div>
        <div class="metric-label">Model Parameters</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Feature Cards ──────────────────────────────────────────────────────────────
st.subheader("🚀 Platform Features")
st.markdown("""
<div class="feat-grid">
    <div class="feat-card">
        <div class="feat-icon">🛰️</div>
        <div class="feat-title">Live Classifier</div>
        <div class="feat-desc">Select or upload I/Q signals and classify modulation schemes in real-time with confidence gauges and XAI analysis.</div>
    </div>
    <div class="feat-card">
        <div class="feat-icon">🎛️</div>
        <div class="feat-title">Satellite Signal Studio</div>
        <div class="feat-desc">Generate synthetic signals with LEO Doppler shift, CFO, Rician fading, and TWTA non-linearity impairments.</div>
    </div>
    <div class="feat-card">
        <div class="feat-icon">📊</div>
        <div class="feat-title">Model Evaluation</div>
        <div class="feat-desc">Interactive confusion matrix, per-class F1 scores, and precision-recall curves across all SNR levels.</div>
    </div>
    <div class="feat-card">
        <div class="feat-icon">🔬</div>
        <div class="feat-title">Signal Visualizer</div>
        <div class="feat-desc">I/Q constellation diagrams, time-domain waveforms, PSD/FFT spectra, and STFT spectrograms.</div>
    </div>
    <div class="feat-card">
        <div class="feat-icon">📖</div>
        <div class="feat-title">Satellite RF Theory</div>
        <div class="feat-desc">Interactive educational guide covering modulation fundamentals, DVB-S2, APSK, and channel effects.</div>
    </div>
    <div class="feat-card">
        <div class="feat-icon">🤖</div>
        <div class="feat-title">Explainable AI (XAI)</div>
        <div class="feat-desc">Shannon entropy uncertainty gauge and latent feature space analysis to understand model decisions.</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Architecture Overview ──────────────────────────────────────────────────────
st.subheader("🧠 CNN Model Architecture")

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("""
| Layer | Config | Output Shape |
|:---|:---|:---|
| **Input** | I/Q samples | `(None, 128, 2)` |
| **Conv1D** | 32 filters, kernel=5, ReLU + BN + MaxPool(2) + Drop(0.2) | `(None, 64, 32)` |
| **Conv1D** | 64 filters, kernel=5, ReLU + BN + MaxPool(2) + Drop(0.25) | `(None, 32, 64)` |
| **Conv1D** | 96 filters, kernel=3, ReLU + BN + GlobalAvgPool | `(None, 96)` |
| **Dense** | 64 units, ReLU + Dropout(0.4) | `(None, 64)` |
| **Dense** | 3 units, Softmax | `(None, 3)` |
    """)

with col2:
    st.markdown("""
**Dataset Info**
- 📦 **Source**: RadioML subset
- 🎯 **Classes**: BPSK · QPSK · 16-QAM
- 📏 **Signal Length**: 128 samples × 2 channels
- 📡 **SNR Range**: -20 dB to +18 dB
- 🔢 **Test Samples**: 900

**Performance**
| SNR Range | Accuracy |
|:---|:---|
| < -6 dB | 20–51% |
| -6 to 0 dB | 75–93% |
| > 0 dB | **95–100%** |
    """)

st.divider()

# ── Navigation Guide ───────────────────────────────────────────────────────────
st.subheader("📌 Navigate the App")
st.info(
    "**Use the sidebar on the left** to navigate between the pages below. "
    "Each page is a self-contained module.",
    icon="👈",
)

pages_info = [
    ("🛰️ Live Classifier", "pages/1_🛰️_Live_Classifier.py", "Classify uploaded or sample I/Q signals — includes file upload, constellation, waveform, PSD, and uncertainty analysis."),
    ("🎛️ Satellite Studio", "pages/2_🎛️_Satellite_Studio.py", "Synthesize satellite signals with full channel impairment controls: Doppler, CFO, TWTA, Rician fading."),
    ("📊 Model Evaluation", "pages/3_📊_Model_Evaluation.py", "Interactive confusion matrix, SNR accuracy curves, classification report, and per-class analysis."),
    ("🔬 Signal Visualizer", "pages/4_🔬_Signal_Visualizer.py", "Deep signal analysis: constellation, time-domain, FFT spectrum, spectrogram, and eye diagram."),
    ("📖 RF Theory", "pages/5_📖_RF_Theory.py", "Learn about modulation schemes, satellite channels, DVB-S2 standards, and APSK constellations."),
]

cols = st.columns(len(pages_info))
for col, (title, path, desc) in zip(cols, pages_info):
    with col:
        st.markdown(f"**{title}**")
        st.caption(desc)

st.divider()
st.caption(
    "Academic research prototype using RadioML dataset signals and a trained CNN. "
    "Not connected to a live satellite receiver. "
    "Built with Streamlit · Plotly · NumPy · TensorFlow/ONNX."
)
