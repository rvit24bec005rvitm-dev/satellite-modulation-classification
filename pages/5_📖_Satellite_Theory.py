"""
pages/5_📖_Satellite_Theory.py
Interactive technical satellite communication encyclopedia & theoretical reference:
  - Modulation Fundamentals (BPSK, QPSK, 16-QAM, 16-APSK, 32-APSK, GMSK)
  - Satellite Channel Physics (LEO Doppler, CFO, Phase Noise, Rician Fading)
  - High Power Amplifier Non-Linearities (TWTA Saleh Model)
  - Deep Learning for Automatic Modulation Classification (AMC)
"""

from pathlib import Path
import sys
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ── Path setup ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.signal_processing import CONSTELLATION_REFS

st.set_page_config(
    page_title="Satellite Theory · SatModClass",
    page_icon="📖",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
html, [class*="css"] { font-family: 'Inter', sans-serif; }
.theory-card {
    background: linear-gradient(135deg, #111827 0%, #0F172A 100%);
    border: 1px solid #1E293B;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}
.math-block {
    background: #020617;
    border-left: 3px solid #00D4FF;
    padding: 0.75rem 1rem;
    border-radius: 4px;
    margin: 0.75rem 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem;
    color: #E2E8F0;
}
</style>
""", unsafe_allow_html=True)

st.title("📖 Satellite Communication & Modulation Theory")
st.markdown("Technical documentation, mathematical formulations, and domain references for AI-based Satellite Automatic Modulation Classification.")

tabs = st.tabs([
    "🛰️ Modulation Schemes",
    "🌌 Satellite Channel Physics",
    "⚡ TWTA Amplifier Non-Linearity",
    "🧠 Deep Learning AMC Architecture",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: Modulation Schemes
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.header("Digital Satellite Modulation Standards")
    
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        ### Why Specialized Modulations for Space?
        Terrestrial wireless communications (such as Wi-Fi or 5G) frequently use high-order rectangular **QAM** (e.g., 64-QAM, 256-QAM, 1024-QAM).
        However, satellite transponders are severely power-limited and operate near the **saturation point** of Travelling Wave Tube Amplifiers (TWTAs) to maximize link budget.
        
        High-order rectangular QAM suffers from high **Peak-to-Average Power Ratio (PAPR)**, causing destructive non-linear intermodulation distortion when passed through saturated satellite amplifiers.
        """)
        
        st.markdown("#### Comparison of Common Satellite Schemes")
        st.markdown("""
        - **BPSK / QPSK**: Constant envelope ($PAPR = 0\\text{ dB}$ before pulse shaping). Maximum resilience against deep space noise, solar flares, and high thermal temperatures.
        - **16-QAM**: High spectral efficiency (4 bits/symbol), but requires significant amplifier back-off ($3\\text{ to }5\\text{ dB}$) to prevent constellation warping.
        - **16-APSK / 32-APSK (DVB-S2 / DVB-S2X Standard)**: Arranges constellation symbols on **concentric circular rings** (4 inner, 12 outer for 16-APSK; 4+12+16 for 32-APSK). Because symbols on each ring have equal amplitude, non-linear TWTA gain compression preserves symbol separation much better than rectangular QAM.
        - **GMSK (Gaussian Minimum Shift Keying)**: Constant envelope continuous-phase frequency modulation widely deployed on CubeSats, meteorological orbiters, and marine AIS payloads.
        """)

    with col2:
        st.markdown("#### Interactive Reference Constellation")
        mod_choice = st.selectbox(
            "Select Scheme to Visualize",
            ["BPSK", "QPSK", "16-QAM", "16-APSK", "32-APSK"],
        )
        
        pts = CONSTELLATION_REFS[mod_choice]
        fig_ref = go.Figure()
        
        # Unit circles for APSK
        if "APSK" in mod_choice:
            theta = np.linspace(0, 2*np.pi, 200)
            if mod_choice == "16-APSK":
                fig_ref.add_trace(go.Scatter(x=np.cos(theta), y=np.sin(theta), mode="lines", line=dict(color="#334155", dash="dot"), name="Inner Ring"))
                fig_ref.add_trace(go.Scatter(x=2.75*np.cos(theta), y=2.75*np.sin(theta), mode="lines", line=dict(color="#334155", dash="dot"), name="Outer Ring"))
            elif mod_choice == "32-APSK":
                for r in [1.0, 2.84, 5.27]:
                    fig_ref.add_trace(go.Scatter(x=r*np.cos(theta), y=r*np.sin(theta), mode="lines", line=dict(color="#334155", dash="dot"), showlegend=False))

        fig_ref.add_trace(go.Scatter(
            x=pts.real,
            y=pts.imag,
            mode="markers+text",
            marker=dict(size=14, color="#00D4FF", line=dict(width=2, color="#FFFFFF")),
            text=[f"S{i}" for i in range(len(pts))],
            textposition="top center",
            textfont=dict(size=10, color="#94A3B8"),
            name="Ideal Symbol",
        ))

        fig_ref.update_layout(
            title=f"Theoretical {mod_choice} Constellation",
            xaxis=dict(title="In-Phase (I)", zeroline=True, zerolinecolor="#475569", gridcolor="#1E293B"),
            yaxis=dict(title="Quadrature (Q)", zeroline=True, zerolinecolor="#475569", gridcolor="#1E293B", scaleanchor="x", scaleratio=1),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#CBD5E1"),
            height=360,
            margin=dict(l=40, r=20, t=50, b=40),
            showlegend=False,
        )
        st.plotly_chart(fig_ref, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: Satellite Channel Physics
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.header("Physical Channel Impairments in Space Links")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 1. LEO Doppler Shift & Doppler Rate")
        st.markdown("""
        Low Earth Orbit (LEO) satellites (e.g., Starlink, Iridium, Earth Observation CubeSats) travel at orbital velocities exceeding **$7.5\\text{ km/s}$** at altitudes between $500\\text{ km}$ and $1200\\text{ km}$.
        This rapid relative velocity creates massive Doppler frequency shifts that evolve rapidly during a ground station pass:
        """)
        st.latex(r"f_d(t) = \frac{v_{\text{rel}}(t)}{c} f_c")
        st.latex(r"s_{\text{doppler}}[n] = s[n] \cdot \exp\left( j \left( 2\pi f_d n T_s + \pi \dot{f}_d (n T_s)^2 \right) \right)")
        st.markdown("""
        - At **Ka-band ($30\\text{ GHz}$)**, Doppler shift can exceed **$\\pm 750\\text{ kHz}$**, rotating constellation points around the origin faster than the symbol duration if uncompensated.
        """)

    with c2:
        st.markdown("### 2. Carrier Frequency Offset (CFO) & Phase Noise")
        st.markdown("""
        Due to thermal gradients across spacecraft orbits (ranging from $-100^\\circ\\text{C}$ in eclipse to $+120^\\circ\\text{C}$ under direct solar radiation), local oscillator (LO) crystals experience frequency drift:
        """)
        st.latex(r"s_{\text{cfo}}[n] = s[n] \cdot \exp(j (2\pi \Delta f_c n T_s + \theta_0))")
        st.markdown("""
        **Wiener Phase Noise Drift**:
        """)
        st.latex(r"\theta[n] = \theta[n-1] + w[n], \quad w[n] \sim \mathcal{N}(0, \sigma_{\Delta \theta}^2)")
        st.markdown("""
        Phase noise turns sharp constellation points into circumferential arcs (smearing symbols along circles of constant radius).
        """)

    st.markdown("---")
    st.markdown("### 3. Fading: Line-of-Sight vs Multipath (Rician Fading)")
    st.markdown("""
    Satellite-to-ground links feature a strong direct Line-of-Sight (LOS) specular component with minor scattered multipath from ground reflections, accurately modeled by the **Rician Distribution**:
    """)
    st.latex(r"h = \sqrt{\frac{K}{K+1}} h_{\text{LOS}} + \sqrt{\frac{1}{K+1}} h_{\text{scatter}}")
    st.markdown("""
    - **$K \\to \\infty$**: Pure Line-of-Sight (Deep space / Geostationary GEO satellite).
    - **$K = 10\\text{ to }25\\text{ dB}$**: Typical clear-sky LEO ground pass at elevation $>30^\\circ$.
    - **$K = 0\\text{ dB}$**: Rayleigh fading (extreme urban canyons or heavy forest canopy).
    """)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: TWTA Saleh Model
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.header("Traveling Wave Tube Amplifier (TWTA) Saleh Model")
    
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown("""
        ### Non-Linear High Power Amplification (HPA)
        Travelling Wave Tube Amplifiers (TWTAs) and Solid-State Power Amplifiers (SSPAs) are deployed onboard satellites to transmit signals across $36,000\\text{ km}$ downlinks.
        
        To achieve maximum DC-to-RF power efficiency, amplifiers must be operated near their **saturation point**, which introduces severe non-linearities:
        1. **AM/AM Conversion**: Amplitude-dependent gain compression.
        2. **AM/PM Conversion**: Amplitude-dependent phase rotation.
        
        The standard aerospace model for TWTA behavior was developed by **Adel A. M. Saleh (IEEE Trans. Comm., 1981)**:
        """)
        
        st.latex(r"A(r) = \frac{\alpha_a \cdot r}{1 + \beta_a \cdot r^2} \quad \text{(AM/AM)}")
        st.latex(r"\Phi(r) = \frac{\alpha_\phi \cdot r^2}{1 + \beta_\phi \cdot r^2} \quad \text{(AM/PM)}")
        
        st.markdown("""
        **Default Saleh Parameters for Ku/Ka-band TWTA:**
        - $\\alpha_a = 2.1587$, $\\beta_a = 1.1517$ (Gain compression)
        - $\\alpha_\\phi = 4.0033$, $\\beta_\\phi = 9.1040$ (Phase conversion)
        """)

    with col_b:
        # Plot Saleh AM/AM and AM/PM transfer curves
        r = np.linspace(0.01, 2.5, 300)
        A_r = (2.1587 * r) / (1 + 1.1517 * r**2)
        Phi_r = (4.0033 * r**2) / (1 + 9.1040 * r**2) * (180.0 / np.pi) # in degrees

        fig_saleh = go.Figure()
        fig_saleh.add_trace(go.Scatter(x=r, y=A_r, mode="lines", name="AM/AM Gain", line=dict(color="#00D4FF", width=3)))
        fig_saleh.add_trace(go.Scatter(x=r, y=Phi_r / 20.0, mode="lines", name="AM/PM Phase (deg/20)", line=dict(color="#F43F5E", width=2, dash="dash")))
        fig_saleh.add_trace(go.Scatter(x=r, y=r, mode="lines", name="Ideal Linear", line=dict(color="#64748B", width=1.5, dash="dot")))

        fig_saleh.update_layout(
            title="Saleh TWTA Transfer Characteristics",
            xaxis=dict(title="Input Amplitude (r)", gridcolor="#1E293B"),
            yaxis=dict(title="Output Amplitude / Normalized Phase", gridcolor="#1E293B"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#CBD5E1"),
            legend=dict(orientation="h", y=1.15),
            height=360,
            margin=dict(l=40, r=20, t=50, b=40),
        )
        st.plotly_chart(fig_saleh, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: Deep Learning AMC Architecture
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.header("Deep Learning Pipeline for Satellite AMC")
    
    st.markdown("""
    ### Why Deep Learning over Classical Feature-Based Classification?
    Traditional Automatic Modulation Classification required manual feature engineering:
    - Higher-Order Statistics (HOS): Cumulants $C_{40}, C_{42}, C_{63}$
    - Spectral Moments and Wavelet Transform coefficients
    
    While effective in pure Gaussian noise, hand-crafted features **fail catastrophically** when subjected to realistic satellite non-linearities (TWTA gain saturation, unknown Doppler rate, frequency offsets).
    
    Deep Neural Networks learn robust, invariant topological representations directly from raw baseband $I[n]$ and $Q[n]$ sequences.
    """)
    
    col_x, col_y = st.columns(2)
    with col_x:
        st.markdown("#### System 1D-CNN Architecture")
        st.markdown("""
        ```
        Input: (None, 128, 2)  [128 time samples x 2 channels (I, Q)]
          │
          ├── Conv1D (32 filters, kernel=5) + BatchNorm + ReLU
          ├── MaxPool1D (pool_size=2) + Dropout(0.20)
          │
          ├── Conv1D (64 filters, kernel=5) + BatchNorm + ReLU
          ├── MaxPool1D (pool_size=2) + Dropout(0.25)
          │
          ├── Conv1D (96 filters, kernel=3) + BatchNorm + ReLU
          ├── GlobalAveragePooling1D (Temporal Collapse) -> (96,)
          │
          ├── Dense (64 units) + ReLU + Dropout(0.40)
          └── Dense (3 units) + Softmax -> [P(BPSK), P(QPSK), P(16-QAM)]
        ```
        """)

    with col_y:
        st.markdown("#### Shannon Entropy for Uncertainty Estimation")
        st.markdown("""
        In real mission operations, the classifier must identify when a signal is corrupted beyond recognition or belongs to an unknown out-of-distribution modulation:
        """)
        st.latex(r"H(p) = -\sum_{i=1}^{C} p_i \log_2(p_i)")
        st.markdown(r"""
        - **$H(p) \approx 0\text{ bits}$**: High confidence. The network has concentrated $>95\%$ probability mass into a single class.
        - **$H(p) > 1.2\text{ bits}$**: Ambiguity alert! The signal-to-noise ratio is too low ($\text{SNR} < -4\text{ dB}$) or the transponder is experiencing uncompensated Doppler drift.
        """)
