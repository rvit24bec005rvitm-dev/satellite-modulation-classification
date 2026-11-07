"""
pages/5_📖_Satellite_Theory.py
Interactive technical satellite communication encyclopedia & theoretical laboratory:
  - Modulation Fundamentals (BPSK, QPSK, 16-QAM, 16-APSK, 32-APSK, GMSK)
  - Interactive DVB-S2 APSK Ring Ratio Calculator
  - LEO Doppler & Channel Physics with Interactive Ground Pass Calculator
  - Real-Time Interactive TWTA Saleh Distortion Lab
  - Deep Learning for Automatic Modulation Classification (AMC) & Shannon Entropy
"""

from pathlib import Path
import sys
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ── Path setup ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.signal_processing import CONSTELLATION_REFS, saleh_twta, simulate_orbit_pass

st.set_page_config(
    page_title="Satellite Theory · SatModClass",
    page_icon="📖",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;600;700;800&display=swap');

html, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; }
.mono-font { font-family: 'JetBrains Mono', monospace !important; }

.theory-card {
    background: linear-gradient(135deg, rgba(17, 24, 39, 0.95) 0%, rgba(15, 23, 42, 0.95) 100%);
    border: 1px solid #1E293B;
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.35);
}
.math-block {
    background: #070B14;
    border-left: 3px solid #00D4FF;
    padding: 0.85rem 1.15rem;
    border-radius: 6px;
    margin: 0.85rem 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.92rem;
    color: #E2E8F0;
}
.calc-box {
    background: rgba(13, 19, 35, 0.9);
    border: 1px solid rgba(0, 212, 255, 0.25);
    border-radius: 12px;
    padding: 1.25rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

st.title("📖 Satellite Communication & Modulation Theory Lab")
st.markdown("Aerospace communication standards, mathematical formulations, and interactive simulators for spaceborne Automatic Modulation Classification.")

tabs = st.tabs([
    "🛰️ Modulation Schemes & APSK",
    "🌌 LEO Doppler & Channel Physics",
    "⚡ Interactive TWTA Saleh Distortion Lab",
    "🧠 Deep Learning AMC Pipeline",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: Modulation Schemes & APSK
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.header("Digital Space Communication Modulation Standards")
    
    col1, col2 = st.columns([1.6, 1.4])
    with col1:
        st.markdown("""
        ### Why Specialized Modulations for Space?
        Terrestrial wireless communications (5G, Wi-Fi 6) frequently deploy high-order rectangular **QAM** (64-QAM to 4096-QAM).
        However, satellite transponders are severely power-limited and operate near the **saturation point** of Travelling Wave Tube Amplifiers (TWTAs) to maximize link budget across tens of thousands of kilometers.
        
        High-order rectangular QAM suffers from high **Peak-to-Average Power Ratio (PAPR)** ($>6\\text{ dB}$), causing severe non-linear intermodulation distortion when passed through saturated satellite amplifiers.
        """)
        
        st.markdown("#### Aerospace Modulation Comparison")
        st.markdown("""
        - **BPSK / QPSK**: Constant envelope ($PAPR = 0\\text{ dB}$ before pulse shaping). Maximum resilience against deep space noise, solar flares, and high thermal temperatures.
        - **16-QAM**: High spectral efficiency (4 bits/symbol), but requires significant amplifier back-off ($3\\text{ to }5\\text{ dB}$) to prevent constellation warping.
        - **16-APSK / 32-APSK (DVB-S2 / DVB-S2X Standard)**: Arranges symbols on **concentric circular rings** (4 inner, 12 outer for 16-APSK). Because symbols on each ring have equal amplitude, non-linear TWTA gain compression preserves symbol separation much better than rectangular QAM.
        - **GMSK**: Constant envelope continuous-phase frequency modulation widely deployed on CubeSats and marine AIS payloads.
        """)

    with col2:
        st.markdown("#### Interactive Reference Constellation")
        mod_choice = st.selectbox(
            "Select Modulation Scheme",
            ["BPSK", "QPSK", "16-QAM", "16-APSK", "32-APSK"],
        )
        
        pts = CONSTELLATION_REFS[mod_choice]
        fig_ref = go.Figure()
        
        # Unit circles for APSK
        if "APSK" in mod_choice:
            theta = np.linspace(0, 2*np.pi, 200)
            if mod_choice == "16-APSK":
                fig_ref.add_trace(go.Scatter(x=np.cos(theta), y=np.sin(theta), mode="lines", line=dict(color="#334155", dash="dot"), name="Inner Ring"))
                fig_ref.add_trace(go.Scatter(x=2.53*np.cos(theta), y=2.53*np.sin(theta), mode="lines", line=dict(color="#334155", dash="dot"), name="Outer Ring"))
            elif mod_choice == "32-APSK":
                for r in [1.0, 2.53, 4.30]:
                    fig_ref.add_trace(go.Scatter(x=r*np.cos(theta), y=r*np.sin(theta), mode="lines", line=dict(color="#334155", dash="dot"), showlegend=False))

        fig_ref.add_trace(go.Scatter(
            x=pts.real if np.iscomplexobj(pts) else pts[:, 0],
            y=pts.imag if np.iscomplexobj(pts) else pts[:, 1],
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

    st.markdown("---")
    st.subheader("🎛️ DVB-S2 16-APSK Ring Radius Ratio (γ = R₂/R₁) Explorer")
    st.markdown("The performance of 16-APSK under TWTA non-linearity is dictated by the ring radius ratio $\\gamma = R_2 / R_1$. Standard DVB-S2 uses $\\gamma \\in [2.57, 3.15]$ depending on code rate.")
    
    gamma_val = st.slider("Ring Radius Ratio γ (R₂ / R₁)", 1.5, 4.5, 2.75, 0.05)
    
    # Generate custom APSK with this gamma
    theta = np.linspace(0, 2*np.pi, 200)
    fig_apsk = go.Figure()
    fig_apsk.add_trace(go.Scatter(x=np.cos(theta), y=np.sin(theta), mode="lines", line=dict(color="#475569", dash="dot"), name="Inner Ring (R1=1)"))
    fig_apsk.add_trace(go.Scatter(x=gamma_val*np.cos(theta), y=gamma_val*np.sin(theta), mode="lines", line=dict(color="#38BDF8", dash="dot"), name=f"Outer Ring (R2={gamma_val})"))
    
    # 4 inner symbols
    inner_syms = np.array([np.exp(1j * (2 * np.pi * k / 4 + np.pi / 4)) for k in range(4)])
    # 12 outer symbols
    outer_syms = np.array([gamma_val * np.exp(1j * (2 * np.pi * k / 12 + np.pi / 12)) for k in range(12)])
    
    all_syms = np.concatenate([inner_syms, outer_syms])
    fig_apsk.add_trace(go.Scatter(
        x=all_syms.real, y=all_syms.imag,
        mode="markers",
        marker=dict(size=12, color="#00D4FF", line=dict(color="white", width=1.5)),
        name="16-APSK Symbols",
    ))
    fig_apsk.update_layout(
        title=f"16-APSK Dynamic Constellation (γ = {gamma_val:.2f})",
        xaxis=dict(title="I", scaleanchor="x", gridcolor="#1E293B"),
        yaxis=dict(title="Q", gridcolor="#1E293B"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#CBD5E1"),
        height=340,
    )
    st.plotly_chart(fig_apsk, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: LEO Doppler & Channel Physics
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.header("Physical Space Channel Mechanics & LEO Doppler Calculator")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 1. LEO Doppler Shift & S-Curve Dynamics")
        st.markdown("""
        Low Earth Orbit satellites travel at orbital velocities exceeding **$7.5\\text{ km/s}$** at altitudes between $500\\text{ km}$ and $1200\\text{ km}$.
        This rapid relative velocity creates massive Doppler frequency shifts that evolve rapidly throughout a ground pass:
        """)
        st.latex(r"f_d(t) = -\frac{v_{\text{LOS}}(t)}{c} f_c = -\frac{\dot{r}(t)}{c} f_c")
        st.markdown("""
        - At **Ka-band ($30\\text{ GHz}$)**, Doppler shift exceeds **$\\pm 750\\text{ kHz}$**, rotating constellation points around the origin faster than the symbol duration if uncompensated.
        """)

    with c2:
        st.markdown("### 2. Carrier Frequency Offset (CFO) & Phase Noise")
        st.markdown("""
        Due to severe thermal gradients across spacecraft orbits ($-100^\\circ\\text{C}$ in eclipse to $+120^\\circ\\text{C}$ in direct solar radiation), local oscillator (LO) crystals experience frequency drift:
        """)
        st.latex(r"s_{\text{cfo}}[n] = s[n] \cdot \exp(j (2\pi \Delta f_c n T_s + \theta_0))")
        st.markdown("""
        **Wiener Phase Noise Random Walk**:
        """)
        st.latex(r"\theta[n] = \theta[n-1] + w[n], \quad w[n] \sim \mathcal{N}(0, \sigma_{\Delta \theta}^2)")

    st.markdown("---")
    st.subheader("🧮 Interactive LEO Orbit & Ground Station Overflight Calculator")
    
    calc_c1, calc_c2 = st.columns([1, 2])
    with calc_c1:
        in_alt = st.number_input("Orbit Altitude (km)", 300, 2000, 550, step=50)
        in_fc = st.selectbox("Carrier Frequency Band", [2.2, 8.4, 14.5, 29.5], index=3, format_func=lambda x: f"{x} GHz ({'S-band' if x<4 else 'X-band' if x<10 else 'Ku-band' if x<18 else 'Ka-band'})")
        in_elev = st.slider("Max Pass Elevation (°)", 20, 90, 75, 5)

        orb_sim = simulate_orbit_pass(altitude_km=in_alt, carrier_freq_ghz=in_fc, max_elevation_deg=in_elev)
        st.markdown(f"""
        <div class="calc-box">
            <div>🚀 <strong>Orbital Velocity:</strong> <span style="color:#00D4FF;">{orb_sim['v_orbit_kms']:.2f} km/s</span></div>
            <div>⏱️ <strong>Orbital Period:</strong> {orb_sim['period_min']:.1f} minutes</div>
            <div>📡 <strong>Peak Doppler Shift:</strong> <span style="color:#34D399;">±{orb_sim['max_doppler_khz']:.1f} kHz</span></div>
            <div>🛰️ <strong>Doppler Rate at TCA:</strong> {(orb_sim['max_doppler_khz']*2 / 300):.2f} kHz/s</div>
        </div>
        """, unsafe_allow_html=True)

    with calc_c2:
        fig_pass = go.Figure()
        fig_pass.add_trace(go.Scatter(
            x=orb_sim["time_sec"], y=orb_sim["doppler_khz"],
            mode="lines",
            line=dict(color="#00D4FF", width=2.5),
            name="Doppler Shift (kHz)",
        ))
        fig_pass.add_vline(x=0, line_dash="dash", line_color="#F59E0B", annotation_text="Zenith TCA")
        fig_pass.update_layout(
            title=f"Doppler S-Curve for {in_fc} GHz Link at {in_alt} km Altitude",
            xaxis=dict(title="Time relative to Closest Approach (seconds)", gridcolor="#1E293B"),
            yaxis=dict(title="Doppler Shift (kHz)", gridcolor="#1E293B"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#CBD5E1"),
            height=300,
        )
        st.plotly_chart(fig_pass, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: Interactive TWTA Saleh Distortion Lab
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.header("Traveling Wave Tube Amplifier (TWTA) Saleh Distortion Lab")
    
    st.markdown("""
    To achieve maximum DC-to-RF power efficiency onboard space probes, High Power Amplifiers operate near **saturation**.
    The standard aerospace model for TWTA non-linearity is the **Saleh Model (IEEE Trans. Comm., 1981)**:
    """)
    st.latex(r"A(r) = \frac{\alpha_a \cdot r}{1 + \beta_a \cdot r^2} \quad \text{[AM/AM Gain Compression]}")
    st.latex(r"\Phi(r) = \frac{\alpha_\phi \cdot r^2}{1 + \beta_\phi \cdot r^2} \quad \text{[AM/PM Phase Rotation]}")

    st.markdown("### 🎛️ Live Constellation Warping Simulator")
    st.markdown("Adjust TWTA saturation and watch how an ideal 16-QAM constellation distorts under AM/AM compression and AM/PM phase rotation:")

    twta_col1, twta_col2 = st.columns([1, 2])
    with twta_col1:
        twta_alpha = st.slider("Saleh α_a (Gain Compression)", 0.5, 3.5, 2.16, 0.1)
        twta_beta = st.slider("Saleh β_a (Saturation Curvature)", 0.1, 2.5, 1.15, 0.05)
        twta_phi = st.slider("Saleh α_φ (Phase Rotation Factor)", 0.0, 8.0, 4.0, 0.2)

        # Baseline 16-QAM points
        qam_ref = CONSTELLATION_REFS["16-QAM"]
        # Apply Saleh distortion
        distorted_pts = saleh_twta(
            qam_ref,
            alpha_a=twta_alpha,
            beta_a=twta_beta,
            alpha_phi=twta_phi,
            beta_phi=9.10,
        )

    with twta_col2:
        fig_dist = go.Figure()
        # Original points
        fig_dist.add_trace(go.Scatter(
            x=qam_ref[:, 0], y=qam_ref[:, 1],
            mode="markers",
            marker=dict(size=12, color="rgba(255,255,255,0.4)", symbol="cross"),
            name="Ideal 16-QAM",
        ))
        # Distorted points
        fig_dist.add_trace(go.Scatter(
            x=distorted_pts[:, 0], y=distorted_pts[:, 1],
            mode="markers+text",
            marker=dict(size=14, color="#00D4FF", line=dict(color="#EC4899", width=2)),
            text=[f"S{i}" for i in range(len(distorted_pts))],
            textposition="top center",
            name="TWTA Saturated",
        ))
        # Draw displacement vectors
        for i in range(len(qam_ref)):
            fig_dist.add_shape(
                type="line",
                x0=qam_ref[i, 0], y0=qam_ref[i, 1],
                x1=distorted_pts[i, 0], y1=distorted_pts[i, 1],
                line=dict(color="rgba(236, 72, 153, 0.4)", width=1.5, dash="dot"),
            )

        fig_dist.update_layout(
            title="16-QAM Constellation Warping under TWTA Non-Linearity",
            xaxis=dict(title="In-Phase (I)", scaleanchor="x", gridcolor="#1E293B"),
            yaxis=dict(title="Quadrature (Q)", gridcolor="#1E293B"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#CBD5E1"),
            height=380,
        )
        st.plotly_chart(fig_dist, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: Deep Learning AMC Architecture
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.header("Deep Learning Automatic Modulation Classification (AMC)")
    
    st.markdown("""
    ### Why Deep Learning over Classical Feature Engineering?
    Classical automatic modulation classification relied upon hand-crafted Higher-Order Cumulants ($C_{40}, C_{42}, C_{63}$) and spectral moments.
    While mathematically elegant, these estimators fail catastrophically under non-linear TWTA gain compression and rapid uncompensated Doppler rate shifts.
    
    Deep Neural Networks learn invariant topological representations directly from raw baseband $I[n]$ and $Q[n]$ sequences.
    """)

    c_net1, c_net2 = st.columns(2)
    with c_net1:
        st.markdown("#### Production 1D-CNN Architecture")
        st.markdown("""
        ```
        Input Tensor: (None, 128, 2)  [128 time samples x 2 channels (I, Q)]
          │
          ├── Conv1D (32 filters, kernel=5) + ReLU + BatchNorm
          ├── MaxPool1D (pool_size=2) + Dropout(0.20)
          │
          ├── Conv1D (64 filters, kernel=5) + ReLU + BatchNorm
          ├── MaxPool1D (pool_size=2) + Dropout(0.25)
          │
          ├── Conv1D (96 filters, kernel=3) + ReLU + BatchNorm
          ├── GlobalAveragePooling1D (Temporal Collapse) -> (96,)
          │
          ├── Dense (64 units) + ReLU + Dropout(0.40)
          └── Dense (3 units) + Softmax -> [P(BPSK), P(QPSK), P(16-QAM)]
        ```
        """)

    with c_net2:
        st.markdown("#### Shannon Entropy for Uncertainty Estimation")
        st.markdown("""
        Prediction entropy measures ambiguity within the trained classes. It cannot establish corruption or identify unknown modulations without independent validation:
        """)
        st.latex(r"H(p) = -\sum_{i=1}^{C} p_i \log_2(p_i)")
        st.markdown(r"""
        - **$H(p) \approx 0\text{ bits}$**: A concentrated probability distribution, which can still be confidently wrong.
        - **$H(p) > 1.2\text{ bits}$**: High within-model ambiguity; this does not uniquely identify an SNR, impairment, or unknown modulation.
        """)
