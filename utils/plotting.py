"""utils/plotting.py
Reusable, dark-mode Plotly chart factories for the satellite modulation
classification Streamlit app.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Shared dark theme ──────────────────────────────────────────────────────────
DARK_LAYOUT = dict(
    paper_bgcolor="#0A0E1A",
    plot_bgcolor="#111827",
    font=dict(color="#E2E8F0", family="Inter, sans-serif"),
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor="#1F2937"),
)

CLASS_COLORS = {
    "BPSK":   "#00D4FF",
    "QPSK":   "#7C3AED",
    "QAM16":  "#F59E0B",
    "8PSK":   "#10B981",
    "16APSK": "#EF4444",
    "32APSK": "#EC4899",
    "GMSK":   "#6366F1",
    "BFSK":   "#14B8A6",
    "Unknown":"#94A3B8",
}

SIGNAL_COLORS = {"I": "#00D4FF", "Q": "#7C3AED", "Amplitude": "#F59E0B", "Phase": "#10B981"}


# ── Constellation Diagram ──────────────────────────────────────────────────────

def plot_constellation(
    iq: np.ndarray,
    predicted_class: str = "Unknown",
    ref_points: np.ndarray | None = None,
) -> go.Figure:
    """
    Scatter plot of I vs Q with optional reference constellation overlay.
    """
    I, Q = iq[:, 0], iq[:, 1]
    color = CLASS_COLORS.get(predicted_class, CLASS_COLORS["Unknown"])

    fig = go.Figure()

    # Received samples
    fig.add_trace(go.Scatter(
        x=I, y=Q,
        mode="markers",
        marker=dict(
            size=5,
            color=color,
            opacity=0.65,
            line=dict(width=0),
        ),
        name=f"Received ({predicted_class})",
        hovertemplate="I: %{x:.4f}<br>Q: %{y:.4f}",
    ))

    # Theoretical reference points
    if ref_points is not None and len(ref_points) > 0:
        fig.add_trace(go.Scatter(
            x=ref_points[:, 0],
            y=ref_points[:, 1],
            mode="markers",
            marker=dict(
                size=14,
                color="white",
                symbol="cross",
                line=dict(color=color, width=2),
            ),
            name="Ideal Symbols",
            hovertemplate="I: %{x:.3f}<br>Q: %{y:.3f}",
        ))

    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text=f"🔭 I/Q Constellation — {predicted_class}", font=dict(size=16)),
        xaxis=dict(title="In-Phase (I)", zeroline=True, zerolinecolor="#374151", gridcolor="#1F2937"),
        yaxis=dict(title="Quadrature (Q)", scaleanchor="x", zeroline=True, zerolinecolor="#374151", gridcolor="#1F2937"),
        height=420,
    )
    return fig


# ── Time-Domain Waveform ───────────────────────────────────────────────────────

def plot_time_domain(iq: np.ndarray) -> go.Figure:
    """
    Three-row subplot: I(t), Q(t), and instantaneous amplitude A(t).
    """
    n = np.arange(len(iq))
    I, Q = iq[:, 0], iq[:, 1]
    A = np.sqrt(I ** 2 + Q ** 2)
    phi = np.degrees(np.arctan2(Q, I))

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=["I(t) — In-Phase", "Q(t) — Quadrature", "Instantaneous Amplitude |A(t)|"],
        vertical_spacing=0.08,
    )

    fig.add_trace(go.Scatter(x=n, y=I, mode="lines",
                             line=dict(color=SIGNAL_COLORS["I"], width=1.5),
                             name="I(t)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=n, y=Q, mode="lines",
                             line=dict(color=SIGNAL_COLORS["Q"], width=1.5),
                             name="Q(t)"), row=2, col=1)
    fig.add_trace(go.Scatter(x=n, y=A, mode="lines",
                             line=dict(color=SIGNAL_COLORS["Amplitude"], width=1.5),
                             fill="tozeroy", fillcolor="rgba(245,158,11,0.12)",
                             name="|A(t)|"), row=3, col=1)

    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text="📊 Time-Domain I/Q Waveforms", font=dict(size=16)),
        height=500,
        showlegend=True,
    )
    for row in [1, 2, 3]:
        fig.update_yaxes(gridcolor="#1F2937", zerolinecolor="#374151", row=row, col=1)
    fig.update_xaxes(title_text="Sample Index", gridcolor="#1F2937", row=3, col=1)
    return fig


# ── Power Spectral Density ─────────────────────────────────────────────────────

def plot_psd(freqs: np.ndarray, psd_db: np.ndarray) -> go.Figure:
    """Line chart of two-sided PSD in dB vs normalised frequency."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=freqs, y=psd_db,
        mode="lines",
        line=dict(color="#00D4FF", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(0,212,255,0.08)",
        name="PSD",
        hovertemplate="Freq: %{x:.4f}<br>PSD: %{y:.2f} dB",
    ))
    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text="📡 Power Spectral Density (Two-Sided)", font=dict(size=16)),
        xaxis=dict(title="Normalised Frequency (×π rad/sample)", gridcolor="#1F2937"),
        yaxis=dict(title="PSD (dBW/Hz)", gridcolor="#1F2937"),
        height=380,
    )
    return fig


# ── Spectrogram ────────────────────────────────────────────────────────────────

def plot_spectrogram(times: np.ndarray, freqs: np.ndarray, Zxx_db: np.ndarray) -> go.Figure:
    """Heatmap STFT spectrogram."""
    fig = go.Figure(go.Heatmap(
        x=times,
        y=freqs,
        z=Zxx_db,
        colorscale="Plasma",
        colorbar=dict(title="dB", tickfont=dict(color="#E2E8F0")),
        hovertemplate="Time: %{x:.3f}<br>Freq: %{y:.4f}<br>PSD: %{z:.2f} dB",
    ))
    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text="🌊 STFT Spectrogram (Time–Frequency)", font=dict(size=16)),
        xaxis=dict(title="Time (samples)", gridcolor="#1F2937"),
        yaxis=dict(title="Normalised Frequency", gridcolor="#1F2937"),
        height=380,
    )
    return fig


# ── SNR Accuracy Curve ────────────────────────────────────────────────────────

def plot_snr_accuracy(df) -> go.Figure:
    """Multi-line SNR vs Accuracy chart from snr_accuracy_results.csv DataFrame."""
    fig = go.Figure()
    class_columns = {
        "BPSK_Accuracy_Percent": ("BPSK", CLASS_COLORS["BPSK"]),
        "QPSK_Accuracy_Percent": ("QPSK", CLASS_COLORS["QPSK"]),
        "QAM16_Accuracy_Percent": ("QAM16", CLASS_COLORS["QAM16"]),
        "Overall_Accuracy_Percent": ("Overall", "#FFFFFF"),
    }
    for col, (label, color) in class_columns.items():
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df["SNR_dB"],
                y=df[col],
                mode="lines+markers",
                line=dict(color=color, width=2.5 if label == "Overall" else 1.8),
                marker=dict(size=6, color=color),
                name=label,
                hovertemplate=f"{label}<br>SNR: %{{x}} dB<br>Accuracy: %{{y:.1f}}%",
            ))
    fig.add_hline(y=100, line_dash="dot", line_color="#374151", annotation_text="100%")
    fig.add_hline(y=33.3, line_dash="dot", line_color="#6B7280", annotation_text="Random (3-class)")
    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text="📈 Classification Accuracy vs SNR Level", font=dict(size=16)),
        xaxis=dict(title="SNR (dB)", gridcolor="#1F2937"),
        yaxis=dict(title="Accuracy (%)", range=[0, 105], gridcolor="#1F2937"),
        height=420,
        hovermode="x unified",
    )
    return fig


# ── Prediction Score Bar Chart ────────────────────────────────────────────────

def plot_class_scores(classes: list[str], scores: np.ndarray, predicted_class: str) -> go.Figure:
    """Horizontal bar chart of model softmax scores per class."""
    colors = [
        CLASS_COLORS.get(c, CLASS_COLORS["Unknown"])
        if c == predicted_class else "#374151"
        for c in classes
    ]
    fig = go.Figure(go.Bar(
        x=scores * 100,
        y=classes,
        orientation="h",
        marker=dict(color=colors, line=dict(color=colors, width=1.5)),
        text=[f"{s * 100:.1f}%" for s in scores],
        textposition="outside",
        textfont=dict(color="#E2E8F0"),
        hovertemplate="%{y}: %{x:.2f}%",
    ))
    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text="🎯 Model Confidence per Class", font=dict(size=16)),
        xaxis=dict(title="Confidence (%)", range=[0, 112], gridcolor="#1F2937"),
        yaxis=dict(gridcolor="#1F2937"),
        height=250,
        showlegend=False,
    )
    return fig


# ── Confusion Matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(matrix: np.ndarray, classes: list[str]) -> go.Figure:
    """Annotated heatmap of the confusion matrix."""
    annotations = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            annotations.append(dict(
                text=str(matrix[i, j]),
                x=classes[j], y=classes[i],
                font=dict(color="white", size=18, family="Inter"),
                showarrow=False,
            ))
    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=classes,
        y=classes,
        colorscale=[
            [0.0, "#0A0E1A"],
            [0.3, "#1E3A5F"],
            [0.7, "#0369A1"],
            [1.0, "#00D4FF"],
        ],
        showscale=True,
        colorbar=dict(title="Count", tickfont=dict(color="#E2E8F0")),
        hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
    ))
    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text="🔬 Confusion Matrix", font=dict(size=16)),
        xaxis=dict(title="Predicted Class", side="bottom"),
        yaxis=dict(title="True Class", autorange="reversed"),
        height=380,
        annotations=annotations,
    )
    return fig


# ── Entropy Gauge ─────────────────────────────────────────────────────────────

def plot_entropy_gauge(entropy: float, max_entropy: float) -> go.Figure:
    """Speedometer gauge showing prediction uncertainty."""
    pct = min(entropy / max_entropy, 1.0) * 100
    color = "#10B981" if pct < 40 else "#F59E0B" if pct < 75 else "#EF4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(entropy, 3),
        delta=dict(reference=0, valueformat=".3f"),
        title=dict(text="Shannon Entropy H(p)<br><span style='font-size:0.7em;color:#9CA3AF'>0 = certain · 1.58 = maximum (3 classes)</span>"),
        gauge=dict(
            axis=dict(range=[0, max_entropy], tickcolor="#E2E8F0"),
            bar=dict(color=color),
            bgcolor="#111827",
            bordercolor="#374151",
            steps=[
                dict(range=[0, max_entropy * 0.4], color="#064E3B"),
                dict(range=[max_entropy * 0.4, max_entropy * 0.75], color="#78350F"),
                dict(range=[max_entropy * 0.75, max_entropy], color="#450A0A"),
            ],
            threshold=dict(line=dict(color="#EF4444", width=3), thickness=0.8, value=max_entropy * 0.75),
        ),
        number=dict(font=dict(color=color, size=32)),
    ))
    fig.update_layout(**DARK_LAYOUT, height=280)
    return fig
