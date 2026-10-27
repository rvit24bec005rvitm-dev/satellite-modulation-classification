"""
pages/3_📊_Model_Evaluation.py
Full model evaluation dashboard:
  - Key metrics (accuracy, loss, samples)
  - Interactive confusion matrix (Plotly heatmap)
  - Per-class classification report
  - SNR vs Accuracy curves (multi-line with all classes)
  - Downloadable CSV reports
"""

from pathlib import Path
import sys, json

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.plotting import plot_snr_accuracy, plot_confusion_matrix

st.set_page_config(
    page_title="Model Evaluation · SatModClass",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html,[class*="css"]{ font-family:'Inter',sans-serif; }
.metric-header{
    background:linear-gradient(135deg,#0A0E1A,#0C1E3E);
    border:1px solid #1F3A5F;border-radius:12px;padding:1.2rem 2rem;
    margin-bottom:1.5rem;
}
.report-cell-good { color:#34D399; font-weight:700; }
.report-cell-ok   { color:#FCD34D; }
.report-cell-bad  { color:#F87171; }
</style>
""", unsafe_allow_html=True)

st.title("📊 CNN Model Evaluation Dashboard")
st.markdown("Complete evaluation of the **1D-CNN** trained on the RadioML dataset across all SNR levels.")

CLASSES = ["BPSK", "QPSK", "QAM16"]

# ── Load evaluation JSON ────────────────────────────────────────────────────────
@st.cache_data
def load_evaluation():
    p = ROOT / "evaluation_results.json"
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_snr_csv():
    p = ROOT / "snr_accuracy_results.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)

ev = load_evaluation()
snr_df = load_snr_csv()

# ── Key metrics ─────────────────────────────────────────────────────────────────
if ev:
    accuracy = ev.get("test_accuracy", ev.get("accuracy", 0))
    if isinstance(accuracy, float) and accuracy <= 1:
        accuracy *= 100
    loss = ev.get("test_loss", ev.get("loss"))
    n_samples = ev.get("test_samples", ev.get("num_test_samples", 900))

    st.markdown('<div class="metric-header">', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🎯 Test Accuracy", f"{accuracy:.2f}%")
    c2.metric("📉 Test Loss", f"{loss:.4f}" if loss else "N/A")
    c3.metric("📦 Test Samples", str(n_samples))
    c4.metric("🔢 Classes", "3 (BPSK/QPSK/QAM16)")
    c5.metric("📡 SNR Range", "−20 to +18 dB")
    st.markdown('</div>', unsafe_allow_html=True)

# ── Tabs ────────────────────────────────────────────────────────────────────────
t1, t2, t3, t4 = st.tabs([
    "🔬 Confusion Matrix",
    "📋 Classification Report",
    "📈 SNR Accuracy Curves",
    "💾 Raw Data",
])

# ── Confusion Matrix ───────────────────────────────────────────────────────────
with t1:
    st.subheader("Confusion Matrix (All SNR Levels Combined)")
    if ev and "confusion_matrix" in ev:
        cm = np.array(ev["confusion_matrix"])
        fig_cm = plot_confusion_matrix(cm, CLASSES[:cm.shape[0]])
        st.plotly_chart(fig_cm, use_container_width=True)

        # Normalised version
        st.subheader("Normalised Confusion Matrix (%)")
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        cm_norm_df = pd.DataFrame(cm_norm, index=CLASSES[:cm.shape[0]], columns=CLASSES[:cm.shape[1]])
        st.dataframe(
            cm_norm_df.style.background_gradient(cmap="Blues", axis=None)
                            .format("{:.1f}%"),
            use_container_width=True,
        )

        # Per-class stats
        st.subheader("Per-Class Diagonal Accuracy")
        diag_cols = st.columns(len(CLASSES[:cm.shape[0]]))
        for i, (col, cls) in enumerate(zip(diag_cols, CLASSES[:cm.shape[0]])):
            diag_acc = cm[i, i] / cm[i, :].sum() * 100
            col.metric(cls, f"{diag_acc:.1f}%")
    else:
        st.warning("Confusion matrix data not found in `evaluation_results.json`.")

# ── Classification Report ──────────────────────────────────────────────────────
with t2:
    st.subheader("Detailed Classification Report")
    if ev and "classification_report" in ev:
        report = ev["classification_report"]
        if isinstance(report, dict):
            rows = []
            for cls in CLASSES:
                if cls in report:
                    r = report[cls]
                    rows.append({
                        "Class": cls,
                        "Precision": r.get("precision", 0),
                        "Recall": r.get("recall", 0),
                        "F1-Score": r.get("f1-score", 0),
                        "Support": int(r.get("support", 0)),
                    })
            for avg_key in ["macro avg", "weighted avg"]:
                if avg_key in report:
                    r = report[avg_key]
                    rows.append({
                        "Class": avg_key.title(),
                        "Precision": r.get("precision", 0),
                        "Recall": r.get("recall", 0),
                        "F1-Score": r.get("f1-score", 0),
                        "Support": int(r.get("support", 0)),
                    })
            
            df_rep = pd.DataFrame(rows).set_index("Class")

            def color_metric(val):
                if isinstance(val, float):
                    if val >= 0.80: return "color: #34D399; font-weight: 700"
                    elif val >= 0.65: return "color: #FCD34D"
                    else: return "color: #F87171"
                return ""

            styler = df_rep.style.format({"Precision": "{:.4f}", "Recall": "{:.4f}", "F1-Score": "{:.4f}"})
            if hasattr(styler, "map"):
                styler = styler.map(color_metric, subset=["Precision", "Recall", "F1-Score"])
            else:
                styler = styler.applymap(color_metric, subset=["Precision", "Recall", "F1-Score"])

            st.dataframe(
                styler,
                use_container_width=True,
                height=280,
            )

            # Bar chart
            import plotly.graph_objects as go
            from utils.plotting import DARK_LAYOUT, CLASS_COLORS
            fig_bar = go.Figure()
            metrics_to_plot = ["Precision", "Recall", "F1-Score"]
            colors = ["#00D4FF", "#7C3AED", "#F59E0B"]
            for metric, color in zip(metrics_to_plot, colors):
                fig_bar.add_trace(go.Bar(
                    name=metric,
                    x=df_rep.index[:3],
                    y=df_rep[metric].values[:3],
                    marker_color=color,
                ))
            fig_bar.update_layout(
                **DARK_LAYOUT,
                title="Precision / Recall / F1 by Class",
                barmode="group",
                yaxis=dict(title="Score", range=[0, 1.05], gridcolor="#1F2937"),
                xaxis=dict(gridcolor="#1F2937"),
                height=350,
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.text(str(report))
    else:
        st.warning("Classification report not found.")

# ── SNR Accuracy Curves ────────────────────────────────────────────────────────
with t3:
    st.subheader("Accuracy vs SNR Level (Per Class + Overall)")
    if snr_df is not None:
        fig_snr = plot_snr_accuracy(snr_df)
        st.plotly_chart(fig_snr, use_container_width=True)

        # Performance tiers
        st.subheader("📊 Performance Tier Summary")
        tier_data = {
            "SNR Tier": ["Very Low (< −10 dB)", "Low (−10 to −2 dB)", "Mid (−2 to +4 dB)", "High (+4 to +10 dB)", "Very High (> +10 dB)"],
            "SNR Range": ["< −10 dB", "−10 to −2 dB", "−2 to +4 dB", "+4 to +10 dB", "> +10 dB"],
            "Typical Accuracy": ["20–45%", "31–82%", "82–96%", "96–100%", "95–100%"],
            "Classification Quality": ["🔴 Poor", "🟠 Degraded", "🟡 Moderate", "🟢 Good", "🟢 Excellent"],
        }
        st.dataframe(pd.DataFrame(tier_data).set_index("SNR Tier"), use_container_width=True)

        # Download
        csv_bytes = snr_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download SNR Results CSV",
            data=csv_bytes,
            file_name="snr_accuracy_results.csv",
            mime="text/csv",
        )
    else:
        st.warning("`snr_accuracy_results.csv` not found.")

# ── Raw Data ───────────────────────────────────────────────────────────────────
with t4:
    st.subheader("Raw Evaluation JSON")
    if ev:
        st.json(ev)
        import json as _json
        json_bytes = _json.dumps(ev, indent=2).encode("utf-8")
        st.download_button(
            "⬇️ Download evaluation_results.json",
            data=json_bytes,
            file_name="evaluation_results.json",
            mime="application/json",
        )
    else:
        st.warning("`evaluation_results.json` not found.")
    
    if snr_df is not None:
        st.subheader("SNR Accuracy Table")
        st.dataframe(snr_df, use_container_width=True)

st.divider()
st.caption(
    "Evaluated on the full RadioML test set spanning −20 dB to +18 dB SNR. "
    "Overall accuracy is averaged across all SNR levels. "
    "High-SNR performance (>+6 dB) reaches 95–100% for all classes."
)
