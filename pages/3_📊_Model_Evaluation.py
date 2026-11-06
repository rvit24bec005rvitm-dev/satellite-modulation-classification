"""
pages/3_📊_Model_Evaluation.py
Full model evaluation dashboard:
  - Global benchmark metrics (accuracy, loss, samples)
  - Interactive SNR-sliced Dynamic Confusion Matrix (slider from -20dB to +18dB)
  - Per-class Radar Profile & classification report
  - SNR vs Accuracy curves (multi-line with all classes)
  - Performance tier diagnostics and downloadable reports
"""

from pathlib import Path
import sys, json

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.plotting import (
    plot_snr_accuracy,
    plot_confusion_matrix,
    plot_class_radar,
    DARK_LAYOUT,
    CLASS_COLORS,
)
from utils.model_backend import ModelBackend
from utils.signal_processing import normalize_signal

st.set_page_config(
    page_title="Model Evaluation · SatModClass",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;600;700;800&display=swap');

html, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; }
.mono-font { font-family: 'JetBrains Mono', monospace !important; }

.metric-header {
    background: linear-gradient(135deg, rgba(10, 14, 26, 0.95), rgba(12, 30, 62, 0.85));
    border: 1px solid rgba(0, 212, 255, 0.3);
    border-radius: 14px;
    padding: 1.4rem 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.35);
}
.report-cell-good { color: #34D399; font-weight: 700; }
.report-cell-ok   { color: #FCD34D; }
.report-cell-bad  { color: #F87171; }
</style>
""", unsafe_allow_html=True)

st.title("📊 CNN Model Evaluation & Telemetry Dashboard")
st.markdown("Comprehensive performance evaluation of the **1D-CNN** trained on the RadioML satellite dataset across all SNR operating regimes.")

CLASSES = ["BPSK", "QPSK", "QAM16"]

# ── Load evaluation JSON & CSV ──────────────────────────────────────────────────
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

@st.cache_data
def load_test_samples():
    p = ROOT / "classifier_samples.npz"
    if p.exists():
        d = np.load(p, allow_pickle=False)
        return d["signals"].copy(), d["labels"].astype(str), d["snrs"].copy()
    return None, None, None

@st.cache_data
def load_norm_stats():
    p = ROOT / "normalization_stats_all_snr.npz"
    if p.exists():
        with np.load(p) as d:
            return float(d["mean"]), float(d["std"])
    return -6.56e-5, 0.00607

@st.cache_resource
def load_eval_backend():
    p = ROOT / "satellite_modulation_cnn_all_snr.keras"
    if p.exists():
        return ModelBackend.load(p)
    return None

ev = load_evaluation()
snr_df = load_snr_csv()
test_signals, test_labels, test_snrs = load_test_samples()
eval_mean, eval_std = load_norm_stats()
eval_backend = load_eval_backend()

# ── Key Metrics HUD ─────────────────────────────────────────────────────────────
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
    c4.metric("🔢 Classes", "3 (BPSK/QPSK/16-QAM)")
    c5.metric("📡 SNR Range", "−20 to +18 dB")
    st.markdown('</div>', unsafe_allow_html=True)

# ── Tabs ────────────────────────────────────────────────────────────────────────
t1, t2, t3, t4 = st.tabs([
    "🔬 Dynamic Confusion Matrix",
    "📋 Classification Report & Radar",
    "📈 SNR vs Accuracy Curves",
    "💾 Raw Data & Export",
])

# ── Tab 1: Dynamic Confusion Matrix ─────────────────────────────────────────────
with t1:
    st.subheader("Dynamic Confusion Matrix Inspector")
    st.markdown("Inspect the confusion matrix across **all SNR levels combined**, or filter dynamically by a specific SNR band to see how noise degrades class separation.")

    snr_view_mode = st.radio(
        "Matrix Evaluation Slice",
        ["All SNR Levels Combined (Full Test Set)", "Dynamic Slice by SNR Level (dB)"],
        horizontal=True,
    )

    if snr_view_mode == "All SNR Levels Combined (Full Test Set)":
        if ev and "confusion_matrix" in ev:
            cm = np.array(ev["confusion_matrix"])
            fig_cm = plot_confusion_matrix(cm, CLASSES[:cm.shape[0]])
            st.plotly_chart(fig_cm, use_container_width=True)

            # Normalised version
            st.subheader("Normalised Confusion Matrix (%)")
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
            cm_norm_df = pd.DataFrame(cm_norm, index=CLASSES[:cm.shape[0]], columns=CLASSES[:cm.shape[1]])
            st.dataframe(
                cm_norm_df.style.background_gradient(cmap="Blues", axis=None).format("{:.1f}%"),
                use_container_width=True,
            )

            # Per-class stats
            st.subheader("Per-Class Accuracy Across All SNRs")
            diag_cols = st.columns(len(CLASSES[:cm.shape[0]]))
            for i, (col, cls) in enumerate(zip(diag_cols, CLASSES[:cm.shape[0]])):
                diag_acc = cm[i, i] / cm[i, :].sum() * 100
                col.metric(cls, f"{diag_acc:.1f}%")
        else:
            st.warning("Confusion matrix data not found in `evaluation_results.json`.")

    else:
        # Dynamic SNR slice
        if test_signals is not None and eval_backend is not None:
            available_snrs = sorted(int(v) for v in np.unique(test_snrs))
            sel_snr_slice = st.select_slider(
                "Filter by SNR Level (dB)",
                options=available_snrs,
                value=6,
            )

            # Compute predictions on samples matching this SNR
            mask = (test_snrs == sel_snr_slice)
            sub_sigs = test_signals[mask]
            sub_labels = test_labels[mask]

            # Normalize and predict
            sub_norm = (sub_sigs - eval_mean) / eval_std
            sub_preds = []
            for sig in sub_norm:
                p = eval_backend.predict(sig)
                sub_preds.append(CLASSES[int(np.argmax(p))])

            # Build confusion matrix
            class_to_idx = {c: i for i, c in enumerate(CLASSES)}
            dyn_cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
            for true_l, pred_l in zip(sub_labels, sub_preds):
                # normalize label string
                t_key = "QAM16" if "QAM" in true_l else true_l
                p_key = "QAM16" if "QAM" in pred_l else pred_l
                if t_key in class_to_idx and p_key in class_to_idx:
                    dyn_cm[class_to_idx[t_key], class_to_idx[p_key]] += 1

            fig_dyn_cm = plot_confusion_matrix(dyn_cm, CLASSES)
            fig_dyn_cm.update_layout(title=f"🔬 Live Confusion Matrix at SNR = {sel_snr_slice} dB ({len(sub_labels)} Samples)")
            st.plotly_chart(fig_dyn_cm, use_container_width=True)

            # Per-class stats at this SNR
            diag_cols = st.columns(len(CLASSES))
            for i, (col, cls) in enumerate(zip(diag_cols, CLASSES)):
                row_sum = dyn_cm[i, :].sum()
                acc_val = (dyn_cm[i, i] / row_sum * 100) if row_sum > 0 else 0.0
                col.metric(f"{cls} @ {sel_snr_slice} dB", f"{acc_val:.1f}%")
        else:
            st.info("Dynamic SNR slicing requires `classifier_samples.npz` and model backend.")

# ── Tab 2: Classification Report & Radar Profile ───────────────────────────────
with t2:
    st.subheader("Classification Precision, Recall & F1 Analysis")
    if ev and "classification_report" in ev:
        report = ev["classification_report"]
        if isinstance(report, dict):
            col_tbl, col_rad = st.columns([1.3, 1])

            with col_tbl:
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

                st.dataframe(styler, use_container_width=True, height=270)

            with col_rad:
                fig_radar = plot_class_radar(report, CLASSES)
                st.plotly_chart(fig_radar, use_container_width=True)

            # Grouped bar chart
            import plotly.graph_objects as go
            fig_bar = go.Figure()
            metrics_to_plot = ["Precision", "Recall", "F1-Score"]
            colors = ["#00D4FF", "#7C3AED", "#F59E0B"]
            for metric, color in zip(metrics_to_plot, colors):
                fig_bar.add_trace(go.Bar(
                    name=metric,
                    x=df_rep.index[:3],
                    y=df_rep[metric].values[:3],
                    marker_color=color,
                    text=[f"{v*100:.1f}%" for v in df_rep[metric].values[:3]],
                    textposition="outside",
                ))
            fig_bar.update_layout(
                **DARK_LAYOUT,
                title="Per-Class Benchmark Scores",
                barmode="group",
                yaxis=dict(title="Score", range=[0, 1.15], gridcolor="#1F2937"),
                xaxis=dict(gridcolor="#1F2937"),
                height=350,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

# ── Tab 3: SNR vs Accuracy Curves ──────────────────────────────────────────────
with t3:
    st.subheader("Accuracy vs SNR Curve Analysis")
    if snr_df is not None:
        fig_snr = plot_snr_accuracy(snr_df)
        st.plotly_chart(fig_snr, use_container_width=True)

        st.subheader("📊 Operational SNR Performance Tiers")
        tier_data = {
            "SNR Tier": ["Extreme Low (< −10 dB)", "Degraded Low (−10 to −2 dB)", "Mid Transition (−2 to +4 dB)", "High Reliable (+4 to +10 dB)", "Clear Space Link (> +10 dB)"],
            "Typical Accuracy": ["20–45%", "31–82%", "82–96%", "96–100%", "98–100%"],
            "Operational Status": ["🔴 Link Disrupted", "🟠 Degraded Service", "🟡 Synchronizing", "🟢 Operational", "🟢 Maximum Throughput"],
            "Recommendation": ["Increase FEC coding", "Switch to BPSK", "Acceptable for QPSK", "Optimal for 16-QAM", "Deploy 32/64-APSK"],
        }
        st.dataframe(pd.DataFrame(tier_data).set_index("SNR Tier"), use_container_width=True)

# ── Tab 4: Raw Data & Export ───────────────────────────────────────────────────
with t4:
    st.subheader("Raw Telemetry Evaluation Data")
    if ev:
        st.json(ev)
        json_bytes = json.dumps(ev, indent=2).encode("utf-8")
        st.download_button(
            "⬇️ Download evaluation_results.json",
            data=json_bytes,
            file_name="evaluation_results.json",
            mime="application/json",
        )
    if snr_df is not None:
        st.subheader("SNR Accuracy Data Table")
        st.dataframe(snr_df, use_container_width=True)
        csv_bytes = snr_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download SNR Results CSV",
            data=csv_bytes,
            file_name="snr_accuracy_results.csv",
            mime="text/csv",
        )
