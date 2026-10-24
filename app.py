
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf


# =========================================================
# 1. PROJECT FILE PATHS
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_DIR / "satellite_modulation_cnn_all_snr.keras"
STATS_PATH = PROJECT_DIR / "normalization_stats_all_snr.npz"
RESULTS_PATH = PROJECT_DIR / "snr_accuracy_results.csv"
EVALUATION_PATH = PROJECT_DIR / "evaluation_results.json"

DATASET_PATH = (
    Path.home() / "Downloads" / "RML2016.10a_dict_optimized.pkl"
)

CLASSES = ["BPSK", "QPSK", "QAM16"]


# =========================================================
# 2. WEBSITE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Satellite Modulation Classification",
    page_icon="📡",
    layout="wide"
)

st.title("📡 AI-Based Satellite Modulation Classification")

st.write(
    "A software-based satellite communication demonstration "
    "using a Convolutional Neural Network (CNN) to classify "
    "BPSK, QPSK and 16-QAM signals."
)

st.info(
    "This project uses RadioML dataset signals. "
    "It is not connected to a live satellite receiver."
)


# =========================================================
# 3. LOAD THE TRAINED CNN MODEL
# =========================================================

@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)


if not MODEL_PATH.exists():
    st.error(f"Trained model not found: {MODEL_PATH}")
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
    st.success("Trained CNN model loaded successfully!")

except Exception as error:
    st.error(f"Could not load the CNN model: {error}")
    st.stop()


col1, col2 = st.columns(2)

with col1:
    st.subheader("Model Input Shape")
    st.code(str(model.input_shape))

with col2:
    st.subheader("Model Output Shape")
    st.code(str(model.output_shape))


# =========================================================
# 4. LOAD NORMALIZATION VALUES
# =========================================================

@st.cache_data
def load_stats(path):
    with np.load(path) as data:
        mean = data["mean"].copy()
        std = data["std"].copy()

    return mean, std


if not STATS_PATH.exists():
    st.error(f"Normalization file not found: {STATS_PATH}")
    st.stop()

try:
    mean, std = load_stats(str(STATS_PATH))
    std = np.maximum(std, 1e-7)
    st.success("Normalization values loaded successfully!")

except Exception as error:
    st.error(f"Could not load normalization values: {error}")
    st.stop()


# =========================================================
# 5. MODEL EVALUATION RESULTS
# =========================================================

st.divider()
st.header("📈 CNN Test Evaluation")

if EVALUATION_PATH.exists():

    try:
        with open(EVALUATION_PATH, "r", encoding="utf-8") as file:
            evaluation = json.load(file)

        test_accuracy = float(evaluation["accuracy"])
        cm_values = evaluation["confusion_matrix"]
        report = evaluation["classification_report"]

        metric1, metric2, metric3 = st.columns(3)

        with metric1:
            st.metric(
                "Overall Test Accuracy",
                f"{test_accuracy:.2f}%"
            )

        with metric2:
            st.metric("Test Signals", "900")

        with metric3:
            st.metric("Modulation Classes", "3")

        st.subheader("Confusion Matrix")

        cm = pd.DataFrame(
            cm_values,
            index=CLASSES,
            columns=CLASSES
        )

        st.caption(
            "Rows represent actual classes. "
            "Columns represent predicted classes."
        )

        st.dataframe(cm, use_container_width=True)

        st.subheader("Classification Report")

        report_rows = []

        for class_name in CLASSES:
            report_rows.append({
                "Modulation": class_name,
                "Precision": report[class_name]["precision"],
                "Recall": report[class_name]["recall"],
                "F1-score": report[class_name]["f1-score"],
                "Support": int(report[class_name]["support"])
            })

        report_df = pd.DataFrame(report_rows)

        for column in ["Precision", "Recall", "F1-score"]:
            report_df[column] = report_df[column].round(3)

        st.dataframe(report_df, use_container_width=True)

        st.caption(
            "Precision, recall and F1-score range from 0 to 1. "
            "Higher values generally indicate better performance."
        )

    except Exception as error:
        st.error(f"Could not display evaluation results: {error}")

else:
    st.warning(
        "Evaluation results were not found. "
        "Run evaluate_cnn.py to generate evaluation_results.json."
    )


# =========================================================
# 6. ACCURACY AT DIFFERENT SNR LEVELS
# =========================================================

st.divider()
st.header("📊 Overall Accuracy at Different SNR Levels")

if RESULTS_PATH.exists():

    try:
        results = pd.read_csv(RESULTS_PATH)

        st.dataframe(results, use_container_width=True)

        if (
            "SNR_dB" in results.columns
            and "Overall_Accuracy_Percent" in results.columns
        ):
            chart_data = results[
                ["SNR_dB", "Overall_Accuracy_Percent"]
            ].copy()

            chart_data["SNR_dB"] = pd.to_numeric(
                chart_data["SNR_dB"]
            )

            chart_data = chart_data.sort_values("SNR_dB")
            chart_data = chart_data.set_index("SNR_dB")

            st.line_chart(chart_data)

            st.caption(
                "Accuracy is generally higher at larger SNR values, "
                "where the signal is clearer relative to noise."
            )

        else:
            st.warning(
                "The CSV does not contain the expected SNR and "
                "accuracy column names."
            )

    except Exception as error:
        st.error(f"Could not display SNR results: {error}")

else:
    st.warning(f"SNR results CSV not found: {RESULTS_PATH}")


# =========================================================
# 7. LOAD THE RADIOML DATASET
# =========================================================

@st.cache_data
def load_dataset(path):
    with open(path, "rb") as file:
        return pickle.load(file, encoding="latin1")


# =========================================================
# 8. TRY THE MODULATION CLASSIFIER
# =========================================================

st.divider()
st.header("🔬 Try the Modulation Classifier")

st.write(
    "Select a modulation class, an SNR level and a signal sample. "
    "The trained CNN will predict its modulation class."
)

if not DATASET_PATH.exists():

    st.error(
        f"RadioML dataset not found at:\n\n{DATASET_PATH}"
    )

    st.info(
        "Make sure RML2016.10a_dict_optimized.pkl is in your "
        "Windows Downloads folder."
    )

else:

    try:
        dataset = load_dataset(str(DATASET_PATH))

    except Exception as error:
        st.error(f"Could not load the RadioML dataset: {error}")
        st.stop()

    available_snrs = sorted({
        key[1]
        for key in dataset
        if isinstance(key, tuple)
        and len(key) == 2
        and key[0] in CLASSES
        and all((modulation, key[1]) in dataset for modulation in CLASSES)
    })

    if not available_snrs:
        st.error("No common SNR levels were found for all classes.")
        st.stop()

    col1, col2, col3 = st.columns(3)

    with col1:
        selected_class = st.selectbox(
            "Select modulation class",
            CLASSES
        )

    with col2:
        default_snr_index = (
            available_snrs.index(0) if 0 in available_snrs else 0
        )

        selected_snr = st.selectbox(
            "Select SNR (dB)",
            available_snrs,
            index=default_snr_index
        )

    signal_key = (selected_class, selected_snr)
    sample_count = len(dataset[signal_key])

    with col3:
        sample_index = st.number_input(
            "Signal sample index",
            min_value=0,
            max_value=sample_count - 1,
            value=0,
            step=1
        )

    if st.button("🚀 Classify Signal", type="primary"):

        try:
            signal = np.asarray(
                dataset[signal_key][int(sample_index)],
                dtype=np.float32
            )

            if signal.ndim != 2:
                st.error(
                    f"Unexpected signal dimensions: {signal.shape}"
                )

            else:
                if signal.shape == (2, 128):
                    signal = signal.T

                elif signal.shape != (128, 2):
                    st.error(
                        f"Unexpected signal shape: {signal.shape}. "
                        "Expected (2, 128) or (128, 2)."
                    )
                    st.stop()

                # Apply the same normalization values as training.
                try:
                    signal = (signal - mean) / std

                except ValueError as error:
                    st.error(
                        "Normalization dimensions do not match "
                        f"the signal: {error}"
                    )
                    st.stop()

                model_input = np.expand_dims(signal, axis=0)

                expected_shape = tuple(model.input_shape[1:])
                actual_shape = tuple(model_input.shape[1:])

                if actual_shape != expected_shape:
                    st.error(
                        f"Model expects {expected_shape}, "
                        f"but the prepared signal has {actual_shape}."
                    )

                elif model.output_shape[-1] != len(CLASSES):
                    st.error(
                        "The model output does not match the three "
                        "modulation classes."
                    )

                else:
                    scores = model.predict(
                        model_input,
                        verbose=0
                    )[0]

                    predicted_index = int(np.argmax(scores))
                    predicted_class = CLASSES[predicted_index]

                    score_percent = float(scores[predicted_index]) * 100

                    st.subheader("Prediction Result")

                    result_col1, result_col2 = st.columns(2)

                    with result_col1:
                        st.metric(
                            "Predicted Modulation",
                            predicted_class
                        )

                    with result_col2:
                        st.metric(
                            "Selected SNR",
                            f"{selected_snr} dB"
                        )

                    st.write(
                        f"Reference dataset class: **{selected_class}**"
                    )

                    if predicted_class == selected_class:
                        st.success("Prediction matches the dataset label.")

                    else:
                        st.warning(
                            "Prediction does not match the dataset label."
                        )

                    st.write(
                        f"Model score for predicted class: "
                        f"**{score_percent:.2f}%**"
                    )

                    score_table = pd.DataFrame({
                        "Modulation": CLASSES,
                        "Model score (%)": scores * 100
                    })

                    st.subheader("Scores for Each Modulation Class")

                    st.bar_chart(
                        score_table.set_index("Modulation")
                    )

                    st.caption(
                        "Scores are model outputs, not necessarily "
                        "calibrated probabilities. This sample may "
                        "have been included in training."
                    )

        except Exception as error:
            st.error(f"Prediction failed: {error}")


# =========================================================
# 9. FOOTER
# =========================================================

st.divider()

st.caption(
    "Academic software prototype using the RadioML dataset and a "
    "trained CNN. It is not connected to a live satellite receiver."
)
