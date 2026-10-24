
from pathlib import Path
import json

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
SAMPLES_PATH = PROJECT_DIR / "classifier_samples.npz"

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
    st.error(f"Trained CNN model not found: {MODEL_PATH}")
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

    # Avoid division by zero if any standard deviation is zero.
    std = np.where(std == 0, 1.0, std)

    return mean, std


if not STATS_PATH.exists():
    st.error(f"Normalization file not found: {STATS_PATH}")
    st.stop()

try:
    mean, std = load_stats(str(STATS_PATH))
    st.success("Normalization values loaded successfully!")

except Exception as error:
    st.error(f"Could not load normalization values: {error}")
    st.stop()


# =========================================================
# 5. DISPLAY MODEL EVALUATION RESULTS
# =========================================================

st.divider()
st.header("📊 CNN Model Evaluation")

if EVALUATION_PATH.exists():

    try:
        with open(EVALUATION_PATH, "r", encoding="utf-8") as file:
            evaluation = json.load(file)

        # Display common evaluation metrics when available.
        metric_columns = st.columns(3)

        accuracy = evaluation.get(
            "test_accuracy",
            evaluation.get("accuracy")
        )

        loss = evaluation.get("test_loss", evaluation.get("loss"))

        sample_count = evaluation.get(
            "test_samples",
            evaluation.get("num_test_samples")
        )

        with metric_columns[0]:
            if accuracy is not None:
                accuracy_value = float(accuracy)
                if accuracy_value <= 1:
                    accuracy_value *= 100

                st.metric("Test Accuracy", f"{accuracy_value:.2f}%")
            else:
                st.metric("Test Accuracy", "See report below")

        with metric_columns[1]:
            if loss is not None:
                st.metric("Test Loss", f"{float(loss):.4f}")
            else:
                st.metric("Test Loss", "N/A")

        with metric_columns[2]:
            if sample_count is not None:
                st.metric("Test Samples", str(sample_count))
            else:
                st.metric("Test Samples", "N/A")

        # Display a confusion matrix if the JSON contains one.
        confusion = evaluation.get(
            "confusion_matrix",
            evaluation.get("confusion")
        )

        if confusion is not None:
            st.subheader("Confusion Matrix")

            confusion_array = np.asarray(confusion)

            if confusion_array.ndim == 2:
                confusion_df = pd.DataFrame(
                    confusion_array,
                    index=CLASSES[:confusion_array.shape[0]],
                    columns=CLASSES[:confusion_array.shape[1]]
                )
                st.dataframe(confusion_df)

        # Display a classification report if available.
        report = evaluation.get("classification_report")

        if report is not None:
            st.subheader("Classification Report")

            if isinstance(report, dict):
                try:
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df)
                except (ValueError, TypeError):
                    st.json(report)
            else:
                st.text(str(report))

    except Exception as error:
        st.warning(f"Could not display evaluation JSON: {error}")

else:
    st.warning(
        "Evaluation results file is not available. "
        "The CNN model and classifier may still work."
    )


# =========================================================
# 6. DISPLAY ACCURACY AT DIFFERENT SNR LEVELS
# =========================================================

st.divider()
st.header("📈 Accuracy at Different SNR Levels")

if RESULTS_PATH.exists():

    try:
        snr_results = pd.read_csv(RESULTS_PATH)

        if not snr_results.empty:
            st.subheader("SNR Accuracy Results")
            st.dataframe(snr_results, use_container_width=True)

            # Find the SNR column and accuracy column.
            snr_column = next(
                (
                    column for column in snr_results.columns
                    if "snr" in column.lower()
                ),
                None
            )

            accuracy_column = next(
                (
                    column for column in snr_results.columns
                    if "accuracy" in column.lower()
                ),
                None
            )

            if snr_column and accuracy_column:
                chart_data = snr_results[
                    [snr_column, accuracy_column]
                ].copy()

                chart_data[snr_column] = pd.to_numeric(
                    chart_data[snr_column], errors="coerce"
                )

                chart_data[accuracy_column] = pd.to_numeric(
                    chart_data[accuracy_column], errors="coerce"
                )

                chart_data = chart_data.dropna().set_index(snr_column)

                if not chart_data.empty:
                    st.subheader("SNR vs Accuracy")
                    st.line_chart(chart_data)

            else:
                st.info(
                    "The CSV columns could not be identified "
                    "automatically for the graph."
                )

    except Exception as error:
        st.warning(f"Could not load SNR results: {error}")

else:
    st.warning("SNR accuracy results CSV file was not found.")


# =========================================================
# 7. INTERACTIVE MODULATION CLASSIFIER
# =========================================================

st.divider()
st.header("🛰️ Interactive Modulation Classifier")

st.write(
    "Choose a modulation class and SNR level, then classify "
    "one of the saved RadioML signal samples."
)

if not SAMPLES_PATH.exists():

    st.error(
        "Classifier sample file not found. "
        "Make sure classifier_samples.npz is in the project folder."
    )

else:

    try:
        with np.load(SAMPLES_PATH, allow_pickle=False) as sample_data:
            signals = sample_data["signals"].copy()
            labels = sample_data["labels"].astype(str)
            snrs = sample_data["snrs"].copy()

    except Exception as error:
        st.error(f"Could not load classifier samples: {error}")
        st.stop()

    # Check that the sample file contains usable data.
    if len(signals) == 0:
        st.error("The classifier sample file contains no signals.")
        st.stop()

    available_snrs = sorted(
        int(value) for value in np.unique(snrs)
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        selected_class = st.selectbox(
            "Select modulation class",
            CLASSES
        )

    with col2:
        default_snr_index = (
            available_snrs.index(0)
            if 0 in available_snrs
            else 0
        )

        selected_snr = st.selectbox(
            "Select SNR (dB)",
            available_snrs,
            index=default_snr_index
        )

    # Find saved signals matching the selected class and SNR.
    matching_indices = np.where(
        (labels == selected_class) &
        (snrs == selected_snr)
    )[0]

    if len(matching_indices) == 0:

        st.warning(
            "No saved signals are available for this "
            "modulation class and SNR."
        )

    else:

        with col3:
            sample_index = st.number_input(
                "Signal sample index",
                min_value=0,
                max_value=len(matching_indices) - 1,
                value=0,
                step=1
            )

        if st.button("🚀 Classify Signal", type="primary"):

            try:
                signal = np.asarray(
                    signals[matching_indices[int(sample_index)]],
                    dtype=np.float32
                )

                # The CNN expects 128 time steps and 2 channels.
                if signal.shape == (2, 128):
                    signal = signal.T

                if signal.shape != (128, 2):
                    st.error(
                        f"Unexpected signal shape: {signal.shape}. "
                        "Expected (128, 2)."
                    )
                    st.stop()

                # Apply the normalization values used for training.
                signal = (signal - mean) / std

                # Add a batch dimension for the CNN.
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
                        "The CNN output does not match "
                        "the three modulation classes."
                    )

                else:

                    # Predict the modulation class.
                    scores = model.predict(
                        model_input,
                        verbose=0
                    )[0]

                    predicted_index = int(np.argmax(scores))
                    predicted_class = CLASSES[predicted_index]

                    score_percent = (
                        float(scores[predicted_index]) * 100
                    )

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
                        st.success(
                            "Prediction matches the dataset label."
                        )
                    else:
                        st.warning(
                            "Prediction does not match the dataset label."
                        )

                    st.metric(
                        "Model score for predicted class",
                        f"{score_percent:.2f}%"
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
                        "Scores are model outputs and are not "
                        "necessarily calibrated probabilities. "
                        "The saved samples may overlap with data "
                        "used during model training."
                    )

            except Exception as error:
                st.error(f"Prediction failed: {error}")


# =========================================================
# 8. FOOTER
# =========================================================

st.divider()

st.caption(
    "Academic software prototype using the RadioML dataset and a "
    "trained CNN. It is not connected to a live satellite receiver."
)
