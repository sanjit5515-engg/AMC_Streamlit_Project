from pathlib import Path
import io

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns

from custom_layers import SelectiveSSM, MambaBlock
from model_config import (
    CLASS_NAMES, SNR_TO_MODEL, MODEL_PATHS, RANGE_SNR_VALUES, BASE_DIR
)

# Updated SNR grouping used by the retrained models.
# Keep this mapping synchronized with model_config.py.
SNR_TO_MODEL.clear()
SNR_TO_MODEL.update({
    10: "Range_1", 12: "Range_1", 14: "Range_1",
    16: "Range_1", 18: "Range_1",
    6: "Range_2", 8: "Range_2",
    0: "Range_3", 2: "Range_3", 4: "Range_3",
    -4: "Range_4", -2: "Range_4",
    -10: "Range_5", -8: "Range_5", -6: "Range_5",
    -20: "Range_6", -18: "Range_6", -16: "Range_6",
    -14: "Range_6", -12: "Range_6",
})

RANGE_SNR_VALUES.clear()
RANGE_SNR_VALUES.update({
    "Range_1": [10, 12, 14, 16, 18],
    "Range_2": [6, 8],
    "Range_3": [0, 2, 4],
    "Range_4": [-4, -2],
    "Range_5": [-10, -8, -6],
    "Range_6": [-20, -18, -16, -14, -12],
})

st.set_page_config(
    page_title="RadioML AMC | CNN–Mamba",
    page_icon="📡",
    layout="wide"
)

st.title("📡 Automatic Modulation Classification")
st.caption("RadioML 2016.10a • Multi-Scale CNN + SE Attention + Mamba")

st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Select a page",
    ["Home", "Signal Classification", "Model Information", "Evaluation"]
)

@st.cache_resource(show_spinner="Loading selected CNN–Mamba model...")
def load_model(model_path):
    custom_objects = {
        "SelectiveSSM": SelectiveSSM,
        "MambaBlock": MambaBlock,
    }
    return tf.keras.models.load_model(
        str(model_path),
        custom_objects=custom_objects,
        compile=False,
        safe_mode=False
    )

def check_model_files():
    missing = [str(p) for p in MODEL_PATHS.values() if not p.exists()]
    return missing

def prepare_iq(array):
    """Convert one IQ record to the model's required (1, 128, 2) shape."""
    x = np.asarray(array, dtype=np.float32)
    x = np.squeeze(x)

    if x.shape == (2, 128):
        x = x.T
    elif x.shape != (128, 2):
        raise ValueError(
            f"Expected one signal with shape (128, 2) or (2, 128); got {x.shape}."
        )

    if not np.isfinite(x).all():
        raise ValueError("Signal contains NaN or infinite values.")

    return x, np.expand_dims(x, axis=0)

def read_signal(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".npy":
        return np.load(uploaded_file, allow_pickle=False)
    if suffix == ".csv":
        frame = pd.read_csv(uploaded_file)
        # Prefer explicitly named I and Q columns, otherwise require two numeric columns.
        lower = {str(c).strip().lower(): c for c in frame.columns}
        if "i" in lower and "q" in lower:
            frame = frame[[lower["i"], lower["q"]]]
        else:
            frame = frame.select_dtypes(include=[np.number])
        if frame.shape[1] != 2:
            raise ValueError("CSV must contain exactly two numeric I/Q columns.")
        return frame.to_numpy(dtype=np.float32)
    raise ValueError("Upload a .npy or .csv file.")

def plot_iq(signal):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(signal[:, 0], label="I")
    axes[0].plot(signal[:, 1], label="Q", alpha=0.8)
    axes[0].set_title("I/Q time-domain waveform")
    axes[0].set_xlabel("Sample")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].scatter(signal[:, 0], signal[:, 1], s=18, alpha=0.7)
    axes[1].set_title("I/Q constellation (raw samples)")
    axes[1].set_xlabel("In-phase (I)")
    axes[1].set_ylabel("Quadrature (Q)")
    axes[1].grid(alpha=0.25)
    axes[1].axis("equal")
    fig.tight_layout()
    return fig

if page == "Home":
    st.subheader("Project overview")
    st.write(
        "This application routes an input I/Q signal to one of six separately "
        "trained CNN–Mamba models according to the supplied SNR."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Modulation classes", "11")
    c2.metric("Input shape", "128 × 2")
    c3.metric("SNR-specific models", "6")

    st.info(
        "Select one of the 20 trained SNR values. "
        
    )
    st.markdown("### Getting started")
    st.markdown(
        "1. Place all six `.keras` files in the `models` folder.\n"
        "2. Open **Signal Classification**.\n"
        "3. Select the known SNR and upload one 128-sample I/Q record.\n"
        "4. Review the predicted modulation, confidence, waveform and constellation."
    )

elif page == "Signal Classification":
    st.subheader("Classify an I/Q signal")
    missing = check_model_files()
    if missing:
        st.error("The following model files are missing:")
        for path in missing:
            st.code(path)
        st.warning(
            "Copy all six CNN_Mamba_Range_*.keras files into the project's models folder."
        )
        st.stop()

    snr = st.selectbox(
        "Known signal-to-noise ratio (SNR)",
        options=sorted(SNR_TO_MODEL.keys()),
        format_func=lambda v: f"{v} dB"
    )
    selected_range = SNR_TO_MODEL[snr]
    st.success(
        f"Selected model: {selected_range} "
        f"(trained SNRs: {RANGE_SNR_VALUES[selected_range]} dB)"
    )

    uploaded = st.file_uploader(
        "Upload one I/Q signal (.npy or .csv)",
        type=["npy", "csv"],
        help="Expected one record with 128 rows and two columns: I and Q."
    )

    if uploaded is not None:
        try:
            raw = read_signal(uploaded)
            signal, model_input = prepare_iq(raw)
            with st.spinner(f"Running {selected_range}..."):
                model = load_model(MODEL_PATHS[selected_range])
                probabilities = model.predict(model_input, verbose=0)[0]

            if probabilities.shape[0] != len(CLASS_NAMES):
                st.error(
                    f"Model returned {len(probabilities)} outputs, but "
                    f"{len(CLASS_NAMES)} class labels are configured. "
                    "Check CLASS_NAMES against the original training `mods` order."
                )
                st.stop()

            predicted_index = int(np.argmax(probabilities))
            predicted_class = CLASS_NAMES[predicted_index]
            confidence = float(probabilities[predicted_index]) * 100

            st.markdown("### Prediction")
            a, b, c = st.columns(3)
            a.metric("Predicted modulation", predicted_class)
            b.metric("Confidence", f"{confidence:.2f}%")
            c.metric("SNR", f"{snr} dB")

            left, right = st.columns(2)
            with left:
                st.pyplot(plot_iq(signal), use_container_width=True)
            with right:
                scores = pd.DataFrame({
                    "Modulation": CLASS_NAMES,
                    "Probability (%)": probabilities * 100
                }).sort_values("Probability (%)", ascending=False)
                st.markdown("#### Class probabilities")
                st.bar_chart(scores.set_index("Modulation"))

            st.markdown("#### Ranked predictions")
            st.dataframe(
                scores.head(11).reset_index(drop=True).style.format(
                    {"Probability (%)": "{:.3f}%"}
                ),
                use_container_width=True
            )
            st.caption(
                "Confidence is the model's softmax output; it is not necessarily "
                "a calibrated probability of correctness."
            )
        except Exception as exc:
            st.error(f"Could not classify this signal: {exc}")

elif page == "Model Information":
    st.subheader("Six SNR-specific CNN–Mamba models")
    rows = []
    for name, snrs in RANGE_SNR_VALUES.items():
        path = MODEL_PATHS[name]
        rows.append({
            "Model": name,
            "SNR values (dB)": ", ".join(map(str, snrs)),
            "File": path.name,
            "Status": "Available" if path.exists() else "Missing"
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.markdown("### Architecture")
    st.markdown(
        "- Input: 128 time steps × 2 I/Q channels\n"
        "- Multi-scale Conv1D branches: kernel sizes 3, 5 and 7\n"
        "- CNN feature fusion and average pooling\n"
        "- Squeeze-and-Excitation channel attention\n"
        "- Three Mamba blocks (d_model=128, d_state=16, d_conv=4)\n"
        "- Global average pooling and dense classification head\n"
        "- Output: 11 modulation classes"
    )
    

elif page == "Evaluation":

    st.subheader("CNN-Mamba Model Evaluation")

    st.write(
        "Performance evaluation of six CNN-Mamba models "
        "using independent test datasets."
    )

    # ==========================================
    # 1. OVERALL PERFORMANCE
    # ==========================================

    st.markdown("## 1. Overall Model Performance")

    overall_file = st.file_uploader(
        "Upload Overall Performance CSV",
        type=["csv"],
        key="overall_csv"
    )

    if overall_file is not None:

        overall_df = pd.read_csv(overall_file)

        st.dataframe(
            overall_df,
            use_container_width=True
        )

        st.markdown("### Accuracy Comparison")

        accuracy_chart = overall_df[
            ["SNR Range", "Accuracy (%)"]
        ].set_index("SNR Range")

        st.bar_chart(accuracy_chart)

        st.markdown("### Macro F1-Score Comparison")

        f1_chart = overall_df[
            ["SNR Range", "Macro F1 Score"]
        ].set_index("SNR Range")

        st.bar_chart(f1_chart)

    st.markdown("---")

    # ==========================================
    # 2. CLASSIFICATION REPORT
    # ==========================================

    st.markdown("## 2. Classification Report")

    report_file = st.file_uploader(
        "Upload Classification Report CSV",
        type=["csv"],
        key="report_csv"
    )

    if report_file is not None:

        report_df = pd.read_csv(report_file)

        selected_range = st.selectbox(
            "Select SNR Range",
            report_df["SNR Range"].unique(),
            key="report_select"
        )

        filtered_report = report_df[
            report_df["SNR Range"] == selected_range
        ]

        st.markdown(
            f"### Classification Report - {selected_range}"
        )

        st.dataframe(
            filtered_report,
            use_container_width=True
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Average Precision",
            f"{filtered_report['Precision'].mean() * 100:.2f}%"
        )

        c2.metric(
            "Average Recall",
            f"{filtered_report['Recall'].mean() * 100:.2f}%"
        )

        c3.metric(
            "Average F1-Score",
            f"{filtered_report['F1 Score'].mean() * 100:.2f}%"
        )

    st.markdown("---")

    # ==========================================
    # 3. CONFUSION MATRIX
    # ==========================================

    st.markdown("## 3. Confusion Matrix")

    confusion_file = st.file_uploader(
        "Upload Confusion_Matrix_Data.csv",
        type=["csv"],
        key="confusion_upload"
    )

    if confusion_file is not None:
        try:
            confusion_df = pd.read_csv(confusion_file)

            # Remove accidental spaces from CSV headers and text values.
            confusion_df.columns = confusion_df.columns.astype(str).str.strip()

            required_columns = [
                "SNR Range", "Actual_Class", "Predicted_Class", "Count"
            ]
            missing_columns = [
                col for col in required_columns if col not in confusion_df.columns
            ]

            if missing_columns:
                st.error(
                    "The CSV is missing required columns: "
                    + ", ".join(missing_columns)
                )
            else:
                for col in ["SNR Range", "Actual_Class", "Predicted_Class"]:
                    confusion_df[col] = (
                        confusion_df[col].astype(str).str.strip()
                    )

                confusion_df["Count"] = pd.to_numeric(
                    confusion_df["Count"], errors="coerce"
                )

                if confusion_df["Count"].isna().any():
                    st.error(
                        "The Count column contains blank or non-numeric values. "
                        "Please check the CSV."
                    )
                elif (confusion_df["Count"] < 0).any():
                    st.error("The Count column cannot contain negative values.")
                else:
                    available_ranges = [
                        f"Range_{i}" for i in range(1, 7)
                        if f"Range_{i}" in confusion_df["SNR Range"].unique()
                    ]

                    if not available_ranges:
                        st.error(
                            "No valid SNR ranges (Range_1 to Range_6) "
                            "were found in the uploaded CSV."
                        )
                    else:
                        selected_range = st.selectbox(
                            "Select SNR Range",
                            available_ranges,
                            key="confusion_range"
                        )

                        filtered_df = confusion_df[
                            confusion_df["SNR Range"] == selected_range
                        ].copy()

                        # Build a complete 11 x 11 matrix in model class order.
                        cm = filtered_df.pivot_table(
                            index="Actual_Class",
                            columns="Predicted_Class",
                            values="Count",
                            aggfunc="sum",
                            fill_value=0
                        )
                        cm = cm.reindex(
                            index=CLASS_NAMES,
                            columns=CLASS_NAMES,
                            fill_value=0
                        )

                        # Show the confusion matrix as a regular data table.
                        # This avoids all plotting-library and rendering issues.
                        st.markdown(f"### Confusion Matrix - {selected_range}")
                        st.caption(
                            "Rows = actual modulation classes; "
                            "columns = predicted modulation classes."
                        )

                        display_cm = cm.copy()
                        display_cm.index.name = "Actual \\ Predicted"
                        display_cm.columns.name = "Predicted Modulation"

                        st.dataframe(
                            display_cm,
                            use_container_width=True,
                            height=440
                        )

                        st.metric(
                            "Total test samples",
                            f"{int(cm.to_numpy().sum()):,}"
                        )

                        # Let the user download the selected 11 x 11 matrix.
                        matrix_csv = display_cm.to_csv().encode("utf-8")
                        st.download_button(
                            label="Download this confusion matrix",
                            data=matrix_csv,
                            file_name=f"{selected_range}_Confusion_Matrix.csv",
                            mime="text/csv",
                            key=f"download_{selected_range}"
                        )

        except Exception as exc:
            st.error(f"Could not display the confusion matrix: {exc}")

    st.markdown("---")

    st.success(
        "CNN-Mamba evaluation results loaded successfully."
    )


st.sidebar.markdown("---")
st.sidebar.caption("CNN–Mamba AMC • RadioML 2016.10a")
