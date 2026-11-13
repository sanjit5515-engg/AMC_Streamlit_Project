# RadioML AMC Streamlit application

## 1. Project structure

```text
AMC_Streamlit_Project/
├── app.py
├── custom_layers.py
├── model_config.py
├── requirements.txt
└── models/
    ├── CNN_Mamba_Range_1.keras
    ├── CNN_Mamba_Range_2.keras
    ├── CNN_Mamba_Range_3.keras
    ├── CNN_Mamba_Range_4.keras
    ├── CNN_Mamba_Range_5.keras
    └── CNN_Mamba_Range_6.keras
```

Copy your six trained `.keras` files into `models/`. The model files are not
bundled in this starter package.

## 2. Python environment (Windows / VS Code)

Use Python 3.10 or 3.11. In the VS Code terminal, from this project folder:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `py -3.11` is unavailable, install Python 3.11 and reopen VS Code. Then select
the `.venv` interpreter using **Python: Select Interpreter**.

## 3. Run

```powershell
python -m streamlit run app.py
```

Streamlit will print a local address, usually `http://localhost:8501`.

## 4. Deployment
The application is deployed using Streamlit Community Cloud.

Application:
https://cnn-mamba-modulation-classification.streamlit.app/

## 5. Input format

Upload one signal as either:

- `.npy`: a NumPy array shaped `(128, 2)` or `(2, 128)`.
- `.csv`: 128 rows and two numeric columns. Columns named `I` and `Q` are preferred.

The app transposes `(2, 128)` to `(128, 2)` and converts to `float32`.
It does not normalize or rescale the signal, because preprocessing must match
the training pipeline exactly.
## 6. Instructions
1. Download an I/Q sample CSV file.
2. Open the deployed Streamlit application.
3. Navigate to Signal Classification.
4. Upload the I/Q sample.
5. Select the corresponding SNR value.
6. View the predicted modulation class.
## 7. Important checks before relying on predictions

1. **Class order:** `CLASS_NAMES` is currently set to the lexicographically sorted
   RadioML 2016.10a labels. Verify it matches the exact `mods` printed by your
   training script. Output index 0 must map to `mods[0]`, and so on.

2. **Model loading:** Custom `SelectiveSSM` and `MambaBlock` layers are provided.
   If loading fails, retain the full traceback and check the TensorFlow/Keras
   version used to save the `.keras` files.

3. **Preprocessing:** Feed raw I/Q samples with the same preprocessing as training.
   Do not apply normalization unless the training pipeline did so.
   
4. **Performance:** The model's softmax maximum is a confidence score, not a
   calibrated guarantee of correctness.

## 8. Evaluation Results
The Evaluation Results folder contains:
- Overall performance metrics
- Classification report
- Confusion matrix data
