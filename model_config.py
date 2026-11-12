from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# These labels follow Python's sorted(mods) ordering from the training script.
# Verify against the printed `mods` list from the original training run.
CLASS_NAMES = [
    "8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK",
    "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"
]

# Explicit SNR membership preserves the exact ranges in the training script.
SNR_TO_MODEL = {
    10: "Range_1",
    12: "Range_1",
    14: "Range_1",
    16: "Range_1",
    18: "Range_1",

    6: "Range_2",
    8: "Range_2",

    0: "Range_3",
    2: "Range_3",
    4: "Range_3",

    -4: "Range_4",
    -2: "Range_4",

    -10: "Range_5",
    -8: "Range_5",
    -6: "Range_5",

    -20: "Range_6",
    -18: "Range_6",
    -16: "Range_6",
    -14: "Range_6",
    -12: "Range_6"
}
MODEL_PATHS = {
    f"Range_{i}": MODEL_DIR / f"CNN_Mamba_Range_{i}.keras"
    for i in range(1, 7)
}

RANGE_SNR_VALUES = {
    "Range_1": [10, 12, 14, 16, 18],
    "Range_2": [6, 8],
    "Range_3": [0, 2, 4],
    "Range_4": [-4, -2],
    "Range_5": [-10, -8, -6],
    "Range_6": [-20, -18, -16, -14, -12]
}
