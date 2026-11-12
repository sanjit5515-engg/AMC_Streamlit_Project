import tensorflow as tf
from pathlib import Path
from custom_layers import SelectiveSSM, MambaBlock

MODEL_DIR = Path("models")

custom_objects = {
    "SelectiveSSM": SelectiveSSM,
    "MambaBlock": MambaBlock
}

model_files = [
    "CNN_Mamba_Range_1.keras",
    "CNN_Mamba_Range_2.keras",
    "CNN_Mamba_Range_3.keras",
    "CNN_Mamba_Range_4.keras",
    "CNN_Mamba_Range_5.keras",
    "CNN_Mamba_Range_6.keras"
]

print("TensorFlow version:", tf.__version__)
print("\nStarting model testing...\n")

for filename in model_files:

    model_path = MODEL_DIR / filename

    print("-" * 55)
    print("Testing:", filename)

    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False,
            safe_mode=False
        )

        dummy_input = tf.zeros((1, 128, 2))
        prediction = model.predict(dummy_input, verbose=0)

        print("Status: SUCCESS")
        print("Input shape:", model.input_shape)
        print("Output shape:", prediction.shape)
        print("Predicted class index:", prediction.argmax(axis=1)[0])

        del model

    except Exception as e:
        print("Status: FAILED")
        print("Error:", str(e))

print("\nModel testing completed.")