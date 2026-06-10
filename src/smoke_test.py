r"""Environment smoke test.

Confirms the stack imports and that a tiny neural network can actually train.
We fabricate a trivial problem the network must learn: y = 2*x0 + 3*x1 - x2.
If the loss drops sharply, the full train loop (forward pass -> loss -> backprop)
works end to end on your machine.

Run:
    .\.venv\Scripts\python.exe src\smoke_test.py
"""

import platform

import numpy as np
import tensorflow as tf
from tensorflow import keras


def main() -> None:
    print(f"Python      : {platform.python_version()}")
    print(f"NumPy       : {np.__version__}")
    print(f"TensorFlow  : {tf.__version__}")
    print(f"Keras       : {keras.__version__}")
    print(f"GPUs visible: {tf.config.list_physical_devices('GPU') or 'none (CPU only)'}")
    print("-" * 48)

    # Reproducibility: fix the random seed so runs are comparable.
    keras.utils.set_random_seed(42)

    # Make a toy dataset: 1000 rows, 3 input features, a known linear target.
    x = np.random.normal(size=(1000, 3)).astype("float32")
    true_weights = np.array([2.0, 3.0, -1.0], dtype="float32")
    y = x @ true_weights

    # A minimal feed-forward network: one hidden layer, then a single output.
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(3,)),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    history = model.fit(x, y, epochs=20, batch_size=32, verbose=0, validation_split=0.2)

    start_loss = history.history["loss"][0]
    end_loss = history.history["loss"][-1]
    print(f"train loss: {start_loss:8.4f}  ->  {end_loss:8.4f}")
    print(f"val   mae : {history.history['val_mae'][-1]:8.4f}")

    if end_loss < start_loss * 0.5:
        print("\nOK: the network learned. Environment is good to go.")
    else:
        print("\nWARNING: loss did not drop as expected; something may be off.")


if __name__ == "__main__":
    main()
