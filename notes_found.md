# Notes Found

---

## 🔑 Why `recurrent_dropout=0` is recommended

- **GPU-accelerated LSTM kernels (CuDNNLSTM equivalent)**:
    TensorFlow uses **highly optimized fused GPU kernels** for LSTMs/GRUs.
    These are **only used if**:

  - `activation="tanh"`
  - `recurrent_activation="sigmoid"`
  - `unroll=False`
  - **`recurrent_dropout=0`** ✅

        If you set `recurrent_dropout > 0`, TensorFlow **cannot use the fast fused kernel**. Instead, it falls back to a slower, step-by-step implementation in Python — which can be **10–20× slower** on GPU/Metal.

- **Apple Silicon + Metal plugin**:
    The same limitation applies. Apple’s GPU backend accelerates the fused kernel, but recurrent dropout breaks that path, forcing CPU or unfused GPU ops.

---
