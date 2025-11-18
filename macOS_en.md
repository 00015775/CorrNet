# CorrNet — Continuous Sign Language Recognition (macOS Adaptation Guide)

This documentation provides a complete operational workflow for successfully reproducing the **CorrNet Continuous Sign Language Recognition System (CSL-Daily Model)** on **macOS / Apple Silicon (M1 / M2 / M3)**.

Since the official project is primarily designed for **Linux + CUDA**, the original code encounters several compatibility issues on macOS, such as:

- MPS does not support certain 3D pooling operations.
- `ctcdecode` cannot be compiled on macOS.
- The dictionary structure of CSL-Daily causes decoding failures.
- Some modules do not account for automatic CPU/MPS switching.
- `demo.py` cannot correctly read image sequences on macOS.

This reproduction document provides a **complete workflow capable of running from scratch on macOS**, including:

- Environment configuration.
- Decoder adaptation (`pyctcdecode` replacing `ctcdecode`).
- Automatic MPS / CPU selection.
- Complete rewrites of `demo.py` + `decode.py`.
- Multi-image input and multi-video format handling.
- Result decoding stability optimization.
- Troubleshooting common errors.

You simply need to follow the document from top to bottom to run the project, load the CSL-Daily model, and perform continuous sign language recognition using image sequences or videos.

---

# Table of Contents

1. Project Introduction (macOS Compatibility Guide)

2. Environment Preparation
    2.1 System Requirements (macOS / Apple Silicon)
    2.2 Conda Environment Creation
    2.3 Key Dependency Versions (torch, pyctcdecode, decord…)
    2.4 pip freeze Example (Actual working environment)

3. Why the Original Version Cannot Run on macOS
    3.1 MPS does not support `max_pool3d`
    3.2 `ctcdecode` cannot be compiled
    3.3 CSL-Daily `gloss_dict` special structure
    3.4 Solution Strategy Summary

4. Mac Adaptation Changes (Core Modifications)
    4.1 Device Selection Logic (MPS → CUDA → CPU)
    4.2 Enabling MPS Fallback
    4.3 Multi-image Sorting + Safe Handling
    4.4 `decode.py` Modifications (`pyctcdecode` + unicode vocab)
    4.5 `demo.py` Modifications (Device migration, video loading, exception handling)

5. Complete `decode.py` (macOS Compatible Version — Copy Ready)

6. Complete `demo.py` (macOS Compatible Version — Copy Ready)

7. How to Run the Project
    7.1 Startup Command (Including MPS fallback)
    7.2 Gradio Usage Instructions
    7.3 Multi-image/Video Input Specifications

---

# 1. Project Introduction (macOS Compatibility Guide)

This project is based on **CorrNet: Correspondence-Aware Network for Continuous Sign Language Recognition**, and uses the **CSL-Daily** pre-trained model released by the authors for Continuous Sign Language Recognition (CSLR).

The official running environment for the original project is:

- **Ubuntu + CUDA (NVIDIA GPU)**
- **ctcdecode (Requires GNU toolchain and Linux environment)**
- **PyTorch with GPU acceleration**

However, **macOS (especially Apple Silicon M1/M2/M3) has multiple incompatibilities with the original environment**, causing the official code to fail directly:

---

## 1.1 Reasons Why the Original Project Cannot Run Directly on macOS

### (1) MPS Does Not Support 3D Pooling
The frontend visual encoder of CorrNet contains operators that require GPU optimization. The MPS Metal backend on macOS currently does not support certain 3D pooling operations (especially `max_pool3d` and specific stride/padding combinations).

Therefore, the model will error out immediately during the intermediate feature extraction stage on Mac.

---

### (2) `ctcdecode` Cannot Be Installed
The original project uses **ctcdecode** (an acceleration library based on C++ & OpenMP).
On macOS:
- OpenMP fails to compile successfully.
- The official `ctcdecode` repository does not provide pre-compiled versions for macOS.

This causes the decode stage to be completely inoperable.

---

### (3) Special Structure of CSL-Daily's `gloss_dict.npy`
The dictionary format for CSL-Daily looks like this:

```json
{
  "他": [1],
  "有": [2],
  "什么": [3]
}