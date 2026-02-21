# Tamper-Proof Medical Imaging: A Zero-Watermarking Approach

A medical-grade computer vision and cryptographic pipeline engineered to verify the authenticity of medical scans (DICOM/PNG) without altering original pixel data. This system detects malicious tampering, data loss, and corruption using a robust zero-watermarking architecture.


## 🔬 System Architecture

The pipeline is divided into three core micro-systems:

### 1. Registration & Feature Extraction Engine
* **DICOM Handling:** Robust ingestion of `.dcm` files with percentile-based normalization and CLAHE enhancement.
* **Hybrid ROI-RONI Segmentation:** Isolates the Region of Interest (ROI) from the Region of Non-Interest (RONI) using Otsu thresholding, Sobel edge detection, and morphological operations to prevent diagnostic interference.
* **HVS-Based Feature Extraction:** Extracts robust, perceptually significant features using a DWT-DCT-SVD (Discrete Wavelet, Cosine, and Singular Value Decomposition) pipeline based on Shannon entropy scores.

### 2. Cryptographic Hashing System
* Generates a tamper-evident signature by fusing the extracted visual features with patient metadata.
* **Multi-Layer Security:** Secures the signature through an HMAC hash, which seeds a multi-iteration **Cellular Automaton (CA)** (Rules 30, 45, 105, etc.), ultimately finalized via **SHA3-512**. 
* **Database Integration:** Securely registers the immutable hash to a MongoDB cluster for future verification.

### 3. Verification & Attack Simulation (Evaluation)
* **Live Verification:** Compares recomputed hashes against database records, accurately detecting corruption types (e.g., random erasure, data loss).
* **Automated Tampering Dataset Generator:** Includes a built-in testing suite that applies both *imperceptible attacks* (Gaussian noise, Salt & Pepper, blur) and *destructive attacks* (copy-move, cropping, elastic distortion) to validate pipeline robustness.


## 📊 Performance Metrics
Tested via automated batch processing on a dataset of 450 original and tampered medical images, evaluated against standard cryptographic and visual fidelity metrics:
* **Detection Metrics:** Accuracy, Precision, Recall (TPR), F1 Score, and Specificity.
* **Visual/Hash Metrics:** Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM), Normalized Cross-Correlation (NCC), Bit Error Rate (BER), and the Avalanche Effect.

## 🛠 Tech Stack
* **Image Processing:** OpenCV, Scikit-image, PyWavelets, SciPy, PyDICOM
* **Cryptography:** `hashlib` (SHA3-512), HMAC, Custom Cellular Automata Logic
* **Data & Storage:** MongoDB, Pandas, NumPy
* **Visualization:** Matplotlib
  
<img width="1373" height="769" alt="image" src="https://github.com/user-attachments/assets/b255454d-1450-4be0-884e-1e5fff76e396" />

<img width="660" height="865" alt="image" src="https://github.com/user-attachments/assets/8c7ece90-72e7-4aa9-8e53-bf08498dddc5" />

<img width="1163" height="835" alt="image" src="https://github.com/user-attachments/assets/56356bac-34c9-4ba8-9904-244d8ec9ca41" />

