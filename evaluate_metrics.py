# 📊 BATCH VERIFICATION & METRICS CALCULATION
import os, cv2, numpy as np, matplotlib.pyplot as plt
from pymongo import MongoClient
import hmac, hashlib
from hashlib import sha3_512
import pywt
from scipy.fftpack import dct
from skimage.util import view_as_blocks
import pydicom
from datetime import datetime
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# 🔗 MongoDB Setup - MASKED FOR SECURITY
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://<username>:<password>@cluster0.ypyd3es.mongodb.net/?retryWrites=true&w=majority")
DATABASE_NAME = "medical_image_db"
COLLECTION_NAME = "image_hashes"

client = MongoClient(MONGO_URI)
collection = client[DATABASE_NAME][COLLECTION_NAME]

# ⚙️ Hashing & CA Functions
def features_to_binary(features, precision=10): return ''.join(format(int(val * (2 ** precision)), f'0{precision+4}b') for val in features)
def flatten_metadata(metadata): return '|'.join(f"{k}:{v}" for k, v in sorted(metadata.items()))
def generate_hmac(data_string, secret="PES_MEDIC_CHAIN_2025"): return hmac.new(secret.encode(), data_string.encode(), hashlib.sha256).hexdigest()
def hex_to_binary_string(hex_str): return bin(int(hex_str, 16))[2:].zfill(len(hex_str)*4)
def build_rule_dict(rule_number): return {k: format(rule_number, '08b')[i] for i, k in enumerate(['111','110','101','100','011','010','001','000'])}
def evolve_ca(seed, rule_dict, iterations=30):
    current = seed
    for _ in range(iterations):
        current = ''.join(rule_dict[current[i-1] + current[i] + current[(i+1)%len(current)]] for i in range(len(current)))
    return current
def select_ca_rule(hmac_hash): return [30, 45, 105, 129, 165, 195, 225][int(hmac_hash[:8], 16) % 7]
def generate_final_hash(features, metadata):
    hmac_hash = generate_hmac(features_to_binary(features) + flatten_metadata(metadata))
    return sha3_512(evolve_ca(hex_to_binary_string(hmac_hash), build_rule_dict(select_ca_rule(hmac_hash))).encode()).hexdigest()

# 🧠 Feature Extraction
def extract_features(image):
    coeffs2 = pywt.dwt2(cv2.resize(image, (512, 512)), 'db2')
    LL_cropped = coeffs2[0][:coeffs2[0].shape[0] - (coeffs2[0].shape[0] % 4), :coeffs2[0].shape[1] - (coeffs2[0].shape[1] % 4)]
    blocks = view_as_blocks(LL_cropped, block_shape=(4,4))
    features = []
    for i in range(blocks.shape[0]):
        for j in range(blocks.shape[1]):
            dct_block = dct(dct(blocks[i,j].T, norm='ortho').T, norm='ortho')
            try: features.extend(np.linalg.svd(dct_block)[1][:2])
            except: features.extend([np.mean(dct_block), np.std(dct_block)])
    return np.array(features)

# 📝 Metadata
def get_metadata(filename):
    try:
        dcm = pydicom.dcmread(filename, force=True)
        return {"Patient_ID": getattr(dcm, "PatientID", "UNKNOWN"), "Image_ID": getattr(dcm, "SOPInstanceUID", os.path.splitext(filename)[0]), "Scan_Timestamp": datetime.utcnow().isoformat(), "Doctor_ID": "DR_AK_MH001", "Modality": getattr(dcm, "Modality", "UNKNOWN"), "ROI_type": "lungs_full", "ROI_coords": [100, 100, 400, 450], "Image_Resolution": [512, 512]}
    except:
        return {"Patient_ID": f"IMG_{os.path.basename(filename)}", "Image_ID": os.path.basename(filename), "Scan_Timestamp": datetime.utcnow().isoformat(), "Doctor_ID": "DR_AK_MH001", "Modality": "PNG/JPG", "ROI_type": "lungs_full", "ROI_coords": [100, 100, 400, 450], "Image_Resolution": [512, 512]}

# 🔍 Verification
def verify_image(file_path):
    try:
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None: img = pydicom.dcmread(file_path, force=True).pixel_array.astype(np.uint8)
    except:
        img = pydicom.dcmread(file_path, force=True).pixel_array.astype(np.uint8)

    metadata = get_metadata(file_path)
    doc = collection.find_one({"image_id": metadata["Image_ID"], "patient_id": metadata["Patient_ID"]})
    return generate_final_hash(extract_features(img), metadata) == doc["hash_key"] if doc else False

if __name__ == "__main__":
    y_true, y_pred = [], []

    for root, dirs, files in os.walk("./DATASET/DATASET"):
        for fname in files:
            if fname.lower().endswith(('.dcm','.png','.jpg','.jpeg')):
                y_true.append(0)
                y_pred.append(0 if verify_image(os.path.join(root, fname)) else 1)

    for root, dirs, files in os.walk("./tampered_dataset"):
        for fname in files:
            if fname.lower().endswith(('.dcm','.png','.jpg','.jpeg')):
                y_true.append(1)
                y_pred.append(0 if verify_image(os.path.join(root, fname)) else 1)

    if y_true and y_pred:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        accuracy, precision, recall, f1 = accuracy_score(y_true, y_pred), precision_score(y_true, y_pred), recall_score(y_true, y_pred), f1_score(y_true, y_pred)
        fpr, fnr, specificity = fp / (fp + tn + 1e-10), fn / (fn + tp + 1e-10), tn / (tn + fp + 1e-10)

        plt.figure(figsize=(10,6))
        bars = plt.bar(['Accuracy', 'Precision', 'Recall', 'F1', 'FPR', 'FNR', 'Specificity'], [accuracy, precision, recall, f1, fpr, fnr, specificity], color=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#F44336', '#E91E63', '#00BCD4'])
        plt.ylim(0,1)
        for bar in bars: plt.text(bar.get_x()+0.05, bar.get_height() + 0.02, f"{bar.get_height():.2f}")
        plt.title("Tampering Detection Metrics - Batch Verification")
        plt.show()
