# 🔍 MEDICAL IMAGE VERIFICATION SYSTEM
# Verification pipeline for images registered using the batch processor

import cv2
import numpy as np
import pywt
import json
import hmac
import hashlib
from hashlib import sha3_512
from scipy.fftpack import dct
from datetime import datetime
import platform
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu, sobel
from skimage.morphology import remove_small_objects, remove_small_holes, binary_closing, binary_opening, disk
from skimage.util import view_as_blocks
from skimage.measure import shannon_entropy
import sys
import os
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import pymongo
from pymongo import MongoClient
from scipy.stats import pearsonr

print("✅ Basic libraries imported successfully!")

# 🌐 ENVIRONMENT CHECK
try:
    from google.colab import files
    ENVIRONMENT = "COLAB"
    print("🌐 Running in Google Colab environment")
except ImportError:
    ENVIRONMENT = "LOCAL"
    print("🖥 Running in local environment")
    def upload_files():
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        file_paths = filedialog.askopenfilenames(
            title="Select Medical Image for Verification",
            filetypes=[("All Images", ".dcm;.png;.jpg;.jpeg")]
        )
        if file_paths:
            return {os.path.basename(file_paths[0]): open(file_paths[0], 'rb').read()}
        return {}

def validate_dicom_file(dicom_path):
    corruption_flags = []
    try:
        dicom = pydicom.dcmread(dicom_path, force=True)
        if not hasattr(dicom, 'PixelData'):
            corruption_flags.append("NO_PIXEL_DATA")
            return False, "No pixel data found", corruption_flags
        if hasattr(dicom, 'Rows') and hasattr(dicom, 'Columns'):
            expected_size = dicom.Rows * dicom.Columns * (getattr(dicom, 'BitsAllocated', 16) // 8)
            actual_size = len(dicom.PixelData)
            if actual_size < expected_size * 0.5:
                corruption_flags.append("SEVERE_DATA_LOSS")
            elif actual_size < expected_size * 0.8:
                corruption_flags.append("MODERATE_DATA_LOSS")
            if actual_size < expected_size:
                return False, f"Insufficient pixel data: {actual_size}/{expected_size} bytes", corruption_flags
        return True, "Valid DICOM file", corruption_flags
    except Exception as e:
        corruption_flags.append("DICOM_PARSE_ERROR")
        return False, str(e), corruption_flags

def process_dicom_for_diagnosis(dicom_path, output_prefix="verify"):
    corruption_detected = False
    corruption_type = "UNKNOWN"
    try:
        dicom = pydicom.dcmread(dicom_path, force=True)
        try:
            image = dicom.pixel_array.astype(np.float32)
        except Exception as pixel_error:
            corruption_detected = True
            if hasattr(dicom, 'PixelData') and dicom.PixelData:
                try:
                    bits_allocated = getattr(dicom, 'BitsAllocated', 16)
                    dtype = np.uint8 if bits_allocated <= 8 else np.uint16
                    raw_data = np.frombuffer(dicom.PixelData, dtype=dtype)
                    rows, cols = getattr(dicom, 'Rows', 512), getattr(dicom, 'Columns', 512)
                    expected_size = rows * cols
                    if len(raw_data) < expected_size:
                        corruption_type = "RANDOM_ERASURE"
                        padded_data = np.zeros(expected_size, dtype=dtype)
                        padded_data[:len(raw_data)] = raw_data
                        image = padded_data.reshape(rows, cols).astype(np.float32)
                    else:
                        image = raw_data[:expected_size].reshape(rows, cols).astype(np.float32)
                except Exception as raw_error:
                    corruption_type = "TOTAL_CORRUPTION"
                    image = np.random.randint(0, 256, (512, 512), dtype=np.uint8).astype(np.float32)
            else:
                raise Exception("No pixel data available")
                
        try:
            p1, p99 = np.percentile(image, 1), np.percentile(image, 99)
            if p99 - p1 > 0:
                image = np.clip(image, p1, p99)
                image = ((image - p1) / (p99 - p1) * 255.0).astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)
        except:
            image = np.clip(image, 0, 255).astype(np.uint8)

        image = cv2.resize(image, (512, 512))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        edges = cv2.convertScaleAbs(cv2.Laplacian(enhanced, cv2.CV_8U))
        blended = cv2.addWeighted(enhanced, 0.9, cv2.GaussianBlur(edges, (3, 3), 0), 0.1, 0)

        output_path = f"{output_prefix}_diagnostic_enhanced.png"
        cv2.imwrite(output_path, blended)
        return blended, output_path, True, corruption_detected, corruption_type
    except Exception as e:
        synthetic_image = np.zeros((512, 512), dtype=np.uint8)
        return synthetic_image, f"{output_prefix}_synthetic_corrupted.png", True, True, "TOTAL_FAILURE"

def process_regular_image(image_path, output_prefix="verify"):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None: return None, None, False, False, "NONE"
        image = cv2.resize(image, (512, 512))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        output_path = f"{output_prefix}_enhanced.png"
        cv2.imwrite(output_path, enhanced)
        return enhanced, output_path, True, False, "NONE"
    except Exception as e:
        return None, None, False, False, "PROCESSING_ERROR"

def hospital_grade_roi_roni(image_path, output_prefix="verify"):
    try:
        img = cv2.resize(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), (512, 512))
        img_clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(img)
        otsu_thresh = threshold_otsu(img_clahe)
        edges = sobel(img_clahe)
        hybrid_mask = (img_clahe < otsu_thresh) | (edges > np.percentile(edges, 80))
        
        cleaned = remove_small_objects(hybrid_mask, min_size=1000)
        cleaned = binary_opening(binary_closing(remove_small_holes(cleaned, area_threshold=1500), footprint=disk(5)), footprint=disk(3))
        
        roi_mask = (cleaned * 255).astype(np.uint8)
        roni_mask = cv2.bitwise_not(roi_mask)
        roni_image = cv2.bitwise_and(img, img, mask=roni_mask)
        return img, roi_mask, roni_mask, roni_image, True
    except:
        return None, None, None, None, False

def extract_robust_hvs_features(image_path, block_size=4, top_k=25):
    try:
        image = cv2.resize(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), (512, 512))
        coeffs2 = pywt.dwt2(image, 'db2')
        LL = coeffs2[0][:coeffs2[0].shape[0] - coeffs2[0].shape[0]%block_size, :coeffs2[0].shape[1] - coeffs2[0].shape[1]%block_size]
        blocks = view_as_blocks(LL, block_shape=(block_size, block_size))
        edge_blocks = view_as_blocks(sobel(LL), block_shape=(block_size, block_size))
        
        scores, feature_vector = [], []
        for i in range(blocks.shape[0]):
            for j in range(blocks.shape[1]):
                block_entropy, edge_mean = shannon_entropy(blocks[i, j]), np.mean(edge_blocks[i, j])
                scores.append(((i, j), (0.0 if np.isnan(block_entropy) else block_entropy) + 0.5 * (0.0 if np.isnan(edge_mean) else edge_mean)))
                
        scores.sort(key=lambda x: x[1], reverse=True)
        for i, j in [coord for coord, _ in scores[:top_k]]:
            dct_block = dct(dct(blocks[i, j].T, norm='ortho').T, norm='ortho')
            try:
                feature_vector.extend(np.nan_to_num(np.linalg.svd(dct_block)[1][:2], nan=0.0))
            except:
                feature_vector.extend([np.mean(dct_block), np.std(dct_block)])
        return np.array(feature_vector), True
    except:
        return None, False

def get_metadata_from_dicom_or_image(filename, corruption_detected=False, corruption_type="NONE"):
    try:
        if filename.lower().endswith(".dcm"):
            dicom = pydicom.dcmread(filename, force=True)
            patient_id = getattr(dicom, "PatientID", "UNKNOWN")
            image_id = getattr(dicom, "SOPInstanceUID", os.path.splitext(filename)[0])
            modality = getattr(dicom, "Modality", "UNKNOWN")
            try:
                scan_date, scan_time = dicom.StudyDate, dicom.StudyTime[:6]
                timestamp = f"{scan_date[:4]}-{scan_date[4:6]}-{scan_date[6:]}T{scan_time[:2]}:{scan_time[2:4]}:{scan_time[4:]}"
            except:
                timestamp = datetime.utcnow().isoformat()
        else:
            patient_id, image_id, modality, timestamp = f"IMG_PATIENT_{datetime.now().strftime('%Y%m%d_%H%M%S')}", os.path.splitext(filename)[0], "X-ray", datetime.utcnow().isoformat()
            
        return {"Patient_ID": patient_id, "Image_ID": image_id, "Scan_Timestamp": timestamp, "Doctor_ID": "DR_AK_MH001", "Modality": modality, "ROI_type": "lungs_full", "ROI_coords": [100, 100, 400, 450], "Image_Resolution": [512, 512]}, True
    except:
        return None, False

def features_to_binary(features, precision=10): return ''.join(format(int(val * (2 ** precision)), f'0{precision+4}b') for val in features)
def flatten_metadata(metadata): return '|'.join(f"{k}:{v}" for k, v in sorted(metadata.items()))
def generate_hmac(data_string, secret="PES_MEDIC_CHAIN_2025"): return hmac.new(secret.encode(), data_string.encode(), hashlib.sha256).hexdigest()
def hex_to_binary_string(hex_str): return bin(int(hex_str, 16))[2:].zfill(len(hex_str)*4)
def build_rule_dict(rule_number): return {k: format(rule_number, '08b')[i] for i, k in enumerate(['111','110','101','100','011','010','001','000'])}
def evolve_ca(seed, rule_dict, iterations=30, boundary='circular'):
    current = seed
    for _ in range(iterations):
        current = ''.join(rule_dict[(current[i-1] if i > 0 else (current[-1] if boundary == 'circular' else '0')) + current[i] + (current[(i+1)%len(current)] if boundary == 'circular' else (current[i+1] if i+1<len(current) else '0'))] for i in range(len(current)))
    return current
def select_ca_rule(hmac_hash, allowed_rules=[30, 45, 105, 129, 165, 195, 225]): return allowed_rules[int(hmac_hash[:8], 16) % len(allowed_rules)]
def generate_final_hash(features, metadata):
    try:
        hmac_hash = generate_hmac(features_to_binary(features) + flatten_metadata(metadata))
        rule = select_ca_rule(hmac_hash)
        return sha3_512(evolve_ca(hex_to_binary_string(hmac_hash), build_rule_dict(rule)).encode()).hexdigest(), rule, True
    except:
        return None, None, False

if __name__ == "__main__":
    if ENVIRONMENT == "COLAB": uploaded = files.upload()
    else: uploaded = upload_files()
    
    if not uploaded: sys.exit()
    filename = list(uploaded.keys())[0]

    if filename.lower().endswith(".dcm"):
        is_valid, validation_msg, corruption_flags = validate_dicom_file(filename)
        processed_image, enhanced_path, success, corrupted, corruption_type = process_dicom_for_diagnosis(filename)
    else:
        processed_image, enhanced_path, success, corrupted, corruption_type = process_regular_image(filename)

    if not success: sys.exit()
    
    original, roi_mask, roni_mask, roni_image, roi_success = hospital_grade_roi_roni(enhanced_path)
    if not roi_success: sys.exit()

    features, feat_success = extract_robust_hvs_features("verify_roni_image.png")
    if not feat_success: sys.exit()

    metadata, meta_success = get_metadata_from_dicom_or_image(filename, corrupted, corruption_type)
    if not meta_success: sys.exit()

    MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://<username>:<password>@cluster0.ypyd3es.mongodb.net/?retryWrites=true&w=majority")
    
    try:
        client = MongoClient(MONGO_URI)
        collection = client["medical_image_db"]["image_hashes"]
        stored_doc = collection.find_one({"image_id": metadata["Image_ID"], "patient_id": metadata["Patient_ID"]}) or collection.find_one({"image_id": metadata["Image_ID"]})
        expected_hash = stored_doc.get("hash_key") if stored_doc else None
    except:
        expected_hash, stored_doc = None, None

    if not expected_hash and not corrupted: sys.exit()

    recomputed_hash, rule_used, hash_success = generate_final_hash(features, metadata)
    if not hash_success: sys.exit()

    verification_status = "VERIFIED" if recomputed_hash == expected_hash else "TAMPERED"
    print(f"RESULT: {verification_status}")
