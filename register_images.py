# 🏥 MEDICAL IMAGE BATCH PROCESSOR & DATABASE REGISTRATION
import os
import sys
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
import pydicom
from pymongo import MongoClient

# SECURITY: Use environment variables for database credentials. NEVER hardcode them.
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://<username>:<password>@cluster0.ypyd3es.mongodb.net/?retryWrites=true&w=majority")
DATABASE_NAME = "medical_image_db"
COLLECTION_NAME = "image_hashes1"

# 🧠 DEFINE IMAGE PROCESSING FUNCTIONS
def process_dicom_for_diagnosis(dicom_path, output_prefix=""):
    try:
        dicom = pydicom.dcmread(dicom_path)
        image = dicom.pixel_array.astype(np.float32)
        p1, p99 = np.percentile(image, 1), np.percentile(image, 99)
        image = np.clip(image, p1, p99)
        image = ((image - p1) / (p99 - p1) * 255.0).astype(np.uint8)
        image = cv2.resize(image, (512, 512))
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        
        edges = cv2.Laplacian(enhanced, cv2.CV_8U)
        edges = cv2.convertScaleAbs(edges)
        edges = cv2.GaussianBlur(edges, (3, 3), 0)
        blended = cv2.addWeighted(enhanced, 0.9, edges, 0.1, 0)
        
        output_path = f"{output_prefix}_diagnostic_enhanced.png"
        cv2.imwrite(output_path, blended)
        return blended, output_path, True
    except Exception as e:
        print(f"❌ Error processing DICOM {dicom_path}: {str(e)}")
        return None, None, False

def process_regular_image(image_path, output_prefix=""):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None: return None, None, False
        image = cv2.resize(image, (512, 512))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        output_path = f"{output_prefix}_enhanced.png"
        cv2.imwrite(output_path, enhanced)
        return enhanced, output_path, True
    except Exception as e:
        return None, None, False

def hospital_grade_roi_roni(image_path, output_prefix=""):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (512, 512))
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img)
        
        otsu_thresh = threshold_otsu(img_clahe)
        edges = sobel(img_clahe)
        edge_mask = edges > np.percentile(edges, 80)
        hybrid_mask = (img_clahe < otsu_thresh) | edge_mask
        
        cleaned = remove_small_objects(hybrid_mask, min_size=1000)
        cleaned = remove_small_holes(cleaned, area_threshold=1500)
        cleaned = binary_closing(cleaned, footprint=disk(5))
        cleaned = binary_opening(cleaned, footprint=disk(3))
        
        roi_mask = (cleaned * 255).astype(np.uint8)
        roni_mask = cv2.bitwise_not(roi_mask)
        roni_image = cv2.bitwise_and(img, img, mask=roni_mask)
        
        cv2.imwrite(f"{output_prefix}_roi_mask.png", roi_mask)
        cv2.imwrite(f"{output_prefix}_roni_mask.png", roni_mask)
        cv2.imwrite(f"{output_prefix}_roni_image.png", roni_image)
        return img, roi_mask, roni_mask, roni_image, True
    except Exception as e:
        return None, None, None, None, False

def extract_robust_hvs_features(image_path, block_size=4, top_k=25):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (512, 512))
        coeffs2 = pywt.dwt2(image, 'db2')
        LL, _ = coeffs2[0], coeffs2[1]
        LL = LL[:LL.shape[0] - LL.shape[0]%block_size, :LL.shape[1] - LL.shape[1]%block_size]
        
        blocks = view_as_blocks(LL, block_shape=(block_size, block_size))
        edge_map = sobel(LL)
        edge_blocks = view_as_blocks(edge_map, block_shape=(block_size, block_size))
        
        scores, feature_vector = [], []
        for i in range(blocks.shape[0]):
            for j in range(blocks.shape[1]):
                hvs_score = shannon_entropy(blocks[i, j]) + 0.5 * np.mean(edge_blocks[i, j])
                scores.append(((i, j), hvs_score))
                
        scores.sort(key=lambda x: x[1], reverse=True)
        top_blocks = [coord for coord, _ in scores[:top_k]]
        
        for i, j in top_blocks:
            dct_block = dct(dct(blocks[i, j].T, norm='ortho').T, norm='ortho')
            _, S, _ = np.linalg.svd(dct_block)
            feature_vector.extend(S[:2])
            
        return np.array(feature_vector), True
    except Exception as e:
        return None, False

def get_metadata_from_dicom_or_image(filename):
    try:
        if filename.lower().endswith(".dcm"):
            dicom = pydicom.dcmread(filename)
            patient_id = getattr(dicom, "PatientID", "UNKNOWN")
            image_id = getattr(dicom, "SOPInstanceUID", os.path.splitext(filename)[0])
            modality = getattr(dicom, "Modality", "UNKNOWN")
            timestamp = datetime.utcnow().isoformat()
        else:
            patient_id = f"IMG_PATIENT_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            image_id = os.path.splitext(filename)[0]
            modality = "X-ray"
            timestamp = datetime.utcnow().isoformat()
            
        metadata = {
            "Patient_ID": patient_id, "Image_ID": image_id, "Scan_Timestamp": timestamp,
            "Doctor_ID": "DR_AK_MH001", "Modality": modality, "ROI_type": "lungs_full",
            "ROI_coords": [100, 100, 400, 450], "Image_Resolution": [512, 512],
            "System_ID": platform.node()
        }
        return metadata, True
    except Exception as e:
        return None, False

def features_to_binary(features, precision=10): return ''.join(format(int(val * (2 ** precision)), f'0{precision+4}b') for val in features)
def flatten_metadata(metadata): return '|'.join(f"{k}:{v}" for k, v in sorted(metadata.items()))
def generate_hmac(data_string, secret="PES_MEDIC_CHAIN_2025"): return hmac.new(secret.encode(), data_string.encode(), hashlib.sha256).hexdigest()
def hex_to_binary_string(hex_str): return bin(int(hex_str, 16))[2:].zfill(len(hex_str)*4)
def build_rule_dict(rule_number):
    bin_rule = format(rule_number, '08b')
    return {k: bin_rule[i] for i, k in enumerate(['111','110','101','100','011','010','001','000'])}
def evolve_ca(seed, rule_dict, iterations=30, boundary='circular'):
    current = seed
    for _ in range(iterations):
        next_gen = ''
        for i in range(len(current)):
            left = current[i-1] if i > 0 else (current[-1] if boundary == 'circular' else '0')
            center = current[i]
            right = current[(i+1)%len(current)] if boundary == 'circular' else (current[i+1] if i+1<len(current) else '0')
            next_gen += rule_dict[left + center + right]
        current = next_gen
    return current
def select_ca_rule(hmac_hash, allowed_rules=[30, 45, 105, 129, 165, 195, 225]): return allowed_rules[int(hmac_hash[:8], 16) % len(allowed_rules)]

def generate_final_hash(features, metadata):
    try:
        hmac_hash = generate_hmac(features_to_binary(features) + flatten_metadata(metadata))
        rule = select_ca_rule(hmac_hash)
        ca_out = evolve_ca(hex_to_binary_string(hmac_hash), build_rule_dict(rule))
        return sha3_512(ca_out.encode()).hexdigest(), rule, True
    except Exception as e:
        return None, None, False

if __name__ == "__main__":
    print("Run this script using the Colab/Jupyter environment for file upload handling.")
    # Implementation of upload logic should be passed here based on environment
