# 🧨 EVALUATION + TAMPERING DATASET GENERATOR
import os, cv2, shutil, random
import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import gaussian_filter, map_coordinates, rotate
from pydicom.dataset import FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian
from hashlib import sha256

# ========== PATH SETUP ==========
orig_dir = "./DATASET/DATASET"
output_dir = "./tampered_final_dataset"
os.makedirs(output_dir, exist_ok=True)

# ========== LOAD DICOM ==========
def load_dicom(path):
    dcm = pydicom.dcmread(path)
    img = dcm.pixel_array.astype(np.float32)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img.astype(np.uint8), dcm

# ========== SAVE DICOM ==========
def save_dicom(meta, img, path):
    meta.Rows, meta.Columns = img.shape
    meta.PixelData = img.astype(np.uint8).tobytes()
    meta.BitsAllocated = 8
    meta.BitsStored = 8
    meta.HighBit = 7
    meta.PixelRepresentation = 0
    if not hasattr(meta, "file_meta") or not meta.file_meta:
        meta.file_meta = FileMetaDataset()
    meta.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.is_implicit_VR = False
    meta.is_little_endian = True
    pydicom.dcmwrite(path, meta, write_like_original=False)

# ========== METRICS ==========
def ncc(im1, im2): return np.corrcoef(im1.flatten(), im2.flatten())[0,1]
def compute_ber(h1, h2): return sum(c1 != c2 for c1, c2 in zip(h1, h2)) / len(h1)
def hamming_distance(h1, h2): return sum(c1 != c2 for c1, c2 in zip(h1, h2))
def hash_entropy(binary_string):
    p0 = binary_string.count('0') / len(binary_string)
    p1 = binary_string.count('1') / len(binary_string)
    return -(p0*np.log2(p0+1e-12) + p1*np.log2(p1+1e-12))
def avalanche_effect(h1, h2): return hamming_distance(h1, h2) / len(h1)
def dummy_hash(img): return sha256(img.tobytes()).hexdigest()
def hex_to_bin(hex_str): return bin(int(hex_str, 16))[2:].zfill(len(hex_str)*4)

# ========== ATTACK FUNCTIONS ==========
def gaussian_noise(img): return np.clip(img + np.random.normal(0, 1.2, img.shape).astype(np.float32), 0, 255).astype(np.uint8)
def salt_pepper(img):
    mask = np.random.choice([0, 255], size=img.shape, p=[0.999, 0.001])
    return cv2.addWeighted(img.copy(), 0.995, np.where(mask == 255, mask, img.copy()).astype(np.uint8), 0.005, 0)
def blur(img): return cv2.GaussianBlur(img, (3,3), 0.3)
def gamma(img): return cv2.LUT(img, np.array([(i/255.0)**1.01 * 255 for i in range(256)]).astype("uint8"))
def elastic(img):
    dx, dy = gaussian_filter((np.random.rand(*img.shape)*2 - 1), sigma=1.5)*0.2, gaussian_filter((np.random.rand(*img.shape)*2 - 1), sigma=1.5)*0.2
    x, y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
    return map_coordinates(img, (np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))), order=1, mode='reflect').reshape(img.shape).astype(np.uint8)
def crop(img): return cv2.resize(img[80:-80, 80:-80], (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
def copy_move(img):
    out = img.copy()
    out[img.shape[0]//2:img.shape[0]//2+10, img.shape[1]//2:img.shape[1]//2+10] = cv2.GaussianBlur(img[img.shape[0]//4:img.shape[0]//4+10, img.shape[1]//4:img.shape[1]//4+10].copy(), (3,3), 0.5)
    return out
def rotation_standard(img): return rotate(img, random.choice([90, 180]), reshape=False, mode='reflect').astype(np.uint8)
def rotation_random(img): return rotate(img, random.choice([12, 45, 81]), reshape=False, mode='reflect').astype(np.uint8)

imperceptible_attacks = {"gaussian_noise": gaussian_noise, "salt_pepper": salt_pepper, "blur": blur, "gamma": gamma, "elastic": elastic}
destructive_attacks = {"crop": crop, "copy_move": copy_move, "rotation_standard": rotation_standard, "rotation_random": rotation_random}

# ========== MAIN LOOP ==========
if __name__ == "__main__":
    files = sorted([f for f in os.listdir(orig_dir) if f.endswith(".dcm")])[:30] if os.path.exists(orig_dir) else []
    all_results = []

    def process_attack(atk_name, atk_func, destructive=False):
        print(f"\n🔥 Applying: {atk_name}")
        atk_dir = os.path.join(output_dir, atk_name)
        os.makedirs(atk_dir, exist_ok=True)
        for f in tqdm(files):
            try:
                orig_img, meta = load_dicom(os.path.join(orig_dir, f))
                tampered_resized = cv2.resize(atk_func(orig_img.copy()), (512, 512))
                save_path = os.path.join(atk_dir, f.replace(".dcm", f"_{atk_name}.dcm"))
                save_dicom(meta, tampered_resized, save_path)
                
                resized_orig = cv2.resize(orig_img, (512, 512))
                h1, h2 = hex_to_bin(dummy_hash(orig_img)), hex_to_bin(dummy_hash(tampered_resized))
                all_results.append({"Original": f, "Tampered": os.path.basename(save_path), "Attack_Type": "Destructive" if destructive else "Imperceptible", "Attack": atk_name, "PSNR": cv2.PSNR(resized_orig, tampered_resized), "SSIM": ssim(resized_orig, tampered_resized), "NCC": ncc(resized_orig, tampered_resized), "BER": compute_ber(h1, h2), "Hamming": hamming_distance(h1, h2), "Entropy": hash_entropy(h2), "Avalanche": avalanche_effect(h1, h2)})
            except:
                pass

    for atk_name, atk_func in imperceptible_attacks.items(): process_attack(atk_name, atk_func, False)
    for atk_name, atk_func in destructive_attacks.items(): process_attack(atk_name, atk_func, True)

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv("./final_tampering_metrics.csv", index=False)
        shutil.make_archive(output_dir, 'zip', output_dir)
