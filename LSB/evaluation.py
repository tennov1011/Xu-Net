import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import time

def calculate_mse(img1_path, img2_path):
    """Calculate Mean Squared Error between two images"""
    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')
    
    arr1 = np.array(img1, dtype=np.float64)
    arr2 = np.array(img2, dtype=np.float64)
    
    mse = np.mean((arr1 - arr2) ** 2)
    return mse

def calculate_psnr(img1_path, img2_path):
    """Calculate Peak Signal-to-Noise Ratio"""
    mse = calculate_mse(img1_path, img2_path)
    
    if mse == 0:
        return float('inf')
    
    psnr = 10 * np.log10((255 ** 2) / mse)
    return psnr

def calculate_ssim(img1_path, img2_path):
    """Calculate Structural Similarity Index"""
    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')
    
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    
    # SSIM untuk setiap channel, kemudian rata-rata
    ssim_value = ssim(arr1, arr2, channel_axis=2, data_range=255)
    return ssim_value

#Print all evaluation metrics
def print_evaluation_results(original_img, stego_img, embed_time, extract_time):
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    # Calculate image quality metrics
    mse = calculate_mse(original_img, stego_img)
    psnr = calculate_psnr(original_img, stego_img)
    ssim_value = calculate_ssim(original_img, stego_img)
    
    print("\n[IMPERCEPTIBILITY METRICS]")
    print(f"MSE  (Mean Squared Error)    : {mse:.6f}")
    print(f"PSNR (Peak Signal-to-Noise)  : {psnr:.4f} dB")
    print(f"SSIM (Structural Similarity) : {ssim_value:.6f}")
    
    print("\n[EXECUTION TIME]")
    print(f"Encryption + Embedding Time  : {embed_time:.4f} seconds")
    print(f"Extraction + Decryption Time : {extract_time:.4f} seconds")
    print(f"Total Time                   : {embed_time + extract_time:.4f} seconds")
    
    # print("\n[QUALITY INTERPRETATION]")
    # if psnr > 40:
    #     print("Image Quality: Excellent (Visually Identical)")
    # elif psnr > 30:
    #     print("Image Quality: Good (Minimal Distortion)")
    # elif psnr > 20:
    #     print("Image Quality: Fair (Noticeable Distortion)")
    # else:
    #     print("Image Quality: Poor (Significant Distortion)")
    
    # if ssim_value > 0.95:
    #     print("Structural Similarity: Excellent")
    # elif ssim_value > 0.90:
    #     print("Structural Similarity: Good")
    # elif ssim_value > 0.80:
    #     print("Structural Similarity: Fair")
    # else:
    #     print("Structural Similarity: Poor")
    
    print("="*60 + "\n")