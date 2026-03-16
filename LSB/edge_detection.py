import os
import cv2
import numpy as np
import argparse
import tkinter as tk
from tkinter import filedialog


def get_edge_coords(img_path, low_threshold=50, high_threshold=150):
    """
    Deteksi tepi menggunakan Canny Edge Detection pada 3 channel RGB secara terpisah.
    """
    # Validasi file citra ada atau tidak
    if not os.path.exists(img_path):
        raise SystemExit(f"File tidak ditemukan: {img_path}")

    # Baca citra dalam format RGB (OpenCV default)
    img_rgb = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_rgb is None:
        raise SystemExit(f"Gagal membuka file sebagai citra: {img_path}")

    # Terapkan Gaussian Blur untuk mengurangi noise sebelum deteksi tepi
    blur_color = cv2.GaussianBlur(img_rgb, (5, 5), 0)

    # Deteksi tepi pada setiap channel secara terpisah menggunakan Canny
    edges_r = cv2.Canny(blur_color[:, :, 2], low_threshold, high_threshold)  # Channel Red
    edges_g = cv2.Canny(blur_color[:, :, 1], low_threshold, high_threshold)  # Channel Green
    edges_b = cv2.Canny(blur_color[:, :, 0], low_threshold, high_threshold)  # Channel Blue
    
    # Gabungkan hasil deteksi dari 3 channel menggunakan OR bitwise
    canny_edges = cv2.bitwise_or(edges_r, cv2.bitwise_or(edges_g, edges_b))
    # Konversi ke mask boolean (True jika tepi, False jika bukan)
    final_edges = canny_edges > 0

    # Koordinat piksel yang terdeteksi sebagai tepi dalam format [y, x]
    coords_yx = np.argwhere(final_edges)
    # Konversi ke format [x, y] (lebih intuitif untuk koordinat CartesiAN)
    coords_xy = np.flip(coords_yx, axis=1)

    return coords_xy, final_edges

def get_non_edge_coords(img_path, edge_mask):
    """
    Dapatkan koordinat NON-TEPI (semua piksel MINUS piksel tepi).
    """
    if not os.path.exists(img_path):
        raise SystemExit(f"File tidak ditemukan: {img_path}")

    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise SystemExit(f"Gagal membuka file sebagai citra: {img_path}")

    # Inverse dari edge mask
    non_edges = ~edge_mask  # True = non-tepi, False = tepi

    # Koordinat piksel non-tepi dalam format [y, x]
    coords_non_edge_yx = np.argwhere(non_edges)
    # Konversi ke format [x, y]
    coords_non_edge_xy = np.flip(coords_non_edge_yx, axis=1)

    return coords_non_edge_xy

def _pick_image_dialog():
    """Buka folder untuk memilih gambar."""
    root = tk.Tk()
    root.withdraw()  # Sembunyikan window root
    filetypes = [("Image files", ".png;"), ("All files", ".*")]
    # Tampilkan dialog folder dan tunggu user memilih file
    img_path = filedialog.askopenfilename(title="Pilih file citra", filetypes=filetypes)
    root.destroy()  # Tutup window
    return img_path

def _save_edge_coords(img_path, coords_xy):
    """Simpan koordinat tepi ke file TXT dalam folder 'koordinat'"""
    # Ekstrak nama file tanpa extension
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    # Tentukan direktori project (tempat edge_detection.py berada)
    project_root = os.path.dirname(os.path.abspath(__file__))
    # Buat path folder output 'koordinat'
    out_dir = os.path.join(project_root, "koordinat")
    # Buat folder jika belum ada
    os.makedirs(out_dir, exist_ok=True)

    # Tentukan path file output TXT
    out_txt = os.path.join(out_dir, f"{base_name}_edge_coords.txt")

    # Simpan ke TXT dengan format "x,y" per baris (manual write)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("x,y\n")  # Header
        for x, y in coords_xy:
            f.write(f"{x},{y}\n")

    print(f"[OK] Koordinat tepi disimpan: {out_txt}")

def _save_non_edge_coords(img_path, coords_xy):
    """Simpan koordinat non-tepi ke file TXT dalam folder 'koordinat'"""
    # Ekstrak nama file tanpa extension
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    # Tentukan direktori project
    project_root = os.path.dirname(os.path.abspath(__file__))
    # Buat path folder output 'koordinat'
    out_dir = os.path.join(project_root, "koordinat")
    # Buat folder jika belum ada
    os.makedirs(out_dir, exist_ok=True)

    # Tentukan path file output TXT
    out_txt = os.path.join(out_dir, f"{base_name}_non_edge_coords.txt")

    # Simpan ke TXT dengan format "x,y" per baris
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("x,y\n")  # Header
        for x, y in coords_xy:
            f.write(f"{x},{y}\n")

    print(f"[OK] Koordinat non-tepi disimpan: {out_txt}")


def main():
    """parsing argument, memilih citra, deteksi tepi, dan simpan hasil."""
    # Setup argument parser untuk CLI
    parser = argparse.ArgumentParser(description="Deteksi tepi (Canny) pada citra input")
    parser.add_argument("-i", "--image", help="Path ke file citra (jpg/png/...)")
    args = parser.parse_args()

    # Jika argument -i diberikan, gunakan itu; jika tidak, buka dialog picker
    img_path = args.image or _pick_image_dialog()
    if not img_path:
        raise SystemExit("Tidak ada file dipilih. Gunakan -i untuk memberikan path.")

    # Jalankan deteksi tepi dan dapatkan koordinat
    coords_edge_xy, edge_mask = get_edge_coords(img_path)
    coords_non_edge_xy = get_non_edge_coords(img_path, edge_mask)
    
    # Tampilkan statistik hasil deteksi
    print(f"\n=== STATISTIK DETEKSI TEPI ===")
    print(f"Total pixel tepi: {len(coords_edge_xy)}")
    print(f"Total pixel non-tepi: {len(coords_non_edge_xy)}")
    print(f"Total pixel citra: {len(coords_edge_xy) + len(coords_non_edge_xy)}")
    
    print(f"\nContoh koordinat tepi (x,y) - 20 pertama:")
    for i, (x, y) in enumerate(coords_edge_xy[:20], start=1):
        print(f"{i:02d}. ({x}, {y})")

    print(f"\nContoh koordinat non-tepi (x,y) - 20 pertama:")
    for i, (x, y) in enumerate(coords_non_edge_xy[:20], start=1):
        print(f"{i:02d}. ({x}, {y})")

    # Simpan koordinat tepi ke file CSV dan TXT
    _save_edge_coords(img_path, coords_edge_xy)
    
    # Simpan koordinat non-tepi ke file CSV dan TXT
    _save_non_edge_coords(img_path, coords_non_edge_xy)

# Entry point: jalankan main() jika script dijalankan langsung (bukan di-import)
if __name__ == "__main__":
    main()