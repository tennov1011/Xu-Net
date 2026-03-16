import sys
import struct
import os
import tkinter as tk
import hashlib
import random
import time
from evaluation import print_evaluation_results
from tkinter import filedialog
from PIL import Image
from crypt import AESCipher, generate_encryption_key
from edge_detection import get_edge_coords, get_non_edge_coords, _save_edge_coords, _save_non_edge_coords


# Path dasar proyek dan file payload default
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PAYLOAD = os.path.join(BASE_DIR, "pesan.txt")

# Ubah bytes menjadi list bit (MSB -> LSB) dengan header panjang
def decompose(data: bytes):
    bits = []
    # Pack file len in 4 bytes (little endian)
    fSize = len(data)
    bytes_list = list(struct.pack("i", fSize))
    bytes_list += list(data)

    for b in bytes_list:
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)
    return bits

# Ubah list bit menjadi bytes
def assemble(bits):
    bytes_out = bytearray()
    length = len(bits)
    for idx in range(0, len(bits) // 8):
        byte = 0
        for i in range(0, 8):
            if (idx*8+i < length):
                byte = (byte<<1) + bits[idx*8+i]
        bytes_out.append(byte)

    # Ambil panjang data dari 4 byte pertama
    payload_size = struct.unpack("i", bytes_out[:4])[0]
    return bytes_out[4: payload_size + 4]

# Set 3 bit LSB dari nilai channel (0..255)
def set_last_3_bits(value, bits3):
    new_val = value & 0b11111000
    new_val |= (bits3[0] << 2) | (bits3[1] << 1) | (bits3[2] << 0)
    return new_val

# Ambil 3 bit LSB dari nilai channel
def get_last_3_bits(value):
    return [(value >> 2) & 1, (value >> 1) & 1, (value >> 0) & 1]

def _pick_image_dialog(title="Choose image"):
    root = tk.Tk()
    root.withdraw()
    filetypes = [("Image files", "*.png"), ("All files", "*.*")]
    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return path

def _pick_payload_dialog(title="Choose payload file"):
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(title=title, filetypes=[("All files", "*.*")])
    root.destroy()
    return path

def _pick_output_dialog(title="Save extracted file"):
    root = tk.Tk()
    root.withdraw()
    path = filedialog.asksaveasfilename(
        title=title,
        defaultextension=".txt",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
    )
    root.destroy()
    return path

def _coords_path(img_path, suffix):
    base = os.path.basename(img_path)
    if "-stego.png" in base:
        base = base.split("-stego.png")[0]
    # Hapus ekstensi .png jika ada
    if base.endswith(".png"):
        base = base[:-4]
    out_dir = os.path.join(BASE_DIR, "koordinat")
    return os.path.join(out_dir, f"{base}_{suffix}.txt")

def _load_coords_txt(path):
    coords = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip baris kosong atau header
            if not line or 'x' in line.lower():
                continue
            x, y = map(int, line.split(","))
            coords.append((x, y))
    return coords

def _prng_permutation(n, aes_key):
    seed = int.from_bytes(aes_key[:16], "big")
    rng = random.Random(seed)
    idxs = list(range(n))
    rng.shuffle(idxs)
    return idxs

# Embed payload ke citra berdasarkan daerah tepi
def embed(imgFile, payload=None):
    start_time = time.time()

    if payload is None:
        payload = DEFAULT_PAYLOAD

    # Baca gambar & deteksi tepi
    img = Image.open(imgFile).convert("RGB")
    (width, height) = img.size
    print(f"[*] Input image size: {width}x{height} pixels.")
    coords_edge, edge_mask = get_edge_coords(imgFile)
    coords_non_edge = get_non_edge_coords(imgFile, edge_mask)
    
	# Simpan koordinat tepi ke file TXT
    _save_edge_coords(imgFile, coords_edge)
    _save_non_edge_coords(imgFile, coords_non_edge)

    # Generate AES key
    password = generate_encryption_key()
    print(f"[+] Generated encryption key: {password}")
    
    key_filename = imgFile + "-key.txt"
    with open(key_filename, "w") as key_file:
        key_file.write(password)
    print(f"[+] Key saved to: {key_filename}")

    cipher = AESCipher(password)
    aes_key = cipher.key 

    # Hitung kapasitas
    edge_capacity_bits = len(coords_edge) * 3
    non_edge_capacity_bits = len(coords_non_edge) * 1
    total_capacity_bits = edge_capacity_bits + non_edge_capacity_bits

    print(f"[*] Edge pixels: {len(coords_edge)}")
    print(f"[*] Non-edge pixels: {len(coords_non_edge)}")
    print(f"[*] Total capacity: {total_capacity_bits / 8 / 1024:.2f} KB")

    with open(payload, "rb") as f:
        data = f.read()
    print(f"[+] Payload size: {len(data)/1024.0:.3f} KB")

    # Enkripsi payload
    data_enc = cipher.encrypt(data)

    # Ubah ke bit (sudah include header 4 byte)
    v = decompose(data_enc)

    # Padding agar kelipatan 3 bit
    while len(v) % 3 != 0:
        v.append(0)

    # Cek kapasitas
    payload_bits = len(v)
    print(f"[+] Encrypted payload size: {payload_bits} bits")
    if payload_bits > total_capacity_bits:
        print("[-] Cannot embed. File too large")
        print(f"[-] Payload bits: {payload_bits}")
        print(f"[-] Capacity bits: {total_capacity_bits}")
        sys.exit()

    # Salin gambar untuk output
    steg_img = img.copy()

    #Tepi: Sisipkan 3 bit/piksel, urutan R-G-B-R Cylic
    idx = 0
    for i, (x, y) in enumerate(coords_edge):
        if idx >= len(v):
            break

        r, g, b = steg_img.getpixel((x, y))
        bits3 = [v[idx], v[idx+1], v[idx+2]]
        idx += 3

        if i % 3 == 0:
            r = set_last_3_bits(r, bits3)
        elif i % 3 == 1:
            g = set_last_3_bits(g, bits3)
        else:
            b = set_last_3_bits(b, bits3)

        steg_img.putpixel((x, y), (r, g, b))

    # Non-tepi: Sisipkan 1 bit/piksel, urutan R-G-B-R Cylic
    if idx < len(v):
        perm = _prng_permutation(len(coords_non_edge), aes_key)
        for bit_idx, perm_idx in enumerate(perm):
            if idx >= len(v):
                break
            x, y = coords_non_edge[perm_idx]
            r, g, b = steg_img.getpixel((x, y))
            bit = v[idx]
            idx += 1

            # Channel bergilir berdasarkan urutan bit
            if bit_idx % 3 == 0:
                r = (r & 0xFE) | bit
            elif bit_idx % 3 == 1:
                g = (g & 0xFE) | bit
            else:
                b = (b & 0xFE) | bit

            steg_img.putpixel((x, y), (r, g, b))

    # Simpan stego image
    out_path = imgFile + "-stego.png"
    steg_img.save(out_path, "PNG")
    print(f"[+] Embedded successfully: {out_path}")

    end_time = time.time()
    embed_time = end_time - start_time
    print(f"[+] Embedding time: {embed_time:.4f} seconds")

    return out_path, password, embed_time

# Ekstrak payload dari citra stego
def extract(in_file, out_file, password):
    start_time = time.time()
    img = Image.open(in_file).convert("RGB")

    edge_path = _coords_path(in_file, "edge_coords")
    non_edge_path = _coords_path(in_file, "non_edge_coords")

    if not os.path.exists(edge_path) or not os.path.exists(non_edge_path):
        print("[-] Koordinat tidak ditemukan di folder 'koordinat'.")
        sys.exit()

    coords_edge = _load_coords_txt(edge_path)
    coords_non_edge = _load_coords_txt(non_edge_path)

    cipher = AESCipher(password)
    aes_key = cipher.key

    # Tepi: Ambil 3 LSB sesuai pola R-G-B
    all_bits = []
    for i, (x, y) in enumerate(coords_edge):
        r, g, b = img.getpixel((x, y))
        if i % 3 == 0:
            all_bits.extend(get_last_3_bits(r))
        elif i % 3 == 1:
            all_bits.extend(get_last_3_bits(g))
        else:
            all_bits.extend(get_last_3_bits(b))

    # Baca header untuk tahu panjang payload 
    if len(all_bits) >= 32:
        # Ubah 32 bit pertama jadi 4 byte (header)
        header_bytes = bytearray()
        for i in range(4):
            byte = 0
            for j in range(8):
                byte = (byte << 1) + all_bits[i*8 + j]
            header_bytes.append(byte)
        
        payload_len = struct.unpack("i", header_bytes)[0]
        total_bits_needed = (4 + payload_len) * 8
        
        print(f"[DEBUG] Payload length (from header): {payload_len} bytes")
        print(f"[DEBUG] Total bits needed: {total_bits_needed}")
        
        # Hitung berapa bit lagi yang diperlukan dari non-edge
        bits_from_edge = len(all_bits)
        bits_needed_from_non_edge = total_bits_needed - bits_from_edge
        
        print(f"[DEBUG] Bits from edge: {bits_from_edge}")
        print(f"[DEBUG] Bits needed from non-edge: {bits_needed_from_non_edge}")
        
        # NON-TEPI
        if bits_needed_from_non_edge > 0:
            perm = _prng_permutation(len(coords_non_edge), aes_key)
            
            for bit_idx, perm_idx in enumerate(perm):
                if len(all_bits) >= total_bits_needed:
                    break
                
                x, y = coords_non_edge[perm_idx]
                r, g, b = img.getpixel((x, y))

                # Channel bergilir berdasarkan urutan bit
                if bit_idx % 3 == 0:
                    all_bits.append(r & 1)
                elif bit_idx % 3 == 1:
                    all_bits.append(g & 1)
                else:
                    all_bits.append(b & 1)
            
    # DEBUG: Tampilkan jumlah bit yang diekstrak
    print(f"[DEBUG] Total bit diekstrak: {len(all_bits)}")
    print(f"[DEBUG] Total byte: {len(all_bits) // 8}")

    # Assemble otomatis baca header 4 byte
    data_enc = assemble(all_bits)

    # Dekripsi dan simpan
    print(f"[DEBUG] Data terenkripsi (byte): {len(data_enc)}")
    data_dec = cipher.decrypt(data_enc)

    with open(out_file, "wb") as f:
        f.write(data_dec)

    print(f"[+] Extracted and decrypted to: {out_file}")

    end_time = time.time()
    extract_time = end_time - start_time
    print(f"[+] Extraction time: {extract_time:.4f} seconds")

    return extract_time

# Main interaktif 
def main():
    mode = input("Select mode (e=embed, x=extract, t=test_evaluation): ").strip().lower()

    if mode == "e":
        img_path = _pick_image_dialog("Choose image to embed")
        if not img_path:
            print("Failed to select image.")
            return
        payload_path = _pick_payload_dialog("Choose payload file")
        if not payload_path:
            print("Failed to select payload file.")
            return
        stego_path, password, embed_time = embed(img_path, payload_path)

    elif mode == "x":
        img_path = _pick_image_dialog("Choose stego image to extract")
        if not img_path:
            print("Failed to select stego image.")
            return
        out_file = _pick_output_dialog("Save extracted file")
        if not out_file:
            print("Failed to select output file.")
            return
        password = input("Key/Password: ").strip()  
        if not password:
            print("Key/Password is required.")
            return
        extract_time = extract(img_path, out_file, password)

    elif mode == "t":
        # Full test with evaluation
        print("\n[FULL EVALUATION TEST]")
        img_path = _pick_image_dialog("Choose original image")
        if not img_path:
            print("Failed to select image.")
            return
        payload_path = _pick_payload_dialog("Choose payload file")
        if not payload_path:
            print("Failed to select payload file.")
            return
        
        # Embed
        stego_path, password, embed_time = embed(img_path, payload_path)
        
        # Extract
        out_file = img_path + "-extracted.txt"
        extract_time = extract(stego_path, out_file, password)
        
        # Print evaluation
        print_evaluation_results(img_path, stego_path, embed_time, extract_time)

    else:
        print("Unknown mode.")

if __name__ == "__main__":
    main()