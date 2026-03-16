import sys
import struct
from tkinter import filedialog
import numpy
import matplotlib.pyplot as plt
from PIL import Image
import tkinter as tk
from tkinter import filedialog
from crypt import AESCipher, generate_encryption_key
import time
from evaluation import print_evaluation_results

# Decompose a binary file into an array of bits
def decompose(data):
	v = []
	
	# Pack file len in 4 bytes
	fSize = len(data)
	data_bytes = list(struct.pack("i", fSize))
	
	if isinstance(data, str):
		data_bytes += [ord(b) for b in data]
	else:
		data_bytes += list(data)

	for b in data_bytes:
		for i in range(7, -1, -1):
			v.append((b >> i) & 0x1)

	return v

# Assemble an array of bits into a binary file
def assemble(v):    
	bytes_out = b""

	length = len(v)
	for idx in range(0, len(v)//8):
		byte = 0
		for i in range(0, 8):
			if (idx*8+i < length):
				byte = (byte<<1) + v[idx*8+i]                
		bytes_out += bytes([byte])

	payload_size = struct.unpack("i", bytes_out[:4])[0]

	return bytes_out[4: payload_size + 4]

# Set the i-th bit of v to x
def set_bit(n, i, x):
	mask = 1 << i
	n &= ~mask
	if x:
		n |= mask
	return n

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
        filetypes=[("All files", "*.*")]
    )
    root.destroy()
    return path

# Embed payload file into LSB bits of an image
def embed(imgFile, payload):
	start_time = time.time()

	# Process source image
	img = Image.open(imgFile)
	(width, height) = img.size
	conv = img.convert("RGB").getdata()
	print("[*] Input image size: %dx%d pixels." % (width, height))

	# Calculate max payload size
	max_size_kb = width*height*3.0/8/1024		# max payload size
	max_size_bits = width*height*3
	print("[*] Usable payload size: %.2f KB (%d bits)" % (max_size_kb, max_size_bits))

	f = open(payload, "rb")
	data = f.read()
	f.close()
	payload_kb = len(data)/1024.0
	payload_bits = len(data) * 8
	print("[+] Payload size: %.3f KB (%d bits)" % (payload_kb, payload_bits))

	# Generate Key
	password = generate_encryption_key()
	print("[+] Generated encryption key: %s" % password)
	# Simpan key ke file
	key_filename = imgFile + "-key.txt"
	with open(key_filename, "w") as key_file:
		key_file.write(password)
		print("[+] Key saved to: %s" % key_filename)
		
	# Encypt
	cipher = AESCipher(password)
	data_enc = cipher.encrypt(data)

	# Process data from payload file
	v = decompose(data_enc)
	
	# Add until multiple of 3
	while(len(v)%3):
		v.append(0)

	payload_size = len(v)/8/1024.0
	payload_size_bits = len(v)
	print("[+] Encrypted payload size: %.3f KB (%d bits)" % (payload_size, payload_size_bits))
	if (payload_size > max_size_kb - 4):
		print("[-] Cannot embed. File too large")
		sys.exit()
		
	# Create output image
	steg_img = Image.new('RGB',(width, height))
	data_img = steg_img.getdata()

	idx = 0

	for h in range(height):
		for w in range(width):
			(r, g, b) = conv.getpixel((w, h))
			if idx < len(v):
				r = set_bit(r, 0, v[idx])
				g = set_bit(g, 0, v[idx+1])
				b = set_bit(b, 0, v[idx+2])
			data_img.putpixel((w,h), (r, g, b))
			idx = idx + 3
    
	steg_img.save(imgFile + "-stego.png", "PNG")
	
	end_time = time.time()
	embed_time = end_time - start_time
	print("[+] %s embedded successfully!" % payload)
	print(f"[+] Embedding time: {embed_time:.4f} seconds")

	return imgFile + "-stego.png", password, embed_time

# Extract data embedded into LSB of the input file
def extract(in_file, out_file, password):
	start_time = time.time()

	# Process source image
	img = Image.open(in_file)
	(width, height) = img.size
	conv = img.convert("RGB").getdata()
	print("[+] Image size: %dx%d pixels." % (width, height))

	# Extract LSBs
	v = []
	for h in range(height):
		for w in range(width):
			(r, g, b) = conv.getpixel((w, h))
			v.append(r & 1)
			v.append(g & 1)
			v.append(b & 1)
			
	data_out = assemble(v)

	# Decrypt
	cipher = AESCipher(password)
	data_dec = cipher.decrypt(data_out)

	# Write decrypted data
	out_f = open(out_file, "wb")
	out_f.write(data_dec)
	out_f.close()
	
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
