import os
import cv2
import numpy as np
import argparse
from pathlib import Path

def generate_random_payload(num_pixels, bpp=0.3):
    # Hitung jumlah bit yang dibutuhkan untuk RGB
    # Total channel = num_pixels * 3 (karena RGB punya 3 channel)
    # Kapasitas = total_channel * bpp
    num_bits = int(num_pixels * 3 * bpp)
    
    # Generate bit acak (0 atau 1)
    payload_bits = np.random.randint(0, 2, size=num_bits, dtype=np.uint8)
    
    return payload_bits

def lsb_embed(cover_image, payload_bits):
    # Copy gambar agar tidak mengubah aslinya
    stego_image = cover_image.copy()
    
    # Flatten gambar RGB menjadi 1D array untuk memudahkan proses
    flat_stego = stego_image.flatten()

    # Sisipkan payload secara berurutan dari channel pertama hingga jumlah bit terpenuhi
    num_bits = min(len(payload_bits), len(flat_stego))
    for i in range(num_bits):
        # Hapus LSB (bit ke-0) dari channel, lalu sisipkan bit payload
        flat_stego[i] = (flat_stego[i] & 0xFE) | payload_bits[i]
    
    # Kembalikan ke bentuk 3D (reshape ke height x width x 3)
    stego_image = flat_stego.reshape(cover_image.shape)
    
    return stego_image

def process_resolution_folder(resolution_name, base_path='.', force_overwrite=False):
    print(f"\n{'='*60}")
    print(f"Memproses folder: {resolution_name.upper()}")
    print(f"{'='*60}")
    
    # Path untuk folder cover dan stego
    cover_base = Path(base_path) / resolution_name / 'cover'
    stego_base = Path(base_path) / resolution_name / 'stego'
    
    # List splits yang akan diproses
    splits = ['train', 'test', 'validation']
    
    # Proses setiap split (train, test, validation)
    for split in splits:
        print(f"\n Split: {split}")
        
        # Path input dan output
        cover_folder = cover_base / split
        stego_folder = stego_base / split
        
        # Cek apakah folder cover exists
        if not cover_folder.exists():
            print(f"Folder tidak ditemukan: {cover_folder}")
            continue
        
        # Buat folder output jika belum ada
        stego_folder.mkdir(parents=True, exist_ok=True)
        
        # Dapatkan semua file .png di folder cover
        image_files = sorted(cover_folder.glob('*.png'))
        
        if len(image_files) == 0:
            print(f"Tidak ada file .png di folder {cover_folder}")
            continue
        
        print(f"  Ditemukan {len(image_files)} gambar")
        
        # Counter untuk tracking progress
        success_count = 0
        skipped_count = 0
        error_count = 0
        
        # Proses setiap gambar
        for idx, image_path in enumerate(image_files, start=1):
            # Cek apakah file stego sudah ada (RESUME FEATURE)
            output_name = image_path.name
            output_path = stego_folder / output_name
            
            if output_path.exists() and not force_overwrite:
                skipped_count += 1
                # Print progress skip setiap 100 gambar
                if skipped_count % 100 == 0:
                    print(f"  Skipped {skipped_count} files (already processed)")
                continue
            try:
                # Baca gambar (RGB/Color)
                cover_img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                
                if cover_img is None:
                    print(f" Gagal membaca: {image_path.name}")
                    error_count += 1
                    continue
                
                # Dapatkan dimensi gambar RGB (height x width x 3)
                height, width, channels = cover_img.shape
                num_pixels = height * width
                
                # Generate payload bit acak (0.3 bpp)
                payload_bits = generate_random_payload(num_pixels, bpp=0.3)
                
                # Lakukan LSB embedding
                stego_img = lsb_embed(cover_img, payload_bits)
                
                # Simpan gambar stego (output_path sudah didefinisikan di atas)
                cv2.imwrite(str(output_path), stego_img)
                
                success_count += 1
                
                # Print progress setiap 100 gambar atau di gambar terakhir
                if success_count % 100 == 0:
                    total_processed = success_count + skipped_count
                    print(f"   Progress: {total_processed}/{len(image_files)} checked ({success_count} processed, {skipped_count} skipped)")
                
            except Exception as e:
                print(f"   Error pada {image_path.name}: {str(e)}")
                error_count += 1
                continue
        
        # Print summary
        total_checked = success_count + skipped_count + error_count
        print(f"\n SUMMARY untuk {split}:")
        print(f"      Total files: {len(image_files)}")
        print(f"      Newly processed: {success_count}")
        print(f"      Skipped (already exists): {skipped_count}")
        print(f"      Errors: {error_count}")
        print(f"      Total checked: {total_checked}/{len(image_files)}")


def main():
    """
    Fungsi utama untuk menjalankan batch processing LSB embedding
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='LSB Embedding Batch Processing dengan fitur Resume',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh penggunaan:
  python embedding_lsb_bpp.py              # Mode resume (skip yang sudah ada)
  python embedding_lsb_bpp.py --force      # Overwrite semua file
  python embedding_lsb_bpp.py --resolutions res128 res256  # Hanya proses resolusi tertentu
        """
    )
    parser.add_argument(
        '--force', 
        action='store_true',
        help='Overwrite file yang sudah ada (default: False, akan skip)'
    )
    parser.add_argument(
        '--resolutions',
        nargs='+',
        default=['res128', 'res256', 'res512', 'res1024'],
        help='List resolusi yang akan diproses (default: semua)'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("LSB EMBEDDING - BATCH PROCESSING WITH RESUME")
    print("="*60)
    print("Payload: 0.3 bpp (bits per pixel)")
    print("Metode: LSB Standar (bit ke-0) - SEQUENTIAL POSITIONING")
    print(f"Mode: {'FORCE OVERWRITE' if args.force else 'RESUME (skip existing)'}")
    print(f"Resolusi: {', '.join(args.resolutions)}")
    print()
    
    # Set random seed untuk reproducibility (optional)
    np.random.seed(42)
    
    # Proses setiap resolusi
    for resolution in args.resolutions:
        try:
            process_resolution_folder(resolution, force_overwrite=args.force)
        except Exception as e:
            print(f"\n Error pada {resolution}: {str(e)}")
            continue
    
    print("\n" + "="*60)
    print(" PROSES EMBEDDING SELESAI!")
    print("="*60)


if __name__ == "__main__":
    main()
