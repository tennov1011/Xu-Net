"""Rename dataset files by adding split prefix (e.g. train_1.png, test_2.png)"""
import os
from pathlib import Path
import shutil

def rename_with_prefix(folder_path, split):
    """Rename files in folder by adding split prefix: 1.png -> {split}_1.png"""
    # Only process files that have NOT already been prefixed
    all_files = list(Path(folder_path).glob('*.png'))
    files = sorted(
        [f for f in all_files if not f.stem.startswith(f"{split}_")],
        key=lambda f: int(f.stem) if f.stem.isdigit() else 0
    )

    if len(files) == 0:
        already = len(all_files)
        if already > 0:
            print(f"   ⏭️  Skipped: all {already} files already have prefix '{split}_'")
        else:
            print(f"   ⚠️  No PNG files found")
        return

    print(f"📁 Processing: {folder_path}")
    print(f"   Found {len(files)} files")

    # Create temporary directory to avoid name collisions
    temp_dir = Path(folder_path) / "_temp"
    temp_dir.mkdir(exist_ok=True)

    # Copy with new names to temp
    for idx, old_file in enumerate(files, start=1):
        new_name = f"{split}_{old_file.stem}.png"
        shutil.copy2(old_file, temp_dir / new_name)
        if idx % 1000 == 0 or idx == len(files):
            print(f"   ✅ Processed {idx}/{len(files)} files...")

    # Remove old files
    for old_file in files:
        old_file.unlink()

    # Move new files back
    for new_file in temp_dir.glob('*.png'):
        shutil.move(str(new_file), folder_path)

    # Remove temp directory
    temp_dir.rmdir()

    print(f"   ✅ Renamed {len(files)} files with prefix '{split}_'")
    print()

def rename_all_datasets(base_path):
    """Rename all datasets across all resolutions and splits"""
    resolutions = ['res128', 'res256', 'res512', 'res1024']
    splits = ['train', 'validation', 'test']
    types = ['cover', 'stego']

    print("="*60)
    print("DATASET RENAMING SCRIPT")
    print("="*60)
    print("⚠️  WARNING: This will rename all files!")
    print("   e.g. 1.png in train/ → train_1.png")
    print()
    response = input("Continue? (yes/no): ")

    if response.lower() != 'yes':
        print("❌ Aborted!")
        return

    print("\nStarting renaming process...\n")

    total_processed = 0

    for res in resolutions:
        print(f"📦 {res.upper()}")
        for split in splits:
            for img_type in types:
                folder = Path(base_path) / res / split / img_type
                if folder.exists():
                    rename_with_prefix(folder, split)
                    total_processed += 1
                else:
                    print(f"   ⚠️  Not found: {folder}")
                    print()

    print("="*60)
    print(f"✅ RENAMING COMPLETE!")
    print(f"   Processed {total_processed} folders")
    print("="*60)

if __name__ == "__main__":
    rename_all_datasets("./dataset")

