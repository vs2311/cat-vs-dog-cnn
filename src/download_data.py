"""
Download the Microsoft "Cats and Dogs" dataset (the same images as Kaggle's
Dogs vs Cats), clean out corrupt files, and split it into
data/train, data/val, data/test (80/10/10).

Usage:
    python -m src.download_data                # 5000 images per class
    python -m src.download_data --limit 0      # use ALL images (~12,500 per class)
    python -m src.download_data --zip path.zip # use a zip you already downloaded
"""
import argparse
import random
import shutil
import zipfile

import tensorflow as tf
from PIL import Image

from .config import DATA_DIR, ROOT, SEED

URL = (
    "https://download.microsoft.com/download/3/E/1/"
    "3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
)
SOURCE_FOLDERS = {"cats": "Cat", "dogs": "Dog"}


def get_zip(zip_arg):
    if zip_arg:
        return zip_arg
    print("Downloading dataset (~800 MB)...")
    return tf.keras.utils.get_file(
        "catsdogs.zip", origin=URL, cache_dir=str(ROOT / ".cache"), cache_subdir=""
    )


def clean_copy(src, dst):
    """Re-save as a clean RGB JPEG. Returns False if the file is corrupt."""
    try:
        with Image.open(src) as im:
            im.convert("RGB").save(dst, "JPEG", quality=90)
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=5000,
                    help="max images per class (0 = all)")
    ap.add_argument("--zip", default=None, help="path to an existing zip file")
    args = ap.parse_args()

    zip_path = get_zip(args.zip)
    extract_dir = ROOT / ".cache" / "extracted"
    if not extract_dir.exists():
        print("Extracting...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract_dir)

    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)

    rng = random.Random(SEED)
    for cls, src_name in SOURCE_FOLDERS.items():
        files = sorted((extract_dir / "PetImages" / src_name).glob("*.jpg"))
        rng.shuffle(files)
        if args.limit:
            files = files[: args.limit * 2]  # extra, since some are corrupt

        good = []
        for i, f in enumerate(files):
            if args.limit and len(good) >= args.limit:
                break
            tmp = DATA_DIR / "_tmp" / cls
            tmp.mkdir(parents=True, exist_ok=True)
            out = tmp / f"{cls[:-1]}_{i}.jpg"
            if clean_copy(f, out):
                good.append(out)

        n = len(good)
        splits = {
            "train": good[: int(0.8 * n)],
            "val": good[int(0.8 * n): int(0.9 * n)],
            "test": good[int(0.9 * n):],
        }
        for split, items in splits.items():
            d = DATA_DIR / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for p in items:
                shutil.move(str(p), d / p.name)
        print(f"{cls}: {n} images -> "
              f"{len(splits['train'])} train / {len(splits['val'])} val / "
              f"{len(splits['test'])} test")

    shutil.rmtree(DATA_DIR / "_tmp", ignore_errors=True)
    print(f"Done. Dataset ready in {DATA_DIR}")


if __name__ == "__main__":
    main()
