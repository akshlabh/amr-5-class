"""
prepare_dataset.py — Filter RML2016.10a to a class subset and save a smaller pickle
=====================================================================================
Run this ONCE locally before uploading to Kaggle.

Usage
-----
# Create 5-class pickle (primary model)
python scripts/prepare_dataset.py \
    --src  data/RML2016.10a_dict.pkl \
    --dst  data/RML2016.10a_5class.pkl \
    --classes BPSK QPSK 8PSK QAM16 QAM64

# Create 4-class pickle (ablation — no BPSK)
python scripts/prepare_dataset.py \
    --src  data/RML2016.10a_dict.pkl \
    --dst  data/RML2016.10a_4class.pkl \
    --classes QPSK 8PSK QAM16 QAM64

After running, upload each .pkl to a separate Kaggle Dataset:
    rml2016-5class  → RML2016.10a_5class.pkl
    rml2016-4class  → RML2016.10a_4class.pkl
"""

import pickle
import hashlib
import argparse
import os


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def filter_and_save(src: str, dst: str, classes: list) -> None:
    print(f"[prepare] Loading {src} ...")
    Xd = pickle.load(open(src, 'rb'), encoding='iso-8859-1')

    all_classes = sorted(set(k[0] for k in Xd.keys()))
    print(f"[prepare] All classes in source: {all_classes}")

    missing = set(classes) - set(all_classes)
    if missing:
        raise ValueError(
            f"Requested classes not found in pickle: {missing}\n"
            f"Available: {all_classes}"
        )

    filtered = {k: v for k, v in Xd.items() if k[0] in classes}
    kept_classes = sorted(set(k[0] for k in filtered.keys()))
    print(f"[prepare] Keeping {len(filtered)} entries for classes: {kept_classes}")

    os.makedirs(os.path.dirname(dst) if os.path.dirname(dst) else '.', exist_ok=True)
    pickle.dump(filtered, open(dst, 'wb'))

    src_md5 = md5(src)
    dst_md5 = md5(dst)
    src_mb  = os.path.getsize(src) / 1e6
    dst_mb  = os.path.getsize(dst) / 1e6

    print(f"\n[prepare] Done.")
    print(f"  Source : {src}  ({src_mb:.1f} MB)  MD5={src_md5}")
    print(f"  Output : {dst}  ({dst_mb:.1f} MB)  MD5={dst_md5}")
    print(f"  Saved {len(classes)} classes, {len(filtered)} (mod, snr) pairs.")
    print(f"\n  >> Add these MD5 hashes to CHANGELOG.md for reproducibility!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Filter RML2016.10a to a class subset')
    parser.add_argument('--src',     required=True,
                        help='Path to RML2016.10a_dict.pkl (original full dataset)')
    parser.add_argument('--dst',     required=True,
                        help='Output path for filtered pickle')
    parser.add_argument('--classes', nargs='+', required=True,
                        help='Class names to include (space-separated)')
    args = parser.parse_args()
    filter_and_save(args.src, args.dst, args.classes)
