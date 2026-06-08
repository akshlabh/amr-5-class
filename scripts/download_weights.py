"""
download_weights.py — Download trained weights from GitHub Releases
====================================================================
Usage
-----
    # Download 5-class baseline weights
    python scripts/download_weights.py \
        --tag   v1.0-5class-baseline \
        --asset mcldnn_5class_baseline_v1.0_best.weights.h5 \
        --dst   experiments/5class_baseline/checkpoints/

    # Download 4-class ablation weights
    python scripts/download_weights.py \
        --tag   v1.0-4class-ablation \
        --asset mcldnn_4class_ablation_v1.0_best.weights.h5 \
        --dst   experiments/4class_ablation/checkpoints/

Requirements
------------
    pip install requests
"""

import os
import sys
import argparse
import hashlib

try:
    import requests
except ImportError:
    sys.exit("Install requests first:  pip install requests")

# ── Change this to your GitHub username and repo name ─────────────────────────
GITHUB_REPO = "YOUR_USERNAME/AMR"          # <-- update this


def download_asset(tag: str, asset: str, dst_dir: str,
                   repo: str = GITHUB_REPO) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    dst_path = os.path.join(dst_dir, asset)

    url = (f"https://github.com/{repo}/releases/download/{tag}/{asset}")
    print(f"[download] Fetching: {url}")

    r = requests.get(url, stream=True)
    if r.status_code != 200:
        sys.exit(f"[download] Error {r.status_code}: {r.text[:200]}")

    total = int(r.headers.get('content-length', 0))
    downloaded = 0
    with open(dst_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = 100 * downloaded / total
                print(f"\r  {pct:.1f}%  ({downloaded/1e6:.1f} MB)", end='')
    print()

    md5 = hashlib.md5(open(dst_path, 'rb').read()).hexdigest()
    print(f"[download] Saved to {dst_path}  MD5={md5}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Download weights from GitHub Releases')
    p.add_argument('--tag',   required=True, help='GitHub Release tag, e.g. v1.0-5class-baseline')
    p.add_argument('--asset', required=True, help='Filename of the release asset')
    p.add_argument('--dst',   required=True, help='Local directory to save the file')
    p.add_argument('--repo',  default=GITHUB_REPO,
                   help=f'GitHub repo (default: {GITHUB_REPO})')
    args = p.parse_args()
    download_asset(args.tag, args.asset, args.dst, args.repo)
