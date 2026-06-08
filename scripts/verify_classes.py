"""
verify_classes.py — Print all class names in an RML2016.10a pickle
===================================================================
Run this first to confirm the exact class name strings in your pickle file.

Usage
-----
    python scripts/verify_classes.py data/RML2016.10a_dict.pkl
"""

import pickle
import sys


def verify(path: str) -> None:
    print(f"Loading {path} ...")
    Xd = pickle.load(open(path, 'rb'), encoding='iso-8859-1')

    mods = sorted(set(k[0] for k in Xd.keys()))
    snrs = sorted(set(k[1] for k in Xd.keys()))

    print(f"\nClasses ({len(mods)}):\n  {mods}")
    print(f"\nSNR range ({len(snrs)} levels):\n  {snrs}")
    print(f"\nSample count per (mod, snr): {Xd[list(Xd.keys())[0]].shape}")
    print(f"Total keys (mod x snr pairs): {len(Xd)}")
    print(f"Total samples: {len(Xd) * Xd[list(Xd.keys())[0]].shape[0]}")

    print("\n--- 5-class target verification ---")
    target_5 = ['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64']
    for cls in target_5:
        status = '✓' if cls in mods else '✗ NOT FOUND'
        print(f"  {cls:8s} {status}")

    print("\n--- 4-class ablation verification ---")
    target_4 = ['QPSK', '8PSK', 'QAM16', 'QAM64']
    for cls in target_4:
        status = '✓' if cls in mods else '✗ NOT FOUND'
        print(f"  {cls:8s} {status}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python scripts/verify_classes.py <path/to/RML2016.10a_dict.pkl>")
        sys.exit(1)
    verify(sys.argv[1])
