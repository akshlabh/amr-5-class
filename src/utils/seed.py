"""
seed.py — Global reproducibility seed setter
=============================================
Call set_all_seeds(seed) at the very top of any script or notebook
before importing TensorFlow / Keras to ensure reproducible results.
"""

import os
import random
import numpy as np


def set_all_seeds(seed: int = 2016) -> None:
    """
    Fix seeds for Python, NumPy, and TensorFlow.

    Parameters
    ----------
    seed : int  (default 2016, matches original AMR-Benchmark convention)
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # TensorFlow seed — import lazily so this module can be imported
    # even without TF installed
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        print(f"[seed] All seeds fixed to {seed}  "
              f"(Python, NumPy, TensorFlow {tf.__version__})")
    except ImportError:
        print(f"[seed] TensorFlow not found — "
              f"seeded Python and NumPy only (seed={seed})")
