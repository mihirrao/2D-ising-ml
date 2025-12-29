import os
import numpy as np

def save_npz(path: str, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **arrays)

def load_npz(path: str):
    return dict(np.load(path, allow_pickle=False))