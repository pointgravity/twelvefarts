"""
Utility functions.
"""

import os
import torch

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)