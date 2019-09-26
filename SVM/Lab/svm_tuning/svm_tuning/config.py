"""
A centralized place for storing paths and common config
"""
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

MODELS_SUBDIR = os.path.join(ROOT_DIR, '../models')
