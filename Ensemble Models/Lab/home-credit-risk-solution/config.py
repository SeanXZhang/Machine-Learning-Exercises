"""
A centralized place for storing paths and common config
"""
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_SUBDIR = os.path.join(ROOT_DIR, 'data')

MODELS_SUBDIR = os.path.join(ROOT_DIR, 'models')
RESULTS_DATA_SUB = os.path.join(ROOT_DIR, 'results/processed_data')
RESULTS_MODEL_SUB = os.path.join(ROOT_DIR, 'results/models')
