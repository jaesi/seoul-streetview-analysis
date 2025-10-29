"""
Configuration settings for Seoul Street View Analysis project.
"""

import os
from pathlib import Path
from typing import Dict, List

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Location configurations
LOCATIONS: Dict[str, Dict] = {
    "hongdae": {
        "name": "Hongdae (홍대)",
        "start_coords": (37.554197, 126.922500),
        "end_coords": (37.550833, 126.921323),
        "num_points": 40,
        "description": "Vibrant youth culture district with high pedestrian activity"
    },
    "syarosu": {
        "name": "Syarosu-gil (샤로수길)",
        "start_coords": (37.479241, 126.952545),
        "end_coords": (37.479476, 126.944457),
        "num_points": 20,
        "description": "Trendy commercial area with cafes and boutiques"
    },
    "ssook": {
        "name": "Sookgogae-gil (쑥고개길)",
        "start_coords": (37.478701, 126.952144),
        "end_coords": (37.479476, 126.944457),
        "num_points": 20,
        "description": "Residential area with mixed commercial use"
    }
}

# Image collection settings
IMAGE_SETTINGS = {
    "size": "256x256",
    "headings": [0, 90, 180, 270],  # Camera headings in degrees
    "pitch": "0"
}

# Segmentation settings
SEGMENTATION_SETTINGS = {
    "threshold": 0.6,
    "model": "fcn_resnet50",
    "batch_size": 1
}

# Feature names for simplified segmentation output
FEATURE_NAMES: List[str] = [
    'unlabelled',
    'ground',
    'building',
    'road',
    'green',
    'sky',
    'pedestrian',
    'building2'
]

# ML Model settings
ML_SETTINGS = {
    "test_size": 0.2,
    "random_state": 42,
    "cv_folds": 5,
    "n_jobs": -1  # Use all CPU cores
}

# Model hyperparameter grids
PARAM_GRIDS = {
    'Decision Tree': {'max_depth': list(range(1, 11))},
    'Random Forest': {'n_estimators': list(range(100, 501, 100))},
    'Gradient Boosting': {'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.09, 0.1]},
    'Support Vector Machine': {'kernel': ['linear', 'poly', 'rbf']},
    'K-Nearest Neighbors': {'n_neighbors': list(range(10, 101, 10))}
}

# Logging settings
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "standard",
            "filename": str(PROJECT_ROOT / "logs" / "app.log"),
            "mode": "a"
        }
    },
    "loggers": {
        "": {
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": True
        }
    }
}

# Create logs directory
(PROJECT_ROOT / "logs").mkdir(exist_ok=True)
