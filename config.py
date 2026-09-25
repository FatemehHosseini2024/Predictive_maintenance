from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DatasetConfig:
    name: str
    trend_sensors: List[str]
    rul_clip: int
    ewma_spans: List[int]
    n_conditions: int
    rf_params: dict


FD001_CONFIG = DatasetConfig(
    name="FD001",
    trend_sensors=[
        "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_11",
        "sensor_12", "sensor_15", "sensor_17", "sensor_20", "sensor_21",
    ],
    rul_clip=125,
    ewma_spans=[5, 20],
    n_conditions=6,
    rf_params={
        "n_estimators": 406,
        "max_depth": 24,
        "min_samples_split": 6,
        "min_samples_leaf": 3,
        "max_features": "sqrt",
        "random_state": 42,
        "n_jobs": -1,
    },
)

FD002_CONFIG = DatasetConfig(
    name="FD002",
    trend_sensors=[
        "sensor_2", "sensor_3", "sensor_4", "sensor_6", "sensor_7", "sensor_8",
        "sensor_12", "sensor_13", "sensor_15", "sensor_17", "sensor_20", "sensor_21",
    ],
    rul_clip=125,
    ewma_spans=[5, 20],
    n_conditions=6,
    rf_params={
        "n_estimators": 221,
        "max_depth": 27,
        "min_samples_split": 26,
        "min_samples_leaf": 10,
        "max_features": None,
        "random_state": 42,
        "n_jobs": -1,
    },
)

DATASET_CONFIGS = {
    "FD001": FD001_CONFIG,
    "FD002": FD002_CONFIG,
}


def get_config(dataset_name: str) -> DatasetConfig:
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}")
    return DATASET_CONFIGS[dataset_name]