# configs/settings.py
# AresVision Configuration Loader
# Loads and validates all system settings from config.yaml

import yaml
import os
from pathlib import Path

# Root directory of the project
ROOT_DIR = Path(__file__).parent.parent

def load_config(config_path: str = None) -> dict:
    """
    Load configuration from yaml file.
    Falls back to default config.yaml if no path provided.
    """
    if config_path is None:
        config_path = ROOT_DIR / "configs" / "config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def get_project_config() -> dict:
    """Return project-level configuration."""
    return load_config()["project"]


def get_model_config() -> dict:
    """Return model configuration."""
    return load_config()["model"]


def get_inference_config() -> dict:
    """Return inference configuration."""
    return load_config()["inference"]


def get_api_config() -> dict:
    """Return API configuration."""
    return load_config()["api"]


def get_monitoring_config() -> dict:
    """Return monitoring configuration."""
    return load_config()["monitoring"]


def get_data_config() -> dict:
    """Return data configuration."""
    return load_config()["data"]


def get_benchmark_config() -> dict:
    """Return benchmark configuration."""
    return load_config()["benchmarks"]


# Validate critical paths exist on import
def validate_environment() -> bool:
    """
    Validate that critical directories exist.
    Returns True if environment is valid.
    """
    critical_dirs = [
        ROOT_DIR / "models",
        ROOT_DIR / "data" / "raw",
        ROOT_DIR / "data" / "processed",
        ROOT_DIR / "benchmarks",
    ]

    all_valid = True
    for directory in critical_dirs:
        if not directory.exists():
            print(f"WARNING: Missing directory: {directory}")
            all_valid = False

    return all_valid


if __name__ == "__main__":
    config = load_config()
    print("Configuration loaded successfully!")
    print(f"Project: {config['project']['name']}")
    print(f"Version: {config['project']['version']}")
    print(f"Model: {config['model']['name']}")
    print(f"Device: {config['model']['device']}")
    validate_environment()
