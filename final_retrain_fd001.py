import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from config import get_config
from shared_train import run_full_retrain

if __name__ == "__main__":
    config = get_config("FD001")
    print("=" * 60)
    print(f"Final Retrain for {config.name} - Full Train Set")
    print("=" * 60)
    results = run_full_retrain(config)