import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from config import get_config
from shared_train import run_validation

if __name__ == "__main__":
    config = get_config("FD001")
    print("=" * 60)
    print(f"Final Model Training for {config.name} - State 2: EWMA replaces Rolling")
    print("=" * 60)
    results = run_validation(config)