import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from config import get_config
from shared_train import run_full_retrain
from shared_error_analysis import run_full_error_analysis

if __name__ == "__main__":
    config = get_config("FD002")
    print("=" * 60)
    print(f"Error Analysis - {config.name}")
    print("=" * 60)

    results = run_full_retrain(config)

    run_full_error_analysis(
        df_test_fe=results["df_test_fe"],
        X_test=results["X_test"],
        y_test_clipped=results["y_test_clipped"],
        y_test_pred=results["y_test_pred"],
        model=results["model"],
        feature_cols=results["feature_cols"],
        dataset_name=config.name,
        rul_clip=config.rul_clip,
    )