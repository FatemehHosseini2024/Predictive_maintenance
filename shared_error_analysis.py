import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error


def analyze_residuals(y_true, y_pred):
    residuals = y_true - y_pred

    print("Residual summary (y_true - y_pred):")
    print(residuals.describe())
    print(f"\nMean residual: {residuals.mean():.3f}")
    print("(Positive = underestimate, Negative = overestimate)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(residuals, bins=50, edgecolor="black", alpha=0.7)
    axes[0].axvline(0, color="red", linestyle="--", label="zero error")
    axes[0].axvline(residuals.mean(), color="green", linestyle="--", label=f"mean = {residuals.mean():.2f}")
    axes[0].set_title("Histogram of Residuals")
    axes[0].set_xlabel("Residual (y_true - y_pred)")
    axes[0].legend()

    axes[1].boxplot(residuals.values, vert=True)
    axes[1].axhline(0, color="red", linestyle="--")
    axes[1].set_title("Boxplot of Residuals")
    axes[1].set_ylabel("Residual")

    plt.tight_layout()
    plt.savefig("error_residuals.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: error_residuals.png")

    return residuals


def analyze_per_bin(y_true, y_pred, rul_clip=125, n_bins=5):
    bins = np.linspace(0, rul_clip, n_bins + 1)
    error_df = pd.DataFrame({
        "y_true": y_true.values,
        "y_pred": y_pred,
    })
    error_df["residual"] = error_df["y_true"] - error_df["y_pred"]
    error_df["abs_error"] = error_df["residual"].abs()
    error_df["RUL_bin"] = pd.cut(error_df["y_true"], bins=bins, include_lowest=True)

    summary = error_df.groupby("RUL_bin", observed=True).agg(
        mean_residual=("residual", "mean"),
        mean_abs_error=("abs_error", "mean"),
        count=("residual", "count"),
    )
    print("\nPer-bin error summary:")
    print(summary)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    summary["mean_residual"].plot(kind="bar", ax=axes[0], color="steelblue")
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_title("Mean Residual by RUL Bin")
    axes[0].set_ylabel("Mean Residual")

    summary["mean_abs_error"].plot(kind="bar", ax=axes[1], color="darkorange")
    axes[1].set_title("Mean Absolute Error by RUL Bin")
    axes[1].set_ylabel("Mean Absolute Error")

    plt.tight_layout()
    plt.savefig("error_per_bin.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: error_per_bin.png")

    return summary


def analyze_per_engine(df_test_fe, y_true, y_pred, rul_clip=125):
    error_df = pd.DataFrame({
        "unit_id": df_test_fe["unit_id"].values,
        "y_true": y_true.values,
        "y_pred": y_pred,
    })
    error_df["residual"] = error_df["y_true"] - error_df["y_pred"]
    error_df["abs_error"] = error_df["residual"].abs()

    per_engine = error_df.groupby("unit_id").agg(
        mean_residual=("residual", "mean"),
        mean_abs_error=("abs_error", "mean"),
        max_abs_error=("abs_error", "max"),
        n_records=("residual", "count"),
    ).reset_index()

    print("\nTop 10 engines with highest mean absolute error:")
    print(per_engine.sort_values("mean_abs_error", ascending=False).head(10).to_string(index=False))

    print("\nTop 10 engines with lowest mean absolute error:")
    print(per_engine.sort_values("mean_abs_error").head(10).to_string(index=False))

    print("\nDistribution of mean absolute error across engines:")
    print(per_engine["mean_abs_error"].describe())

    fig, ax = plt.subplots(figsize=(12, 5))
    sorted_eng = per_engine.sort_values("mean_abs_error").reset_index(drop=True)
    ax.bar(range(len(sorted_eng)), sorted_eng["mean_abs_error"], color="steelblue")
    ax.set_xlabel("Engine (sorted by error)")
    ax.set_ylabel("Mean Absolute Error")
    ax.set_title("Per-Engine Mean Absolute Error (sorted)")
    ax.axhline(per_engine["mean_abs_error"].mean(), color="red", linestyle="--", label="overall mean")
    ax.legend()
    plt.tight_layout()
    plt.savefig("error_per_engine.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: error_per_engine.png")

    return per_engine


def analyze_feature_importance(model, feature_cols, dataset_name):
    importances = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    print(f"\nTop 20 features ({dataset_name}):")
    print(importances.head(20).to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 8))
    top20 = importances.head(20).sort_values("importance")
    ax.barh(top20["feature"], top20["importance"], color="steelblue")
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top 20 Feature Importances - Random Forest ({dataset_name})")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: feature_importance.png")

    def classify_feature(name):
        if "_roll_mean_" in name:
            return "rolling_mean"
        elif "_roll_std_" in name:
            return "rolling_std"
        elif "_ewma_" in name:
            return "ewma"
        elif "_diff2" in name:
            return "diff2_ewma"
        elif "_slope_" in name:
            return "slope"
        elif name in ["cycle"]:
            return "cycle"
        else:
            return "raw_sensor"

    importances["feature_type"] = importances["feature"].apply(classify_feature)
    type_summary = importances.groupby("feature_type")["importance"].sum().sort_values(ascending=False)
    print("\nTotal importance by feature type:")
    print(type_summary)

    fig, ax = plt.subplots(figsize=(8, 5))
    type_summary.plot(kind="bar", ax=ax, color="teal")
    ax.set_title("Total Importance by Feature Type")
    ax.set_ylabel("Total Importance")
    plt.tight_layout()
    plt.savefig("feature_importance_by_type.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: feature_importance_by_type.png")

    return importances


def analyze_shap(model, X_test, feature_cols, Y_test, y_pred, dataset_name, sample_size=1000):
    try:
        import shap
    except ImportError:
        print("\nSHAP not installed, skipping SHAP analysis.")
        return None

    print("\n" + "=" * 60)
    print(f"SHAP Feature Importance on Test Set ({dataset_name})")
    print("=" * 60)

    X_sample = X_test.sample(min(sample_size, len(X_test)), random_state=42)
    y_sample = y_pred[X_sample.index]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_df = pd.DataFrame(shap_values, columns=feature_cols, index=X_sample.index)
    print(f"SHAP values computed on {len(X_sample)} samples")

    bins = np.linspace(0, Y_test.max(), 6)
    X_sample_with_rul = X_sample.copy()
    X_sample_with_rul["RUL"] = Y_test.loc[X_sample.index].values
    X_sample_with_rul["RUL_bin"] = pd.cut(X_sample_with_rul["RUL"], bins=bins, include_lowest=True)

    overall_mean_abs = shap_df.abs().mean().sort_values(ascending=False)
    top_feats = overall_mean_abs.head(10).index.tolist()

    comparison = {}
    for bin_label in X_sample_with_rul["RUL_bin"].cat.categories:
        mask = X_sample_with_rul["RUL_bin"] == bin_label
        mean_shap = shap_df[mask].mean(axis=0)
        comparison[str(bin_label)] = mean_shap

    comparison_df = pd.DataFrame(comparison)
    comparison_df.loc["overall"] = shap_df.mean(axis=0)

    print("\nMean SHAP per feature per RUL bin (top 10 features):")
    for feat in top_feats:
        vals = []
        for col in comparison_df.columns:
            v = comparison_df.loc[feat, col]
            sign = "+" if v >= 0 else "-"
            vals.append(f"{sign}{v:.4f}")
        print(f"  {feat}: {vals}")

    shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, show=False)
    plt.savefig("shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved: shap_summary.png")
    print("SHAP analysis complete.")

    return shap_df


def run_full_error_analysis(df_test_fe, X_test, y_test_clipped, y_test_pred, model, feature_cols, dataset_name, rul_clip=125):
    print("=" * 60)
    print(f"Error Analysis - {dataset_name}")
    print("=" * 60)

    print("\n=== Section 1: Residual Distribution ===")
    analyze_residuals(y_test_clipped, pd.Series(y_test_pred, index=y_test_clipped.index))

    print("\n=== Section 2: Error by RUL Bin ===")
    analyze_per_bin(y_test_clipped, y_test_pred, rul_clip=rul_clip)

    print("\n=== Section 3: Per-Engine Error ===")
    analyze_per_engine(df_test_fe, y_test_clipped, y_test_pred, rul_clip=rul_clip)

    print("\n=== Section 4: Feature Importance ===")
    analyze_feature_importance(model, feature_cols, dataset_name)

    print("\n=== Section 5: SHAP Analysis ===")
    analyze_shap(model, X_test, feature_cols, df_test_fe["RUL"], y_test_pred, dataset_name)