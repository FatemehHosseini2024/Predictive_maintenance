import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from config import get_config
from shared_train import run_full_retrain
from shared_features import apply_state2_features
from shared_eval import evaluate_per_bin
from utils import clip_rul

st.set_page_config(
    page_title="Predictive Maintenance - RUL Prediction",
    page_icon="🔧",
    layout="wide",
)


@st.cache_resource
def load_model_and_data(dataset_name):
    config = get_config(dataset_name)
    results = run_full_retrain(config)
    return config, results


def show_metrics(results, config):
    st.header(f"{config.name} - Test Set Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE", f"{results['test_rmse']:.3f}")
    col2.metric("MAE", f"{results['test_mae']:.3f}")
    col3.metric("Features", str(len(results["feature_cols"])))

    st.subheader("Per-Bin Metrics (RUL bins of 25)")
    per_bin_data = results["per_bin_metrics"]
    per_bin_df = pd.DataFrame([
        {
            "Bin": k,
            "RMSE": f"{v['rmse']:.3f}",
            "MAE": f"{v['mae']:.3f}",
            "Samples": v["n"],
        }
        for k, v in per_bin_data.items()
    ])
    st.dataframe(per_bin_df, use_container_width=True)


def show_feature_importance(results):
    st.subheader("Feature Importance")

    fi = results["feature_importance"]
    top20 = fi.head(20).sort_values("importance")

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top20["feature"], top20["importance"], color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title("Top 20 Feature Importances")
    plt.tight_layout()
    st.pyplot(fig)

    def classify_feature(name):
        if "_ewma_" in name and "_diff2" not in name:
            return "ewma"
        elif "_diff2" in name:
            return "diff2_ewma"
        elif "_slope_" in name:
            return "slope"
        elif name in ["cycle"]:
            return "cycle"
        else:
            return "raw_sensor"

    fi_copy = fi.copy()
    fi_copy["feature_type"] = fi_copy["feature"].apply(classify_feature)
    type_summary = fi_copy.groupby("feature_type")["importance"].sum().sort_values(ascending=False)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    type_summary.plot(kind="bar", ax=ax2, color="teal")
    ax2.set_title("Total Importance by Feature Type")
    ax2.set_ylabel("Total Importance")
    plt.tight_layout()
    st.pyplot(fig2)


def show_per_engine_error(results, config):
    st.subheader("Per-Engine Error Analysis")

    df_test_fe = results["df_test_fe"]
    y_test_clipped = results["y_test_clipped"]
    y_test_pred = pd.Series(results["y_test_pred"], index=y_test_clipped.index)

    error_df = pd.DataFrame({
        "unit_id": df_test_fe["unit_id"].values,
        "y_true": y_test_clipped.values,
        "y_pred": y_test_pred,
    })
    error_df["abs_error"] = (error_df["y_true"] - error_df["y_pred"]).abs()

    per_engine = error_df.groupby("unit_id").agg(
        mean_abs_error=("abs_error", "mean"),
        max_abs_error=("abs_error", "max"),
        n_records=("abs_error", "count"),
    ).reset_index()

    st.write(f"Engines in test set: {per_engine['unit_id'].nunique()}")
    st.write(f"Mean MAE per engine: {per_engine['mean_abs_error'].mean():.3f}")

    sorted_eng = per_engine.sort_values("mean_abs_error").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(sorted_eng)), sorted_eng["mean_abs_error"], color="steelblue")
    ax.axhline(per_engine["mean_abs_error"].mean(), color="red", linestyle="--", label="Overall Mean")
    ax.set_xlabel("Engine (sorted by MAE)")
    ax.set_ylabel("Mean Absolute Error")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("**Top 10 Worst Engines:**")
    st.dataframe(per_engine.nlargest(10, "mean_abs_error"), use_container_width=True)

    st.markdown("**Top 10 Best Engines:**")
    st.dataframe(per_engine.nsmallest(10, "mean_abs_error"), use_container_width=True)


def show_residual_analysis(results):
    st.subheader("Residual Distribution")

    y_test_clipped = results["y_test_clipped"]
    y_test_pred = pd.Series(results["y_test_pred"], index=y_test_clipped.index)
    residuals = y_test_clipped - y_test_pred

    st.write(f"Mean residual: {residuals.mean():.3f}")
    st.write("Positive = underestimate, Negative = overestimate")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(residuals, bins=50, edgecolor="black", alpha=0.7)
        ax.axvline(0, color="red", linestyle="--", label="zero error")
        ax.axvline(residuals.mean(), color="green", linestyle="--", label=f"mean = {residuals.mean():.2f}")
        ax.set_xlabel("Residual (y_true - y_pred)")
        ax.legend()
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.boxplot(residuals.values, vert=True)
        ax.axhline(0, color="red", linestyle="--")
        st.pyplot(fig)


def show_prediction_explorer(results, config):
    st.subheader("Interactive Prediction Explorer")

    df_test_fe = results["df_test_fe"]
    y_test_clipped = results["y_test_clipped"]
    y_test_pred = pd.Series(results["y_test_pred"], index=y_test_clipped.index)
    feature_cols = results["feature_cols"]
    model = results["model"]

    df_test_fe = df_test_fe.copy()
    df_test_fe["y_true"] = y_test_clipped.values
    df_test_fe["y_pred"] = y_test_pred.values
    df_test_fe["abs_error"] = df_test_fe["y_true"] - df_test_fe["y_pred"]

    unit_ids = sorted(df_test_fe["unit_id"].unique())
    selected_unit = st.selectbox("Select Engine", unit_ids, index=0)

    unit_data = df_test_fe[df_test_fe["unit_id"] == selected_unit]
    unit_data_sorted = unit_data.sort_values("cycle")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(unit_data_sorted["cycle"], unit_data_sorted["y_true"], "b-", label="True RUL", linewidth=2)
    ax.plot(unit_data_sorted["cycle"], unit_data_sorted["y_pred"], "r--", label="Predicted RUL", linewidth=2)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("RUL")
    ax.set_title(f"Engine {selected_unit} - True vs Predicted RUL")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Engine MAE: {unit_data['abs_error'].abs().mean():.3f}")
        st.write(f"Engine Max Error: {unit_data['abs_error'].abs().max():.3f}")
        st.write(f"Samples: {len(unit_data)}")

    with col2:
        feature_importance = pd.DataFrame({
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        st.write("Top 5 features:")
        st.dataframe(feature_importance.head(5), use_container_width=True)


@st.cache_data
def load_test_sample_predictions(dataset_name, n_samples=1000):
    config, results = load_model_and_data(dataset_name)
    X_test = results["X_test"]
    y_test_clipped = results["y_test_clipped"]
    y_test_pred = results["y_test_pred"]

    sample_idx = X_test.sample(min(n_samples, len(X_test)), random_state=42).index
    X_sample = X_test.loc[sample_idx]
    y_sample_true = y_test_clipped.loc[sample_idx]
    y_sample_pred = pd.Series(y_test_pred, index=y_test_clipped.index).loc[sample_idx]

    return X_sample, y_sample_true, y_sample_pred


def show_shap_analysis(dataset_name):
    st.subheader("SHAP Feature Importance (Sample)")
    st.caption("Computed on 1000 random test samples. May take a few seconds.")

    X_sample, y_sample_true, y_sample_pred = load_test_sample_predictions(dataset_name)

    try:
        import shap
    except ImportError:
        st.warning("shap not installed. Install with `pip install shap`.")
        return

    config, results = load_model_and_data(dataset_name)
    model = results["model"]
    feature_cols = results["feature_cols"]

    with st.spinner("Computing SHAP values..."):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

    shap.summary = shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, show=False)
    st.pyplot(plt.gcf())
    plt.close()

    mean_abs_shap = pd.DataFrame(shap_values, columns=feature_cols).abs().mean().sort_values(ascending=False)
    st.write("Mean |SHAP| by feature (top 15):")
    st.dataframe(mean_abs_shap.head(15), use_container_width=True)


def main():
    st.sidebar.title("Predictive Maintenance Dashboard")
    st.sidebar.markdown("---")

    dataset_name = st.sidebar.selectbox("Dataset", ["FD002", 'FD001'], index=0)

    st.sidebar.markdown("---")
    st.sidebar.info(
        "Uses RandomForest with EWMA + Second-Diff features (State 2).\n\n"
        "Datasets: NASA C-MAPSS Turbofan Engine Degradation Simulation."
    )

    st.title("🔧 Predictive Maintenance - RUL Prediction")
    st.markdown(f"### {dataset_name} Dashboard")

    with st.spinner(f"Training model on {dataset_name} (may take ~1-2 minutes)..."):
        config, results = load_model_and_data(dataset_name)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Metrics", "🔍 Feature Importance", "📈 Residuals", "🔎 Engine Explorer", "🔮 SHAP"
    ])

    with tab1:
        show_metrics(results, config)

    with tab2:
        show_feature_importance(results)

    with tab3:
        show_residual_analysis(results)
        show_per_bin_metrics(results)

    with tab4:
        show_prediction_explorer(results, config)

    with tab5:
        show_shap_analysis(dataset_name)

    st.markdown("---")
    st.caption(
        "Data: NASA C-MAPSS | Reference: Saxena et al., PHM08\n\n"
        "Model: RandomForest Regressor | Features: EWMA + Second-Diff (State 2)\n\n"
        f"Test RMSE: {results['test_rmse']:.3f} | Test MAE: {results['test_mae']:.3f} | Features: {len(results['feature_cols'])}"
    )


def show_per_bin_metrics(results):
    st.subheader("Error by RUL Bin")

    df_test_fe = results["df_test_fe"]
    y_test_clipped = results["y_test_clipped"]
    y_test_pred = pd.Series(results["y_test_pred"], index=y_test_clipped.index)

    error_df = pd.DataFrame({
        "y_true": y_test_clipped.values,
        "y_pred": y_test_pred,
    })
    error_df["residual"] = error_df["y_true"] - error_df["y_pred"]
    error_df["RUL_bin"] = pd.cut(error_df["y_true"], bins=np.linspace(0, 125, 6), include_lowest=True)

    summary = error_df.groupby("RUL_bin", observed=True).agg(
        mean_residual=("residual", "mean"),
        mae=("residual", lambda x: np.mean(np.abs(x))),
        count=("residual", "count"),
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    summary["mean_residual"].plot(kind="bar", ax=axes[0], color="steelblue")
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_title("Mean Residual by RUL Bin")
    axes[0].set_ylabel("Mean Residual")

    summary["mae"].plot(kind="bar", ax=axes[1], color="darkorange")
    axes[1].set_title("MAE by RUL Bin")
    axes[1].set_ylabel("MAE")

    plt.tight_layout()
    st.pyplot(fig)


if __name__ == "__main__":
    main()