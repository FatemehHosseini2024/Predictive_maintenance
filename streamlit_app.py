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

from config import get_config
from shared_train import run_full_retrain
from utils import clip_rul

st.set_page_config(
    page_title="Predictive Maintenance - Engine Playback",
    page_icon="🔧",
    layout="wide",
)


@st.cache_resource
def load_model_and_data(dataset_name):
    config = get_config(dataset_name)
    results = run_full_retrain(config)
    df_test_fe = results["df_test_fe"]
    model = results["model"]
    feature_cols = results["feature_cols"]
    predictions = model.predict(df_test_fe[feature_cols])
    return config, results, predictions


def get_engine_info(df_test_fe, engine_id):
    engine_data = df_test_fe[df_test_fe["unit_id"] == engine_id].sort_values("cycle")
    max_cycle = int(engine_data["cycle"].max())
    min_cycle = int(engine_data["cycle"].min())
    return engine_data, min_cycle, max_cycle


st.sidebar.title("🔧 Real Engine Playback")
st.sidebar.markdown("---")

dataset_name = st.sidebar.selectbox("Dataset", ["FD002", "FD001"], index=0)

with st.spinner(f"Training model on {dataset_name} (may take ~1-2 minutes)..."):
    config, results, all_predictions = load_model_and_data(dataset_name)

df_test_fe = results["df_test_fe"]
model = results["model"]
feature_cols = results["feature_cols"]
y_test_clipped = results["y_test_clipped"]

df_test_fe = df_test_fe.copy()
df_test_fe["y_pred"] = all_predictions
df_test_fe["y_true"] = y_test_clipped.values

engine_ids = sorted(df_test_fe["unit_id"].unique())
selected_engine = st.sidebar.selectbox("Engine ID", engine_ids, index=0)

engine_full, min_cycle, max_cycle = get_engine_info(df_test_fe, selected_engine)

selected_cycle = st.sidebar.slider(
    "Cycle",
    min_value=min_cycle,
    max_value=max_cycle,
    value=max_cycle,
    step=1,
)

engine_data = engine_full[engine_full["cycle"] <= selected_cycle]

sensor_cols = [c for c in df_test_fe.columns if c.startswith("sensor_") and "_roll_" not in c and "_slope_" not in c and "_ewma" not in c]

default_sensors = [s for s in ["sensor_2", "sensor_4", "sensor_7", "sensor_11", "sensor_12", "sensor_15", "sensor_20", "sensor_21"] if s in sensor_cols]
selected_sensors = st.sidebar.multiselect(
    "Key Sensors to Display",
    options=sensor_cols,
    default=default_sensors[:5] if len(default_sensors) >= 5 else default_sensors,
)

st.title("🔧 Predictive Maintenance Dashboard")
st.markdown(f"### {dataset_name} - Engine {selected_engine} Playback")
st.markdown(f"Viewing cycles **{min_cycle}** to **{selected_cycle}** (max: {max_cycle})")

current_row = engine_full[engine_full["cycle"] == selected_cycle]
if len(current_row) == 0:
    current_row = engine_data.iloc[[-1]]

predicted_rul = float(current_row["y_pred"].iloc[0])
actual_rul = float(current_row["y_true"].iloc[0])
abs_error = abs(predicted_rul - actual_rul)
pct_error = (abs_error / actual_rul * 100) if actual_rul > 0 else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Predicted RUL", f"{predicted_rul:.1f}")
col2.metric("Actual RUL", f"{actual_rul:.1f}")
col3.metric("Absolute Error", f"{abs_error:.1f}")
col4.metric("Error %", f"{pct_error:.1f}%")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["📈 Sensor Trends", "📊 RUL Tracking", "📋 Raw Data", "🔍 Model Info"])

with tab1:
    if selected_sensors:
        fig, ax = plt.subplots(figsize=(14, 5))
        for sensor in selected_sensors:
            ax.plot(engine_data["cycle"], engine_data[sensor], label=sensor, linewidth=1.5)
        ax.axvline(x=selected_cycle, color="red", linestyle="--", alpha=0.7, label=f"Current cycle ({selected_cycle})")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Normalized Sensor Value")
        ax.set_title(f"Sensor Trends - Engine {selected_engine}")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Select at least one sensor to display trends.")

with tab2:
    st.subheader("Predicted vs Actual RUL (Cycle 1 to Current)")

    history = engine_data[["cycle", "y_true", "y_pred"]].copy()

    if len(history) > 0:
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(history["cycle"], history["y_true"], "b-", label="Actual RUL", linewidth=2)
        ax.plot(history["cycle"], history["y_pred"], "r--", label="Predicted RUL", linewidth=2)
        ax.fill_between(history["cycle"], history["y_true"], history["y_pred"], alpha=0.15, color="gray")
        ax.axvline(x=selected_cycle, color="green", linestyle=":", alpha=0.7, label=f"Playback head ({selected_cycle})")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("RUL")
        ax.set_title(f"RUL Tracking - Engine {selected_engine}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown(f"""
        As the slider moves toward the end of the engine's life, the model's prediction
        converges toward the actual RUL. Early predictions have wider error bands because
        EWMA and second-difference features have limited history.
        """)
    else:
        st.info("No data available for this engine up to the selected cycle.")

with tab3:
    st.subheader("Raw Data at Selected Cycle")

    display_cols = ["unit_id", "cycle"] + selected_sensors + ["y_true", "y_pred"] if selected_sensors else ["unit_id", "cycle", "y_true", "y_pred"]
    display_cols = [c for c in display_cols if c in current_row.columns]

    st.dataframe(current_row[display_cols].T, use_container_width=True)

    st.subheader("Full Engine Timeline (Cycles 1 to Selected)")
    timeline_cols = ["cycle", "y_true", "y_pred", "abs_error"] + selected_sensors if selected_sensors else ["cycle", "y_true", "y_pred", "abs_error"]
    timeline_cols = [c for c in timeline_cols if c in engine_data.columns]

    st.dataframe(engine_data[timeline_cols].round(3), use_container_width=True)

with tab4:
    st.subheader("Model Information")
    st.write(f"**Dataset:** {config.name}")
    st.write(f"**RUL Clip:** {config.rul_clip}")
    st.write(f"**Features:** {len(feature_cols)}")
    st.write(f"**RF Params:** {config.rf_params}")
    st.write(f"**Test RMSE:** {results['test_rmse']:.3f}")
    st.write(f"**Test MAE:** {results['test_mae']:.3f}")

    st.subheader("Per-Bin Test Performance")
    per_bin_df = pd.DataFrame([
        {"Bin": k, "RMSE": f"{v['rmse']:.3f}", "MAE": f"{v['mae']:.3f}", "Samples": v["n"]}
        for k, v in results["per_bin_metrics"].items()
    ])
    st.dataframe(per_bin_df, use_container_width=True)

    st.subheader("Top 10 Features")
    fi = results["feature_importance"].head(10)
    st.dataframe(fi, use_container_width=True)

    st.markdown("---")
    st.caption(
        "Data: NASA C-MAPSS | Reference: Saxena et al., PHM08\n\n"
        "Model: RandomForest Regressor | Features: EWMA + Second-Diff (State 2)"
    )