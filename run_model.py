import sys
sys.path.insert(0, '.')
from data_preprocessing import load_fd_data
from feature_engineering import add_feature_engineering, TREND_SENSORS_FD002
from utils import stratified_engine_split, clip_rul
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np

RUL_CLIP = 125

df_train, df_test, rul = load_fd_data('FD002')
df_train_fe, df_test_fe, scaler, sensor_cols = add_feature_engineering(df_train, df_test, dataset_name='FD002', n_conditions=6)

def compute_ewma_features(df, sensors, spans=[5, 20]):
    for span in spans:
        for col in sensors:
            if col in df.columns:
                df[f'{col}_ewma_{span}'] = (
                    df.groupby('unit_id')[col]
                    .transform(lambda s: s.ewm(span=span, min_periods=1).mean())
                )
    return df

df_train_fe = compute_ewma_features(df_train_fe, TREND_SENSORS_FD002, spans=[5, 20])
df_test_fe = compute_ewma_features(df_test_fe, TREND_SENSORS_FD002, spans=[5, 20])

df_train_final, df_val = stratified_engine_split(df_train_fe, test_size=0.2, n_bins=4, random_state=42)

exclude_cols = ['unit_id', 'RUL']
feature_cols = [c for c in df_train_final.columns if c not in exclude_cols]

X_train = df_train_final[feature_cols]
y_train = df_train_final['RUL']
y_train_clipped = clip_rul(y_train, RUL_CLIP)

X_val = df_val[feature_cols]
y_val = df_val['RUL']
y_val_clipped = clip_rul(y_val, RUL_CLIP)

model = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1)

print('Training Random Forest model...')
model.fit(X_train, y_train_clipped)

y_pred = model.predict(X_val)

rmse = np.sqrt(mean_squared_error(y_val_clipped, y_pred))
mae = mean_absolute_error(y_val_clipped, y_pred)

print(f'Number of features used: {len(feature_cols)}')
print(f'RUL Clip: {RUL_CLIP}')
print(f'Final Model - Validation RMSE: {rmse:.3f}')
print(f'Final Model - Validation MAE: {mae:.3f}')

# Per bin evaluation (5 bins)
n_bins = 5
bin_edges = np.linspace(0, RUL_CLIP, n_bins + 1)
y_val_binned = pd.cut(y_val_clipped, bins=bin_edges, labels=False, include_lowest=True)

print(f'\nPer-bin metrics (RUL bins of {int(RUL_CLIP/n_bins)}):')
print(f'{"Bin":<20} {"RMSE":<10} {"MAE":<10} {"Samples":<10}')

for bin_idx in sorted(y_val_binned.unique()):
    bin_mask = y_val_binned == bin_idx
    y_true_bin = y_val_clipped[bin_mask]
    y_pred_bin = y_pred[bin_mask]
    
    if len(y_true_bin) > 0:
        bin_rmse = np.sqrt(mean_squared_error(y_true_bin, y_pred_bin))
        bin_mae = mean_absolute_error(y_true_bin, y_pred_bin)
    else:
        bin_rmse = float('nan')
        bin_mae = float('nan')
    
    lower = int(bin_idx * RUL_CLIP / n_bins)
    upper = int((bin_idx + 1) * RUL_CLIP / n_bins)
    bin_label = f'[{lower}, {upper})'
    print(f'{bin_label:<20} {bin_rmse:<10.3f} {bin_mae:<10.3f} {len(y_true_bin):<10}')

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print('\nTop 20 Feature Importances:')
print(feature_importance.head(20).to_string(index=False))