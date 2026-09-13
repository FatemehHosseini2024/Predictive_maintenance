from utils import compute_rolling_features, compute_trend_features


def add_feature_engineering(df_train, df_test, trend_sensors, windows):
    df_train = compute_rolling_features(df_train, windows=windows)
    df_train = compute_trend_features(df_train, trend_sensors, windows=windows)
    df_test = compute_rolling_features(df_test, windows=windows)
    df_test = compute_trend_features(df_test, trend_sensors, windows=windows)
    return df_train, df_test
