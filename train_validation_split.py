from data_preprocessing import df_train
from utils import stratified_engine_split

# تقسیم داده‌های train به train_final و validation
df_train_final, df_val = stratified_engine_split(
    df_train,
    test_size=0.2,
    n_bins=4,
    random_state=42,
)
