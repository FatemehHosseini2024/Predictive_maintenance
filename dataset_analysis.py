import pandas as pd
import matplotlib.pyplot as plt

# خواندن دیتاست
df = pd.read_csv(
    "train_FD001.txt",
    sep=r"\s+",
    header=None
)
df_test =  pd.read_csv(
    "test_FD001.txt",
    sep=r"\s+",
    header=None
)
columns = [
    "unit_id",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    *[f"sensor_{i}" for i in range(1, 22)]
]

df.columns = columns

df_test.columns=columns
# نمایش نوع داده هر ستون
print(df.dtypes)
print(df.shape)
print(df_test.dtypes)
print(df_test.shape)
print(df.info())
print(df_test.info())
cycle_counts = df.groupby("unit_id")["cycle"].count()

print(f"cycle counts for train :{cycle_counts}")
print(df.isnull().sum())
duplicates = df[df.duplicated()]
print(f"duplicates :{duplicates}")
constant_features = df.columns[df.nunique() == 1]

print("Constant Features:")
print(constant_features)
print(f"cycle counts statistics : {cycle_counts.describe()}")


#plt.hist(cycle_counts, bins=10)

#plt.xlabel("Number of Cycles")
#plt.ylabel("Number of Engines")
#plt.title("Distribution of Engine Lifetimes")

#plt.show()

sensor_cols = [f"sensor_{i}" for i in range(1, 22)]

engine = df[df["unit_id"] == 59]

for sensor in sensor_cols:
    #plt.figure(figsize=(10, 4))
    
    #plt.plot(engine["cycle"], engine[sensor])
    
    plt.xlabel("Cycle")
    plt.ylabel(sensor)
    plt.title(f"{sensor} Behavior Over Engine Lifetime")
    
    #plt.show()

engines = [1, 30, 20, 70, 100]

for sensor in sensor_cols:
    plt.figure(figsize=(10, 5))

    for engine_id in engines:
        engine = df[df["unit_id"] == engine_id]
        plt.plot(
            engine["cycle"],
            engine[sensor],
            label=f"Engine {engine_id}"
        )

    plt.xlabel("Cycle")
    plt.ylabel(sensor)
    plt.title(f"{sensor} Across Multiple Engines")
    plt.legend()
    plt.show()
    


