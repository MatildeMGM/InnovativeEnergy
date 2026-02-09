#%%
import pandas as pd

in_path = "ConsumptionConsumerCategoryHour.csv"
out_path = "consumption_private_scaled_hourly.csv"

# ---- Set your target annual household consumption here (kWh/year) ----
TARGET_ANNUAL_KWH = 4200.0

# Read the downloaded dataset (Danish CSV formatting)
df = pd.read_csv(
    in_path,
    sep=";",
    decimal=",",
)

# Parse timestamps
df["TimeUTC"] = pd.to_datetime(df["TimeUTC"], utc=True)
df["TimeDK"] = pd.to_datetime(df["TimeDK"])  # local clock time as provided

# Sort ascending and set index (use TimeUTC for unambiguous alignment)
df = df.sort_values("TimeUTC").set_index("TimeUTC")

# Ensure numeric
df["ConsumptionkWh"] = pd.to_numeric(df["ConsumptionkWh"], errors="coerce")
df = df.dropna(subset=["ConsumptionkWh"])

# Annual target scaling
annual_sum = df["ConsumptionkWh"].sum()
scale = TARGET_ANNUAL_KWH / annual_sum

out = pd.DataFrame(index=df.index)
out["load_kwh"] = df["ConsumptionkWh"] * scale

# Optional: keep metadata and TimeDK for convenience
for col in ["TimeDK", "RegionName", "ConsumerCategory3", "ConsumerCategory2"]:
    if col in df.columns:
        out[col] = df[col].values

# Save clean CSV (comma-delimited, dot decimal)

# out.to_csv(out_path, index=True)

# print(f"Annual sum (before scaling): {annual_sum:.3f} kWh")
# print(f"Scale factor: {scale:.12e}")
# print(f"Saved: {out_path}")
# print(out.head())

# %%
import matplotlib.pyplot as plt

# load once
df = pd.read_csv(
    "consumption_private_scaled_hourly.csv",
    parse_dates=[0],
    index_col=0
)


def plot_days(days):
    """
    days: list of strings ['YYYY-MM-DD', ...]
    """
    plt.figure(figsize=(10,5))

    for d in days:
        day = df.loc[d]
        plt.plot(
            day.index.hour,
            day["load_kwh"],
            marker="o",
            label=d
        )

    plt.xlabel("Hour of day")
    plt.ylabel("Consumption [kWh]")
    plt.title("Hourly household consumption comparison")
    plt.xticks(range(24))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# example usage
plot_days([
    "2025-01-15",
    "2025-04-15",
    "2025-07-15"
])

# %%
