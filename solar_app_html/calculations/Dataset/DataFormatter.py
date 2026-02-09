#%% PACKAGES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% _____________________ CONSUMPTION FORMAT _____________________
# REFORMATTING CONSUMPTION
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

# %%_____________________ CONSUMPTION PLOTS _____________________
# PLOTTING CONSUMPTION

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

# %% _____________________ PRIS FORMAT _____________________
# REFORMATTING PRICES

in_path = "Elspotprices.csv"
out_path = "elspot_DK1_clean_hourly.csv"

df = pd.read_csv(
    in_path,
    sep=";",        # IMPORTANT: your file is semicolon-separated
    decimal=",",    # IMPORTANT: decimal comma
)

# Strip column names
df.columns = df.columns.str.strip()

# Parse time columns
df["HourUTC"] = pd.to_datetime(df["HourUTC"], utc=True, errors="coerce")
df["HourDK"]  = pd.to_datetime(df["HourDK"], errors="coerce")

# Convert prices to numeric (decimal=',' already handled, but make sure)
df["SpotPriceDKK"] = pd.to_numeric(df["SpotPriceDKK"], errors="coerce")
df["SpotPriceEUR"] = pd.to_numeric(df["SpotPriceEUR"], errors="coerce")

# Drop bad rows and index by UTC time
df = df.dropna(subset=["HourUTC", "SpotPriceDKK"]).sort_values("HourUTC").set_index("HourUTC")

# Unit conversion: DKK/MWh -> DKK/kWh and øre/kWh
df["price_dkk_per_kwh"] = df["SpotPriceDKK"] / 1000.0
df["price_ore_per_kwh"] = df["price_dkk_per_kwh"] * 100.0

out = df[["HourDK", "PriceArea", "price_dkk_per_kwh", "price_ore_per_kwh", "SpotPriceDKK"]].copy()
# out.to_csv(out_path)

# print("Saved:", out_path)
# print(out.head())
# print("Price (øre/kWh) min/median/max:",
#       out["price_ore_per_kwh"].min(),
#       out["price_ore_per_kwh"].median(),
#       out["price_ore_per_kwh"].max())



#%% _____________________ PRIS DIAGNOSTIC PLOTS _____________________
# PRICES DIAGNOSTIC PLOTS

df = pd.read_csv("elspot_DK1_clean_hourly.csv", parse_dates=[0], index_col=0)

# 1) Time series (first 14 days)
sample = df.iloc[:24*30]
plt.figure(figsize=(12,4))
plt.plot(sample.index, sample["price_ore_per_kwh"])
plt.xlabel("Time (UTC)")
plt.ylabel("Spot price [øre/kWh]")
plt.title("Elspot DK1 — first 14 days (spot only)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 2) Histogram
plt.figure(figsize=(8,4))
plt.hist(df["price_ore_per_kwh"], bins=80)
plt.xlabel("Spot price [øre/kWh]")
plt.ylabel("Hours")
plt.title("Elspot DK1 — price distribution (spot only)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 3) Price duration curve
sorted_prices = np.sort(df["price_ore_per_kwh"].to_numpy())[::-1]
plt.figure(figsize=(10,4))
plt.plot(np.arange(len(sorted_prices)), sorted_prices)
plt.xlabel("Hour rank (highest to lowest)")
plt.ylabel("Spot price [øre/kWh]")
plt.title("Elspot DK1 — price duration curve (spot only)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 4) Average daily profile (DK clock)
tmp = df.copy()
tmp["HourDK"] = pd.to_datetime(tmp["HourDK"])
tmp["hour_dk"] = tmp["HourDK"].dt.hour
avg_profile = tmp.groupby("hour_dk")["price_ore_per_kwh"].mean()

plt.figure(figsize=(8,4))
plt.plot(avg_profile.index, avg_profile.values, marker="o")
plt.xlabel("Hour of day (DK clock)")
plt.ylabel("Average spot price [øre/kWh]")
plt.title("Elspot DK1 — average daily price profile (spot only)")
plt.xticks(range(24))
plt.grid(True)
plt.tight_layout()
plt.show()

#%% _____________________ TILFØJ TARIF ETC _____________________
# ADD TARRIFS TO PRICES

df = pd.read_csv(
    "elspot_DK1_clean_hourly.csv",
    parse_dates=[0],
    index_col=0
)

# ----- PARAMETERS (easy to change later) -----
grid_tariff   = 0.45   # DKK/kWh
system_tariff = 0.10
elafgift      = 0.90
supplier_fee  = 0.10
vat           = 0.25
# --------------------------------------------

addons = grid_tariff + system_tariff + elafgift + supplier_fee

df["buy_price_dkk_per_kwh"] = (df["price_dkk_per_kwh"] + addons) * (1 + vat)

# selling usually only gets spot
df["sell_price_dkk_per_kwh"] = df["price_dkk_per_kwh"]

# df.to_csv("elspot_DK1_with_total_prices.csv")

# print(df[["price_dkk_per_kwh", "buy_price_dkk_per_kwh"]].head())
print("Average buy price:", df["buy_price_dkk_per_kwh"].mean())

#%% _____________________ PRIS COMPARISON PLOTS _____________________
# PRICES COMPARISON PLOTS

df = pd.read_csv(
    "elspot_DK1_with_total_prices.csv",  # your cleaned file
    parse_dates=[0],
    index_col=0
)

def plot_price_days(days, column="buy_price_dkk_per_kwh", title=None):
    """
    days: ['YYYY-MM-DD', ...]
    column:
        'buy_price_dkk_per_kwh'   -> what household pays
        'price_dkk_per_kwh'       -> spot only
    """

    plt.figure(figsize=(10,5))

    for d in days:
        day = df.loc[d]

        plt.plot(
            day.index.hour,
            day[column],
            marker="o",
            label=d
        )

    plt.xlabel("Hour of day")
    plt.ylabel("Price [DKK/kWh]")
    plt.title(title or f"Hourly electricity price — {column}")
    plt.xticks(range(24))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ----- Examples -----

# what you actually pay
plot_price_days(
    ["2025-01-15", "2025-04-15", "2025-07-15"],
    column="buy_price_dkk_per_kwh",
    title="Total household electricity price"
)

# # spot only (for comparison)
# plot_price_days(
#     ["2025-01-15", "2025-04-15", "2025-07-15"],
#     column="price_dkk_per_kwh",
#     title="Spot price only"
# )

# %%
