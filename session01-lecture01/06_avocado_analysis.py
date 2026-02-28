import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

# --- INITIAL SETUP ---
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
sns.set_palette("husl")

# Standard plot settings
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 11

# --- DATA LOADING & CLEANING ---
# We load and clean the data in a single pipeline for clarity and performance
avocado = (
    pd.read_csv(
        "https://raw.githubusercontent.com/kostis-christodoulou/e628/main/data/avocado.csv"
    )
    .assign(
        date=lambda d: pd.to_datetime(d["date"], dayfirst=True),
        type=lambda d: d["type"].astype("category"),
        region=lambda d: d["region"].astype("category"),
        revenue=lambda d: d["total_volume"] * d["average_price"],
        month=lambda d: d["date"].dt.month,
        year=lambda d: d["date"].dt.year,
    )
    # Filter out aggregate regions
    .query('region != "TotalUS"')
)

# Helper for month mapping
month_names = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]
avocado["month_name"] = (
    avocado["month"].map(lambda x: month_names[x - 1]).astype("category")
)

# --- EXPLORATION ---
print("\n--- Dataset Summary ---")
avocado.info()

print("\n--- Highest Volume Regions (Top 5) ---")
top_vol = (
    avocado.groupby(["region", "type"])["total_volume"]
    .sum()
    .sort_values(ascending=False)
    .head(5)
)
print(top_vol)

print("\n--- Highest Revenue Regions (Top 5) ---")
top_rev = (
    avocado.groupby(["region", "type"])["revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(5)
)
print(top_rev)

# --- SEASONAL PATTERNS ---
# Aggregate by month and type
monthly_agg = (
    avocado.groupby(["month", "type", "month_name"])
    .agg({"total_volume": "sum", "average_price": "mean"})
    .reset_index()
    .sort_values(["month", "type"])
)

# Reshape for multi-metric plotting
melted_monthly = monthly_agg.melt(
    id_vars=["type", "month", "month_name"],
    value_vars=["total_volume", "average_price"],
    var_name="metric",
    value_name="value",
)

# Visualize seasonal trends
g = sns.FacetGrid(
    melted_monthly,
    row="metric",
    col="type",
    hue="metric",
    sharey=False,
    height=3.5,
    aspect=2,
)
g.map_dataframe(sns.lineplot, x="month_name", y="value", marker="o")
g.set_titles(template="{row_name} | {col_name}")
g.fig.suptitle("Monthly Volume and Price Trends", y=1.02, fontsize=16)
plt.show()

# --- ORGANIC VS CONVENTIONAL ---
# Compare average prices
avg_prices = avocado.groupby("type", observed=True)["average_price"].mean()
premium = (avg_prices["organic"] / avg_prices["conventional"] - 1) * 100
print(f"\nPrice Premium: Organic costs {premium:.1f}% more on average.")

# Yearly growth by type
yearly_stats = (
    avocado.groupby(["year", "type"], observed=True)
    .agg({"total_volume": "sum", "average_price": "mean"})
    .assign(vol_bil=lambda d: d["total_volume"] / 1e9)
    .reset_index()
)

# Reshape for comparison plot
yearly_long = yearly_stats.melt(
    id_vars=["year", "type"],
    value_vars=["vol_bil", "average_price"],
    var_name="metric",
    value_name="value",
)

# Plot multi-year trends
sns.relplot(
    data=yearly_long,
    x="year",
    y="value",
    col="metric",
    hue="type",
    kind="line",
    facet_kws={"sharey": False},
    palette=["brown", "green"],
    linewidth=3,
    marker="o",
)
plt.show()

# --- REGIONAL PRICE VARIATIONS ---
# Identify most expensive markets
top_15_regions = (
    avocado.groupby("region", observed=True)["average_price"]
    .mean()
    .sort_values(ascending=False)
    .head(15)
    .reset_index()
)

plt.figure(figsize=(10, 8))
sns.barplot(data=top_15_regions, x="average_price", y="region", palette="Reds_r")
plt.title("Top 15 Most Expensive Avocado Regions", fontsize=15)
plt.xlabel("Average Price ($)")
plt.ylabel("")
plt.tight_layout()
plt.show()

# --- VOLUME BREAKDOWN BY SKU ---
sku_cols = ["plu4046", "plu4225", "plu4770", "total_bags"]
sku_data = (
    avocado[sku_cols].sum().reset_index().rename(columns={"index": "sku", 0: "volume"})
)

sns.barplot(data=sku_data, x="sku", y="volume", palette="viridis")
plt.title("Volume Breakdown by SKU / Bag Size")
plt.show()

# --- PRICE-VOLUME RELATIONSHIP ---
# Check correlation between quantity and price (elasticity)
sns.lmplot(
    data=avocado.assign(vol_mil=lambda d: d["total_volume"] / 1e6),
    x="vol_mil",
    y="average_price",
    col="type",
    hue="type",
    palette=["brown", "green"],
    facet_kws={"sharex": False, "sharey": False},
    scatter_kws={"alpha": 0.2},
    line_kws={"color": "red"},
)
plt.show()

# --- YEAR-OVER-YEAR GROWTH ---
growth_metrics = (
    yearly_stats.sort_values(["type", "year"])
    .groupby("type")
    .apply(
        lambda x: x.assign(
            vol_growth=x["total_volume"].pct_change() * 100,
            price_growth=x["average_price"].pct_change() * 100,
        )
    )
    .dropna()
    .melt(
        id_vars=["year", "type"],
        value_vars=["vol_growth", "price_growth"],
        var_name="metric",
        value_name="pct_change",
    )
)

g_growth = sns.relplot(
    data=growth_metrics,
    x="year",
    y="pct_change",
    hue="type",
    col="metric",
    kind="line",
    marker="o",
    facet_kws={"sharey": False},
)
g_growth.map(plt.axhline, y=0, color="gray", linestyle="--", alpha=0.5)
plt.show()

# --- CORRELATION HEATMAP ---
numeric_cols = avocado.select_dtypes(include=[np.number]).columns
corr_matrix = avocado[numeric_cols].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.title("Correlation Matrix of Numeric Features")
plt.show()
