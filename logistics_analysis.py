"""
Week 1 Task: Data Cleaning and EDA on a Simulated Logistics Dataset
Run: python logistics_analysis.py
Needs: pandas numpy matplotlib seaborn
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
sns.set_theme(style="whitegrid")
log = {}  # collects numbers for the report

# ---------------------------------------------------------------
# STEP 1: SIMULATE A DIRTY LOGISTICS DATASET
# ---------------------------------------------------------------
n = 5000
cities = ["Mumbai", "Delhi", "Bengaluru", "Chennai", "Kolkata", "Hyderabad", "Pune", "Bhubaneswar"]
carriers = ["BlueDart", "Delhivery", "DTDC", "Ecom Express", "XpressBees"]
vehicles = ["Bike", "Van", "Truck", "Container"]
routes = [f"R{str(i).zfill(2)}" for i in range(1, 21)]

route_info = {}
for r in routes:
    o, d = np.random.choice(cities, 2, replace=False)
    route_info[r] = (o, d, np.random.randint(150, 2200))

ship_dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(np.random.randint(0, 365, n), unit="D")
route = np.random.choice(routes, n)
origin = [route_info[r][0] for r in route]
dest = [route_info[r][1] for r in route]
dist = np.array([route_info[r][2] for r in route]) + np.random.normal(0, 15, n)
vehicle = np.random.choice(vehicles, n, p=[0.15, 0.35, 0.35, 0.15])
vmult = {"Bike": 0.8, "Van": 1.0, "Truck": 1.4, "Container": 2.0}
weight = np.round(np.random.gamma(3, 120, n) * np.array([vmult[v] for v in vehicle]), 1)
dtime = np.round(dist / np.random.uniform(35, 55, n) + np.random.normal(6, 2, n), 1)
cost = np.round(dist * np.array([vmult[v] for v in vehicle]) * 2.2 + weight * 0.9 + np.random.normal(0, 150, n), 2)
status = np.random.choice(["Delivered", "In Transit", "Delayed", "Cancelled"], n, p=[0.72, 0.1, 0.14, 0.04])
carrier = np.random.choice(carriers, n)

df = pd.DataFrame({
    "shipment_id": [f"SH{100000 + i}" for i in range(n)],
    "shipment_date": ship_dates,
    "delivery_date": ship_dates + pd.to_timedelta(np.ceil(dtime / 24), unit="D"),
    "route_code": route, "origin": origin, "destination": dest,
    "carrier": carrier, "vehicle_type": vehicle,
    "distance_km": np.round(dist, 1), "weight_kg": weight,
    "delivery_time_hrs": dtime, "shipping_cost": cost, "status": status,
})

# --- inject data quality problems ---
df["shipment_date"] = df["shipment_date"].dt.strftime("%Y-%m-%d")
df["delivery_date"] = df["delivery_date"].dt.strftime("%Y-%m-%d")
idx = np.random.choice(n, 400, replace=False)                       # mixed date formats
df.loc[idx, "shipment_date"] = pd.to_datetime(df.loc[idx, "shipment_date"]).dt.strftime("%d/%m/%Y")
for col, k in [("weight_kg", 250), ("delivery_time_hrs", 300), ("shipping_cost", 200),
               ("carrier", 150), ("vehicle_type", 120), ("distance_km", 100)]:
    df.loc[np.random.choice(n, k, replace=False), col] = np.nan     # missing values
for col in ["carrier", "status", "origin", "destination"]:          # messy text
    i = np.random.choice(n, 500, replace=False)
    df.loc[i, col] = df.loc[i, col].astype(str).str.upper() + " "
    j = np.random.choice(n, 300, replace=False)
    df.loc[j, col] = df.loc[j, col].astype(str).str.lower()
df.loc[np.random.choice(n, 40, replace=False), "shipping_cost"] = df["shipping_cost"].max() * 8   # outliers
df.loc[np.random.choice(n, 40, replace=False), "delivery_time_hrs"] = 400                         # outliers
df.loc[np.random.choice(n, 25, replace=False), "weight_kg"] = -50                                 # invalid
df.loc[np.random.choice(n, 15, replace=False), "distance_km"] = -100                              # invalid
bad = np.random.choice(n, 30, replace=False)                                                       # date logic error
df.loc[bad, "delivery_date"] = "2024-12-01"
df = pd.concat([df, df.sample(120, random_state=1)], ignore_index=True)                           # duplicates
df.to_csv("logistics_raw.csv", index=False)

# ---------------------------------------------------------------
# STEP 2: INITIAL INSPECTION
# ---------------------------------------------------------------
log["raw_shape"] = list(df.shape)
log["missing_before"] = df.isna().sum()[lambda s: s > 0].to_dict()
log["duplicates"] = int(df.duplicated().sum())
log["neg_weight"] = int((df["weight_kg"] < 0).sum())
log["neg_distance"] = int((df["distance_km"] < 0).sum())
log["raw_carrier_values"] = int(df["carrier"].nunique())
log["raw_status_values"] = int(df["status"].nunique())

# ---------------------------------------------------------------
# STEP 3: DATA CLEANING
# ---------------------------------------------------------------
# 3.1 remove duplicates
df = df.drop_duplicates().reset_index(drop=True)

# 3.2 clean text columns
fmt = {"origin": str.title, "destination": str.title, "carrier": str.title, "status": str.title}
for c in ["origin", "destination", "carrier", "status", "vehicle_type"]:
    df[c] = df[c].astype("string").str.strip().str.title()
df["carrier"] = df["carrier"].replace({"Bluedart": "BlueDart", "Xpressbees": "XpressBees", "Dtdc": "DTDC"})
log["clean_carrier_values"] = int(df["carrier"].nunique())
log["clean_status_values"] = int(df["status"].nunique())

# 3.3 fix dates
sd = df["shipment_date"].astype(str)
iso = sd.str.match(r"^\d{4}-")          # YYYY-MM-DD  vs  DD/MM/YYYY
df["shipment_date"] = pd.NaT
df.loc[iso, "shipment_date"] = pd.to_datetime(sd[iso], format="%Y-%m-%d", errors="coerce")
df.loc[~iso, "shipment_date"] = pd.to_datetime(sd[~iso], format="%d/%m/%Y", errors="coerce")
df["shipment_date"] = pd.to_datetime(df["shipment_date"])
log["mixed_date_formats_fixed"] = int((~iso).sum())
df["delivery_date"] = pd.to_datetime(df["delivery_date"], errors="coerce")
bad_dates = df["delivery_date"] < df["shipment_date"]
log["bad_dates"] = int(bad_dates.sum())
df.loc[bad_dates, "delivery_date"] = pd.NaT

# 3.4 invalid values -> missing
df.loc[df["weight_kg"] <= 0, "weight_kg"] = np.nan
df.loc[df["distance_km"] <= 0, "distance_km"] = np.nan

# 3.5 fill missing values
df["distance_km"] = df["distance_km"].fillna(df.groupby("route_code")["distance_km"].transform("median"))
for c in ["origin", "destination"]:
    df[c] = df[c].fillna(df.groupby("route_code")[c].transform(lambda s: s.mode()[0]))
df["carrier"] = df["carrier"].fillna("Unknown")
df["vehicle_type"] = df["vehicle_type"].fillna(df["vehicle_type"].mode()[0])
for c in ["weight_kg", "delivery_time_hrs", "shipping_cost"]:
    df[c] = df[c].fillna(df.groupby("vehicle_type")[c].transform("median"))
df["delivery_date"] = df["delivery_date"].fillna(
    df["shipment_date"] + pd.to_timedelta(np.ceil(df["delivery_time_hrs"] / 24), unit="D"))

# 3.6 outliers: IQR method, cap values
def cap_iqr(s):
    q1, q3 = s.quantile([0.25, 0.75])
    lo, hi = max(q1 - 1.5 * (q3 - q1), 0), q3 + 1.5 * (q3 - q1)
    return int(((s < lo) | (s > hi)).sum()), s.clip(lo, hi), (round(lo, 1), round(hi, 1))

log["outliers"] = {}
before = df[["shipping_cost", "delivery_time_hrs"]].copy()
for c in ["shipping_cost", "delivery_time_hrs", "weight_kg"]:
    cnt, df[c], bounds = cap_iqr(df[c])
    log["outliers"][c] = {"count": cnt, "bounds": bounds}

# boxplots before / after
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
sns.boxplot(y=before["shipping_cost"], ax=ax[0], color="#e8590c"); ax[0].set_title("Shipping Cost - Before")
sns.boxplot(y=df["shipping_cost"], ax=ax[1], color="#2f9e44"); ax[1].set_title("Shipping Cost - After")
plt.tight_layout(); plt.savefig("fig_outliers.png", dpi=120); plt.close()

# 3.7 new columns
df["transit_days"] = (df["delivery_date"] - df["shipment_date"]).dt.days
df["cost_per_km"] = (df["shipping_cost"] / df["distance_km"]).round(2)
df["month"] = df["shipment_date"].dt.to_period("M").astype(str)
df["is_delayed"] = (df["status"] == "Delayed").astype(int)

log["clean_shape"] = list(df.shape)
log["missing_after"] = int(df.isna().sum().sum())
df.to_csv("logistics_clean.csv", index=False)

# ---------------------------------------------------------------
# STEP 4: EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------------------
num = ["distance_km", "weight_kg", "delivery_time_hrs", "shipping_cost", "cost_per_km"]
desc = df[num].describe().T.round(2)
log["describe"] = desc.reset_index().values.tolist()
log["status_pct"] = (df["status"].value_counts(normalize=True) * 100).round(1).to_dict()
log["carrier_delay"] = (df.groupby("carrier")["is_delayed"].mean() * 100).round(1).sort_values(ascending=False).to_dict()
log["vehicle_cost"] = df.groupby("vehicle_type")["shipping_cost"].mean().round(0).sort_values().to_dict()
log["route_time"] = df.groupby("route_code")["delivery_time_hrs"].mean().round(1).sort_values(ascending=False).head(3).to_dict()
log["corr_dist_cost"] = round(df["distance_km"].corr(df["shipping_cost"]), 2)
log["corr_dist_time"] = round(df["distance_km"].corr(df["delivery_time_hrs"]), 2)
log["corr_weight_cost"] = round(df["weight_kg"].corr(df["shipping_cost"]), 2)
monthly = df.groupby("month").size()
log["peak_month"] = [monthly.idxmax(), int(monthly.max())]
log["low_month"] = [monthly.idxmin(), int(monthly.min())]

# charts
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for a, c, col in zip(ax, ["shipping_cost", "delivery_time_hrs", "weight_kg"], ["#1971c2", "#e8590c", "#2f9e44"]):
    sns.histplot(df[c], bins=30, kde=True, ax=a, color=col); a.set_title(f"Distribution of {c}")
plt.tight_layout(); plt.savefig("fig_hist.png", dpi=120); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
sns.scatterplot(data=df.sample(1500, random_state=1), x="distance_km", y="shipping_cost", hue="vehicle_type", alpha=.6, ax=ax[0])
ax[0].set_title("Distance vs Shipping Cost")
sns.scatterplot(data=df.sample(1500, random_state=1), x="distance_km", y="delivery_time_hrs", alpha=.5, color="#7048e8", ax=ax[1])
ax[1].set_title("Distance vs Delivery Time")
plt.tight_layout(); plt.savefig("fig_scatter.png", dpi=120); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
sns.boxplot(data=df, x="vehicle_type", y="shipping_cost", ax=ax[0], order=["Bike", "Van", "Truck", "Container"])
ax[0].set_title("Shipping Cost by Vehicle Type")
(df.groupby("carrier")["is_delayed"].mean() * 100).sort_values().plot(kind="barh", ax=ax[1], color="#e03131")
ax[1].set_title("Delay Rate (%) by Carrier"); ax[1].set_xlabel("Delay %")
plt.tight_layout(); plt.savefig("fig_groups.png", dpi=120); plt.close()

plt.figure(figsize=(9, 4))
monthly.plot(marker="o", color="#1971c2"); plt.title("Monthly Shipment Volume"); plt.xticks(rotation=45)
plt.tight_layout(); plt.savefig("fig_monthly.png", dpi=120); plt.close()

plt.figure(figsize=(6.5, 5))
sns.heatmap(df[num].corr(), annot=True, cmap="coolwarm", fmt=".2f"); plt.title("Correlation Heatmap")
plt.tight_layout(); plt.savefig("fig_corr.png", dpi=120); plt.close()

json.dump(log, open("results.json", "w"), indent=2, default=str)
print(json.dumps(log, indent=2, default=str))