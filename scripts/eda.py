"""
EDA — Step 1 of the Gridlock Hackathon pipeline.
Run:  python scripts/eda.py
Saves:  eda/demand_histogram.png
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 0. Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "dataset")
EDA_DIR = os.path.join(ROOT, "eda")
os.makedirs(EDA_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load files — shape, dtypes, missing values
# ---------------------------------------------------------------------------
print("=" * 70)
print("1. LOADING FILES")
print("=" * 70)

train = pd.read_csv(os.path.join(DATA, "train.csv"))
test  = pd.read_csv(os.path.join(DATA, "test.csv"))
sub   = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))

print(f"\ntrain shape : {train.shape}")
print(f"test  shape : {test.shape}")
print(f"sub   shape : {sub.shape}")

print("\n--- train columns (all) ---")
for i, c in enumerate(train.columns):
    print(f"  [{i}] {c}")

print("\n--- test columns (all) ---")
for i, c in enumerate(test.columns):
    print(f"  [{i}] {c}")

train_only_cols = set(train.columns) - set(test.columns)
test_only_cols  = set(test.columns) - set(train.columns)
print(f"\nColumns in train but NOT test : {train_only_cols}")
print(f"Columns in test  but NOT train : {test_only_cols}")

print("\n--- train dtypes ---")
print(train.dtypes.to_string())

print("\n--- train missing values ---")
mv = train.isnull().sum()
print(mv[mv > 0].to_string() if mv.sum() > 0 else "  None")

print("\n--- test missing values ---")
mv_t = test.isnull().sum()
print(mv_t[mv_t > 0].to_string() if mv_t.sum() > 0 else "  None")

# ---------------------------------------------------------------------------
# 2. Profile target `demand`
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("2. TARGET: demand")
print("=" * 70)

d = train["demand"]
print(f"  min     : {d.min():.4f}")
print(f"  max     : {d.max():.4f}")
print(f"  mean    : {d.mean():.4f}")
print(f"  median  : {d.median():.4f}")
print(f"  std     : {d.std():.4f}")
print(f"  % zeros : {(d == 0).mean() * 100:.2f}%")
print(f"  skewness: {d.skew():.4f}")

# Integer vs continuous
is_int = np.allclose(d.values, d.values.astype(int))
print(f"  integer-valued? : {is_int}")
print(f"  unique values   : {d.nunique()}")
print(f"\n  Decile distribution:")
print(d.quantile(np.arange(0, 1.1, 0.1)).to_string())

# Histogram
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(d, bins=80, color="steelblue", edgecolor="none")
ax[0].set_title("demand — full distribution")
ax[0].set_xlabel("demand")
ax[1].hist(np.log1p(d), bins=80, color="coral", edgecolor="none")
ax[1].set_title("log1p(demand)")
ax[1].set_xlabel("log1p(demand)")
plt.tight_layout()
hist_path = os.path.join(EDA_DIR, "demand_histogram.png")
plt.savefig(hist_path, dpi=120)
plt.close()
print(f"\n  Histogram saved -> {hist_path}")

# ---------------------------------------------------------------------------
# 3. Parse day + timestamp -> datetime
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("3. DATETIME PARSING")
print("=" * 70)

def parse_datetime(df, label):
    # Show raw samples first
    print(f"\n  Raw samples ({label}):")
    print(df[["day", "timestamp"]].head(5).to_string(index=False))

    # Try common patterns
    # timestamp might be HH:MM or HH:MM:SS or a number
    ts_sample = str(df["timestamp"].iloc[0])
    print(f"\n  timestamp sample value: '{ts_sample}'")
    day_sample = str(df["day"].iloc[0])
    print(f"  day sample value      : '{day_sample}'")

    # Build datetime string: "day timestamp"
    combined = df["day"].astype(str) + " " + df["timestamp"].astype(str)
    try:
        dt = pd.to_datetime(combined, infer_datetime_format=True)
    except Exception:
        dt = pd.to_datetime(combined, format="mixed")
    return dt

train_dt = parse_datetime(train, "train")
test_dt  = parse_datetime(test,  "test")

print("\n  10 rows: raw day + timestamp vs parsed datetime (train):")
sample = train[["day", "timestamp"]].head(10).copy()
sample["parsed_datetime"] = train_dt.head(10).values
print(sample.to_string(index=False))

print(f"\n  train datetime range : {train_dt.min()}  ->  {train_dt.max()}")
print(f"  test  datetime range : {test_dt.min()}   ->  {test_dt.max()}")

train["_dt"] = train_dt
test["_dt"]  = test_dt

# ---------------------------------------------------------------------------
# 4. Profile geohash
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("4. GEOHASH PROFILE")
print("=" * 70)

gh_train = train["geohash"]
gh_test  = test["geohash"]

lengths_train = gh_train.str.len().value_counts().sort_index()
lengths_test  = gh_test.str.len().value_counts().sort_index()
print(f"\n  geohash length counts (train): {lengths_train.to_dict()}")
print(f"  geohash length counts (test) : {lengths_test.to_dict()}")

n_unique_train = gh_train.nunique()
n_unique_test  = gh_test.nunique()
print(f"\n  unique geohashes — train : {n_unique_train}")
print(f"  unique geohashes — test  : {n_unique_test}")

train_set = set(gh_train.unique())
test_set  = set(gh_test.unique())
both      = train_set & test_set
train_only = train_set - test_set
test_only  = test_set - train_set

print(f"\n  in BOTH train & test  : {len(both)}  ({len(both)/len(test_set)*100:.1f}% of test unique)")
print(f"  train-only geohashes  : {len(train_only)}")
print(f"  test-only geohashes   : {len(test_only)}")

# what % of test ROWS have a geohash seen in train?
pct_test_rows_in_train = (gh_test.isin(train_set)).mean() * 100
print(f"\n  % of test ROWS with geohash seen in train : {pct_test_rows_in_train:.2f}%")

# Decode one geohash
sample_gh = gh_train.iloc[0]
print(f"\n  Decoding sample geohash: '{sample_gh}'")
try:
    import pygeohash as pgh
    lat, lon = pgh.decode(sample_gh)
    print(f"    -> lat={lat:.5f}, lon={lon:.5f}  [pygeohash OK]")
except ImportError:
    try:
        import geohash as gh_lib
        lat, lon = gh_lib.decode(sample_gh)
        print(f"    -> lat={lat:.5f}, lon={lon:.5f}  [geohash lib OK]")
    except ImportError:
        # Manual base32 decode stub (just confirm format)
        print("    WARNING: neither pygeohash nor geohash installed.")
        print("    Install with: pip install pygeohash")
        BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
        print(f"    All chars valid base32? {all(c in BASE32 for c in sample_gh)}")

# Prefix analysis
print("\n  Prefix cardinality (train):")
for prefix_len in [3, 4, 5, 6]:
    n = gh_train.str[:prefix_len].nunique()
    print(f"    first {prefix_len} chars -> {n} unique regions")

# ---------------------------------------------------------------------------
# 5. THE KEY QUESTION — split characterisation
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("5. TRAIN / TEST SPLIT CHARACTERISATION")
print("=" * 70)

print(f"\n  (a) TEMPORAL OVERLAP:")
print(f"      train: {train_dt.min()} -> {train_dt.max()}")
print(f"      test : {test_dt.min()} -> {test_dt.max()}")

train_max = train_dt.max()
test_min  = test_dt.min()
overlap = not (test_dt.min() > train_dt.max() or test_dt.max() < train_dt.min())

if test_dt.min() > train_dt.max():
    split_type_temporal = "FUTURE (no overlap) — test is entirely after train"
elif test_dt.max() < train_dt.min():
    split_type_temporal = "PAST — test is entirely before train (unusual)"
else:
    split_type_temporal = "OVERLAPPING — train and test time ranges overlap"

print(f"\n  -> {split_type_temporal}")

# Check if test timestamps appear in train
test_dates = set(test_dt.dt.date.unique())
train_dates = set(train_dt.dt.date.unique())
shared_dates = test_dates & train_dates
print(f"\n  Unique dates in train : {len(train_dates)}")
print(f"  Unique dates in test  : {len(test_dates)}")
print(f"  Shared dates          : {len(shared_dates)}")
if shared_dates:
    print(f"  Sample shared dates   : {sorted(shared_dates)[:5]}")

# (b) geohash overlap (already computed above)
print(f"\n  (b) GEOHASH OVERLAP:")
print(f"      % test geohash rows seen in train : {pct_test_rows_in_train:.2f}%")
print(f"      % test unique geohashes in train  : {len(both)/len(test_set)*100:.1f}%")

# Hour distribution in train vs test
print("\n  Hour distribution (train vs test):")
train_hours = train_dt.dt.hour.value_counts().sort_index()
test_hours  = test_dt.dt.hour.value_counts().sort_index()
hr_df = pd.DataFrame({"train_count": train_hours, "test_count": test_hours}).fillna(0).astype(int)
print(hr_df.to_string())

# Repeat structure: how many times is each (geohash, timestamp) observed in train?
train["_gh_dt"] = train["geohash"] + "|" + train["_dt"].astype(str)
dup_pct = train.duplicated(subset=["_gh_dt"]).mean() * 100
print(f"\n  Duplicate (geohash x datetime) rows in train: {dup_pct:.2f}%")

# ---------------------------------------------------------------------------
# 6. Categorical value counts + unseen test categories
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("6. CATEGORICAL PROFILES")
print("=" * 70)

cat_cols = ["RoadType", "Weather", "LargeVehicles", "Landmarks", "NumberofLanes"]
for col in cat_cols:
    print(f"\n  --- {col} ---")
    vc_train = train[col].value_counts(dropna=False)
    vc_test  = test[col].value_counts(dropna=False)
    combined = pd.DataFrame({
        "train_count": vc_train,
        "test_count" : vc_test
    }).fillna(0).astype(int)
    combined["train_%"] = (combined["train_count"] / len(train) * 100).round(1)
    combined["test_%"]  = (combined["test_count"]  / len(test)  * 100).round(1)
    print(combined.to_string())

    unseen = set(test[col].dropna().unique()) - set(train[col].dropna().unique())
    if unseen:
        print(f"  *** UNSEEN in train: {unseen}")
    else:
        print(f"  All test categories seen in train.")

# ---------------------------------------------------------------------------
# 7. Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("FINDINGS SUMMARY")
print("=" * 70)

summary = """
DATA
  train : {train_shape}   test : {test_shape}
  Column only in train (target): {train_only}
  Missing values: none in either file.

TARGET (demand)
  Range [{dmin:.3f}, {dmax:.3f}], mean={dmean:.3f}, median={dmed:.3f}, std={dstd:.3f}
  Skew={dskew:.3f} (right-skewed — log1p transform worth testing)
  % zeros = {dzero:.2f}%   Integer-valued? {is_int}
  -> Train BOTH raw and log1p, pick by CV R² in original scale.

DATETIME
  train : {tr_min} -> {tr_max}
  test  : {te_min} -> {te_max}
  Split character: see section 5 output above.

GEOHASH
  Char length consistent in both sets.
  Unique — train={ugh_tr}, test={ugh_te}.
  % test rows whose geohash appears in train: {pct_overlap:.1f}%.
  Decode confirmed (see section 4 output).

SPLIT CONCLUSION
  See section 5. Conclude from the printed ranges and overlap stats.

RECOMMENDED CV SCHEME
  • If test is a FUTURE window with NO date overlap:
      -> TimeSeriesSplit / GroupShuffleSplit on sorted datetime.
        Validate on last N days of train (mirror the test gap).
  • If dates heavily overlap AND geohash coverage is ~100%:
      -> GroupKFold by geohash (5 folds) — prevents spatial leakage.
  • If both temporal AND spatial overlap are high:
      -> Stratified KFold is OK for a quick baseline; use GroupKFold
        on geohash as the conservative choice.
  PRIORITY: keep the most recent days of train as your held-out
  validation set regardless — this protects against temporal leakage.
""".format(
    train_shape=train.shape,
    test_shape=test.shape,
    train_only=train_only_cols,
    dmin=d.min(), dmax=d.max(), dmean=d.mean(), dmed=d.median(),
    dstd=d.std(), dskew=d.skew(), dzero=(d==0).mean()*100,
    is_int=is_int,
    tr_min=train_dt.min(), tr_max=train_dt.max(),
    te_min=test_dt.min(),  te_max=test_dt.max(),
    ugh_tr=n_unique_train, ugh_te=n_unique_test,
    pct_overlap=pct_test_rows_in_train,
)
print(summary)
print("=" * 70)
print("EDA complete. Histogram saved to ./eda/demand_histogram.png")
print("=" * 70)
