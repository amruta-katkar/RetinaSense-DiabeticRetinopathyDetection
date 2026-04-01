"""
Dataset Splitting — split_dataset.py
======================================
Corrected version of your original splitting code.

Changes from your original:
  1. Stratifies by (source × grade) combined key — not just grade
  2. Creates a proper test set (was missing before)
  3. Uses CSV-only splits for train/val/test (no file copying for those)
     Only frontend_demo physically copies files (as you had before)
  4. Saves split CSVs to drDataset/ folder

Final split proportions (of total 19,729 images):
  frontend_demo :  ~2%   (≈394  images) — for UI demo, physically copied
  train         : ~70%   (≈13,700 images)
  val           : ~14%   (≈2,760  images)
  test          : ~14%   (≈2,760  images) ← was missing before

Your folder structure after running this:
  D:/RetinaSense/
  ├── DR_Master_Dataset/
  │   ├── images/              ← all 19,729 images (flat, untouched)
  │   └── master_metadata.csv
  └── drDataset/
      ├── frontend_demo/       ← physically copied ~394 images
      ├── train_metadata.csv   ← CSV only (no copy needed)
      ├── val_metadata.csv     ← CSV only
      ├── test_metadata.csv    ← CSV only  ← NEW
      └── frontend_demo_metadata.csv
"""

import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split
from collections import Counter


# ─── Paths — adjust if needed ────────────────────────────────────────────────

MASTER_CSV   = "D:/RetinaSense/DR_Master_Dataset/master_metadata.csv"
IMAGE_ROOT   = "D:/RetinaSense/DR_Master_Dataset/images"
OUTPUT_ROOT  = "D:/RetinaSense/drDataset"
RANDOM_STATE = 42


# ─── Step 1: Load master CSV ──────────────────────────────────────────────────

df = pd.read_csv(MASTER_CSV)
print(f"Total images: {len(df)}")
print(f"\nGrade distribution:\n{df['dr_grade'].value_counts().sort_index()}")
print(f"\nSource distribution:\n{df['source'].value_counts()}")


# ─── Step 2: Create compound stratification key ───────────────────────────────

# Combine source + grade into one key for stratified splitting
# e.g. "DDR_3", "APTOS_2", "Messidor2_0"
df["_strat_key"] = df["source"].astype(str) + "_" + df["dr_grade"].astype(str)

# Check for rare combinations (< 4 samples can't be split 4 ways)
key_counts = df["_strat_key"].value_counts()
rare_keys  = key_counts[key_counts < 4].index.tolist()
if rare_keys:
    print(f"\n⚠  Rare (source,grade) combinations (< 4 samples): {rare_keys}")
    print("   These will be added directly to train.")

rare_df  = df[df["_strat_key"].isin(rare_keys)].copy()
split_df = df[~df["_strat_key"].isin(rare_keys)].copy()


# ─── Step 3: Split into frontend_demo + model_pool ───────────────────────────

# 2% → frontend_demo (small but enough for UI demo)
# 98% → model_pool (train + val + test)
frontend_df, model_df = train_test_split(
    split_df,
    test_size=0.98,                        # 98% to model pool
    stratify=split_df["_strat_key"],
    random_state=RANDOM_STATE
)
print(f"\nFrontend demo: {len(frontend_df)} images")
print(f"Model pool   : {len(model_df)} images")


# ─── Step 4: Split model_pool → train / val / test ───────────────────────────

# First: carve out test (14% of model_pool ≈ 14% of total)
train_val_df, test_df = train_test_split(
    model_df,
    test_size=0.14,
    stratify=model_df["_strat_key"],
    random_state=RANDOM_STATE
)

# Then: split remainder into train / val (86% train, 14% val of model_pool)
# val_size relative to train_val_df:
#   we want val ≈ 14% of model_pool
#   train_val_df = 86% of model_pool
#   so relative val fraction = 14/86 ≈ 0.163
train_df, val_df = train_test_split(
    train_val_df,
    test_size=0.163,
    stratify=train_val_df["_strat_key"],
    random_state=RANDOM_STATE
)

# Add rare samples back to train only
if len(rare_df) > 0:
    train_df = pd.concat([train_df, rare_df], ignore_index=True)

# Drop helper column
for d in [frontend_df, train_df, val_df, test_df]:
    if "_strat_key" in d.columns:
        d.drop(columns=["_strat_key"], inplace=True)
    d.reset_index(drop=True, inplace=True)

print(f"\nSplit sizes:")
print(f"  frontend_demo : {len(frontend_df):>6}")
print(f"  train         : {len(train_df):>6}")
print(f"  val           : {len(val_df):>6}")
print(f"  test          : {len(test_df):>6}")
total_check = len(frontend_df) + len(train_df) + len(val_df) + len(test_df)
print(f"  total check   : {total_check:>6}  (original: {len(df)})")


# ─── Step 5: Verify source × grade coverage ──────────────────────────────────

print(f"\n{'='*55}")
print("Verifying source × grade coverage per split...")
print(f"{'='*55}")

all_combos = set(zip(df["source"], df["dr_grade"]))
issues = []

for split_name, split_df_check in [
    ("train", train_df), ("val", val_df), ("test", test_df)
]:
    split_combos = set(zip(split_df_check["source"], split_df_check["dr_grade"]))
    missing = all_combos - split_combos
    if missing:
        issues.append(f"  ⚠  {split_name} is missing: {missing}")
    else:
        print(f"  ✓  {split_name}: all (source, grade) combinations present")

if issues:
    for issue in issues:
        print(issue)
    print("\n  Note: Missing combos are likely very rare (< 3 images).")
    print("  They are safely absorbed into train via rare_df handling.")


# ─── Step 6: Save CSVs ───────────────────────────────────────────────────────

os.makedirs(OUTPUT_ROOT, exist_ok=True)

train_df.to_csv(    f"{OUTPUT_ROOT}/train_metadata.csv",         index=False)
val_df.to_csv(      f"{OUTPUT_ROOT}/val_metadata.csv",           index=False)
test_df.to_csv(     f"{OUTPUT_ROOT}/test_metadata.csv",          index=False)
frontend_df.to_csv( f"{OUTPUT_ROOT}/frontend_demo_metadata.csv", index=False)

print(f"\nCSVs saved to {OUTPUT_ROOT}/")


# ─── Step 7: Physically copy only frontend_demo images ───────────────────────
# train/val/test do NOT need copying — the pipeline reads from
# DR_Master_Dataset/images/ directly using the image_id column

def copy_frontend_images(dataframe, dest_folder):
    os.makedirs(dest_folder, exist_ok=True)
    copied, missing = 0, 0

    for _, row in dataframe.iterrows():
        src = os.path.join(IMAGE_ROOT, row["image_id"])
        dst = os.path.join(dest_folder, row["image_id"])

        if os.path.exists(src):
            shutil.copy(src, dst)
            copied += 1
        else:
            missing += 1

    print(f"  Frontend demo: {copied} copied, {missing} missing")

print(f"\nCopying frontend_demo images...")
copy_frontend_images(frontend_df, f"{OUTPUT_ROOT}/frontend_demo")

print(f"\n{'='*55}")
print("Split complete.")
print(f"{'='*55}")
print(f"\nYour drDataset/ folder now contains:")
print(f"  train_metadata.csv          ({len(train_df)} rows)")
print(f"  val_metadata.csv            ({len(val_df)} rows)")
print(f"  test_metadata.csv           ({len(test_df)} rows)")
print(f"  frontend_demo_metadata.csv  ({len(frontend_df)} rows)")
print(f"  frontend_demo/              ({len(frontend_df)} images physically)")
print(f"\nTrain/val/test images remain in:")
print(f"  {IMAGE_ROOT}")
print(f"\nNext step: run run_pipeline.py to preprocess + train.")
