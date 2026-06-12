# ============================================================
# F1 Race Performance — AI/ML Analysis
# DataCore Analytics Internship 2026
# ============================================================

# --- IMPORTS ---
import fastf1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import warnings
warnings.filterwarnings('ignore')

# Create plots folder if it doesn't exist
os.makedirs('plots', exist_ok=True)
os.makedirs('f1_cache', exist_ok=True)

# ============================================================
# PHASE 1 — TASK 1: DATA ACQUISITION
# ============================================================

# Enable caching — saves downloaded data so we don't re-download every run
fastf1.Cache.enable_cache('f1_cache')

# Load the 2024 Bahrain Grand Prix Race session
print("=" * 60)
print("PHASE 1 — TASK 1: Loading 2024 Bahrain GP session...")
print("First run may take 1-2 minutes to download data.")
print("=" * 60)

session = fastf1.get_session(2024, 'Bahrain', 'R')
session.load()

# Get the laps data — each row = one lap by one driver
laps = session.laps

# Verify data loaded correctly
print("\n--- DATA VERIFICATION ---")
print("Shape (rows, columns):", laps.shape)
print("\nColumn names:")
print(laps.columns.tolist())
print("\nFirst 5 rows:")
print(laps.head())
print("\n✅ Phase 1 Task 1 Complete — Session loaded successfully!")

# ============================================================
# PHASE 1 — TASK 2: DATA CLEANING & PREPARATION
# ============================================================
print("\n" + "=" * 60)
print("PHASE 1 — TASK 2: Data Cleaning & Preparation")
print("=" * 60)

# Step 1: Keep only the columns we need
cols_needed = [
    'Driver', 'Team', 'LapNumber', 'LapTime',
    'Sector1Time', 'Sector2Time', 'Sector3Time',
    'Compound', 'TyreLife', 'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST'
]
laps = laps[cols_needed]
print("\n--- STEP 1: Columns selected ---")
print("Columns kept:", laps.columns.tolist())

# Step 2: Convert timedelta columns to float seconds
# FastF1 stores times as "0 days 00:01:32.850000" — we convert to 92.85
laps['LapTime']     = laps['LapTime'].dt.total_seconds()
laps['Sector1Time'] = laps['Sector1Time'].dt.total_seconds()
laps['Sector2Time'] = laps['Sector2Time'].dt.total_seconds()
laps['Sector3Time'] = laps['Sector3Time'].dt.total_seconds()
print("\n--- STEP 2: Times converted to float seconds ---")
print(laps[['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time']].head())

# Step 3: Record rows before cleaning
rows_before = len(laps)
print(f"\nRows BEFORE cleaning: {rows_before}")

# Step 4: Drop rows where LapTime is null
laps = laps[laps['LapTime'].notna()]

# Step 5: Drop rows where LapTime > 120 seconds (pit stops, safety car, crashes)
laps = laps[laps['LapTime'] <= 120]

# Step 6: Drop rows where any sector time is null
laps = laps[laps['Sector1Time'].notna()]
laps = laps[laps['Sector2Time'].notna()]
laps = laps[laps['Sector3Time'].notna()]

# Step 7: Reset the index after dropping rows
laps = laps.reset_index(drop=True)

# Step 8: Report cleaning results
rows_after = len(laps)
rows_removed = rows_before - rows_after
print(f"Rows AFTER cleaning:  {rows_after}")
print(f"Rows REMOVED:         {rows_removed}")
print(f"\nClean laps dataframe shape: {laps.shape}")
print("\nSample clean data:")
print(laps.head())
print("\n✅ Phase 1 Task 2 Complete — Data cleaned successfully!")

# ============================================================
# PHASE 2 — TASK 3: LAP TIME DISTRIBUTION ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("PHASE 2 — TASK 3: Lap Time Distribution Analysis")
print("=" * 60)

# Set seaborn theme for all plots
sns.set_theme(style='darkgrid')

# Calculate mean and median
lap_mean   = laps['LapTime'].mean()
lap_median = laps['LapTime'].median()
print(f"Mean LapTime:   {lap_mean:.2f}s")
print(f"Median LapTime: {lap_median:.2f}s")

# Plot histogram of LapTime
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(laps['LapTime'], bins=30, color='crimson', edgecolor='black', alpha=0.85)

# Add vertical dashed lines for mean and median
ax.axvline(lap_mean,   color='gold',  linestyle='--', linewidth=2, label=f'Mean: {lap_mean:.2f}s')
ax.axvline(lap_median, color='teal',  linestyle='--', linewidth=2, label=f'Median: {lap_median:.2f}s')

ax.set_title('Lap Time Distribution — 2024 Bahrain GP', fontsize=14, fontweight='bold')
ax.set_xlabel('Lap Time (seconds)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.legend()
plt.tight_layout()
plt.savefig('plots/lap_distribution.png', dpi=150)
plt.close()
print("✅ Saved: plots/lap_distribution.png")

# Top 5 fastest average lap times by driver
top5 = (laps.groupby('Driver')['LapTime']
            .mean()
            .sort_values()
            .head(5)
            .reset_index())
top5.columns = ['Driver', 'Avg LapTime (s)']
top5['Avg LapTime (s)'] = top5['Avg LapTime (s)'].round(3)
top5.index = top5.index + 1  # rank starts from 1
print("\nTop 5 Fastest Drivers (avg lap time):")
print(top5.to_string())
print("\n✅ Phase 2 Task 3 Complete!")

# ============================================================
# PHASE 2 — TASK 4: COMPOUND BOXPLOT
# ============================================================
print("\n" + "=" * 60)
print("PHASE 2 — TASK 4: Compound Boxplot")
print("=" * 60)

# Define compound colours
compound_palette = {'SOFT': 'red', 'MEDIUM': 'gold', 'HARD': 'grey'}

# Filter only known compounds
laps_compound = laps[laps['Compound'].isin(['SOFT', 'MEDIUM', 'HARD'])]

fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(
    data=laps_compound,
    x='Compound',
    y='LapTime',
    palette=compound_palette,
    order=['SOFT', 'MEDIUM', 'HARD'],
    ax=ax
)
ax.set_title('Lap Time by Tyre Compound — 2024 Bahrain GP', fontsize=14, fontweight='bold')
ax.set_xlabel('Tyre Compound', fontsize=12)
ax.set_ylabel('Lap Time (seconds)', fontsize=12)
plt.tight_layout()
plt.savefig('plots/compound_boxplot.png', dpi=150)
plt.close()
print("✅ Saved: plots/compound_boxplot.png")
print("\n✅ Phase 2 Task 4 Complete!")

# ============================================================
# PHASE 2 — TASK 5: SECTOR & SPEED TRAP ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("PHASE 2 — TASK 5: Sector & Speed Trap Analysis")
print("=" * 60)

# Calculate average sector times per driver, sorted by total lap time
sector_avg = (laps.groupby('Driver')[['Sector1Time', 'Sector2Time', 'Sector3Time']]
                  .mean()
                  .reset_index())
sector_avg['TotalTime'] = sector_avg['Sector1Time'] + sector_avg['Sector2Time'] + sector_avg['Sector3Time']
sector_avg = sector_avg.sort_values('TotalTime').head(8)  # top 8 drivers

# Melt for grouped bar chart
sector_melt = sector_avg.melt(
    id_vars='Driver',
    value_vars=['Sector1Time', 'Sector2Time', 'Sector3Time'],
    var_name='Sector',
    value_name='Time'
)
sector_melt['Sector'] = sector_melt['Sector'].map({
    'Sector1Time': 'S1',
    'Sector2Time': 'S2',
    'Sector3Time': 'S3'
})

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(
    data=sector_melt,
    x='Driver',
    y='Time',
    hue='Sector',
    palette={'S1': 'crimson', 'S2': 'gold', 'S3': 'teal'},
    ax=ax
)
ax.set_title('Average Sector Times by Driver — 2024 Bahrain GP', fontsize=14, fontweight='bold')
ax.set_xlabel('Driver', fontsize=12)
ax.set_ylabel('Time (seconds)', fontsize=12)
ax.legend(title='Sector')
plt.tight_layout()
plt.savefig('plots/sector_comparison.png', dpi=150)
plt.close()
print("✅ Saved: plots/sector_comparison.png")

# Pearson correlation between SpeedST and LapTime
corr = laps['SpeedST'].corr(laps['LapTime'])
print(f"\nPearson Correlation (SpeedST vs LapTime): r = {corr:.3f}")
if corr < 0:
    print("Interpretation: Higher straight-line speed is weakly associated with FASTER lap times (negative = inverse relationship).")
else:
    print("Interpretation: Higher straight-line speed is associated with SLOWER lap times.")

print("\n✅ Phase 2 Task 5 Complete!")

# ============================================================
# PHASE 2 — TASK 6: SPEED VS LAP TIME SCATTER
# ============================================================
print("\n" + "=" * 60)
print("PHASE 2 — TASK 6: Speed vs Lap Time Scatter")
print("=" * 60)

# Filter only known compounds for colour coding
laps_sc = laps[laps['Compound'].isin(['SOFT', 'MEDIUM', 'HARD'])].dropna(subset=['SpeedST'])

compound_colors = {'SOFT': 'red', 'MEDIUM': 'gold', 'HARD': 'grey'}

fig, ax = plt.subplots(figsize=(10, 6))

# Plot each compound separately for legend
for compound, color in compound_colors.items():
    subset = laps_sc[laps_sc['Compound'] == compound]
    ax.scatter(subset['SpeedST'], subset['LapTime'],
               color=color, label=compound, alpha=0.6, s=30)

# Add regression line
m, b = np.polyfit(laps_sc['SpeedST'], laps_sc['LapTime'], 1)
x_line = np.linspace(laps_sc['SpeedST'].min(), laps_sc['SpeedST'].max(), 100)
ax.plot(x_line, m * x_line + b, color='white', linestyle='--', linewidth=2, label='Trend')

ax.set_title(f'Speed Trap vs Lap Time  (r = {corr:.3f})', fontsize=14, fontweight='bold')
ax.set_xlabel('Speed Trap (km/h)', fontsize=12)
ax.set_ylabel('Lap Time (seconds)', fontsize=12)
ax.legend(title='Compound')
plt.tight_layout()
plt.savefig('plots/speed_correlation.png', dpi=150)
plt.close()
print("✅ Saved: plots/speed_correlation.png")
print("\n✅ Phase 2 Task 6 Complete!")

# ============================================================
# PHASE 3 — TASK 7: FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 60)
print("PHASE 3 — TASK 7: Feature Engineering")
print("=" * 60)

# Work on a copy so original laps dataframe stays clean
laps_ml = laps.copy()

# Feature 1: SectorBalance — difference between S1 and S3
# Captures whether a driver is stronger in sector 1 vs sector 3
laps_ml['SectorBalance'] = laps_ml['Sector1Time'] - laps_ml['Sector3Time']

# Feature 2: Bin TyreLife into Fresh / Used / Old
# pd.cut() divides a continuous number into labelled buckets
laps_ml['TyreAge_Bucket'] = pd.cut(
    laps_ml['TyreLife'],
    bins=[0, 10, 25, 100],
    labels=['Fresh', 'Used', 'Old']
)

# Feature 3: One-hot encode Compound and TyreAge_Bucket
# Converts text categories into 0/1 columns the model can read
laps_ml = pd.get_dummies(laps_ml, columns=['Compound', 'TyreAge_Bucket'], dtype=int)

# Feature 4: Encode Driver as a number
# LabelEncoder converts VER→0, HAM→1, LEC→2 etc.
le = LabelEncoder()
laps_ml['Driver_Enc'] = le.fit_transform(laps_ml['Driver'])

# Step 5: Define feature columns (X) and target (y)
# IMPORTANT: Never put raw sector times or LapTime in X — that leaks the answer!
feature_cols = ['LapNumber', 'TyreLife', 'SectorBalance',
                'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST', 'Driver_Enc']

# Add any dummy columns that were created
dummy_cols = [c for c in laps_ml.columns if c.startswith('Compound_') or c.startswith('TyreAge_')]
feature_cols += dummy_cols

# Drop rows with any nulls in feature columns or target
laps_ml = laps_ml.dropna(subset=feature_cols + ['LapTime'])

X = laps_ml[feature_cols]
y = laps_ml['LapTime']

print(f"Feature matrix X shape: {X.shape}")
print(f"Target y shape:         {y.shape}")
print(f"Null values in X:       {X.isnull().sum().sum()}")
print(f"Null values in y:       {y.isnull().sum()}")
print(f"\nFeature columns used:")
for i, col in enumerate(feature_cols, 1):
    print(f"  {i:2}. {col}")
print("\n✅ Phase 3 Task 7 Complete — Features ready!")

# ============================================================
# PHASE 3 — TASK 8: RANDOM FOREST MODEL
# ============================================================
print("\n" + "=" * 60)
print("PHASE 3 — TASK 8: Random Forest Regression Model")
print("=" * 60)

# Split data: 80% for training, 20% for testing
# random_state=42 ensures same split every run (reproducible)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

# Train the Random Forest model
# n_estimators=100 means it builds 100 decision trees and averages them
print("\nTraining Random Forest model...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
print("Training complete!")

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print(f"\n--- MODEL EVALUATION ---")
print(f"MAE  (Mean Absolute Error):      {mae:.4f} seconds")
print(f"RMSE (Root Mean Squared Error):  {rmse:.4f} seconds")
print(f"R²   (R-Squared Score):          {r2:.4f}")

if r2 >= 0.85:
    print("✅ R² > 0.85 — Model meets the assignment target!")
else:
    print("⚠️  R² < 0.85 — Try revisiting feature engineering.")

print("\n✅ Phase 3 Task 8 Complete — Model trained!")

# ============================================================
# PHASE 3 — TASK 8 (PLOTS): PREDICTED VS ACTUAL
# ============================================================
print("\n" + "=" * 60)
print("PHASE 3 — TASK 8 PLOT: Predicted vs Actual Lap Times")
print("=" * 60)

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_test, y_pred, color='crimson', alpha=0.6, s=30, label='Predictions')

# Perfect prediction reference line (where predicted == actual)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
ax.plot([min_val, max_val], [min_val, max_val],
        color='gold', linestyle='--', linewidth=2, label='Perfect prediction')

ax.set_title(f'Predicted vs Actual Lap Times\nRandom Forest — R² = {r2:.2f}',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Actual Lap Time (s)', fontsize=12)
ax.set_ylabel('Predicted Lap Time (s)', fontsize=12)
ax.legend()
plt.tight_layout()
plt.savefig('plots/predicted_vs_actual.png', dpi=150)
plt.close()
print("✅ Saved: plots/predicted_vs_actual.png")

# ============================================================
# PHASE 3 — TASK 9: FEATURE IMPORTANCE
# ============================================================
print("\n" + "=" * 60)
print("PHASE 3 — TASK 9: Feature Importance Chart")
print("=" * 60)

# Get feature importances from the trained model
importances = pd.Series(rf_model.feature_importances_, index=feature_cols)
importances = importances.sort_values(ascending=True).tail(10)  # top 10

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(importances.index, importances.values, color='crimson')

# Add value labels on each bar
for bar, val in zip(bars, importances.values):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f'{val:.2f}', va='center', fontsize=9)

ax.set_title('Top Feature Importances — Random Forest', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance Score', fontsize=12)
plt.tight_layout()
plt.savefig('plots/feature_importance.png', dpi=150)
plt.close()
print("✅ Saved: plots/feature_importance.png")
print("\n✅ Phase 3 Task 9 Complete!")

# ============================================================
# PHASE 3 — TASK 10: ANOMALY DETECTION
# ============================================================
print("\n" + "=" * 60)
print("PHASE 3 — TASK 10: Anomaly Detection")
print("=" * 60)

# For each driver, compute median and standard deviation of LapTime
driver_stats = laps.groupby('Driver')['LapTime'].agg(['median', 'std']).reset_index()
driver_stats.columns = ['Driver', 'MedianLap', 'StdLap']

# Merge stats back into laps
laps = laps.merge(driver_stats, on='Driver', how='left')

# Flag anomaly laps: LapTime > median + 2 × std
# This is the statistical "2 sigma" rule — anything beyond 2 standard
# deviations from the median is considered unusual
laps['IsAnomaly'] = laps['LapTime'] > (laps['MedianLap'] + 2 * laps['StdLap'])

# Per-driver anomaly summary
anomaly_summary = (laps.groupby('Driver')['IsAnomaly']
                       .sum()
                       .sort_values(ascending=False)
                       .reset_index())
anomaly_summary.columns = ['Driver', 'Anomaly Laps']
print("\nPer-driver anomaly lap count:")
print(anomaly_summary.to_string(index=False))

# Plot for top 3 drivers (by most laps = most data)
top3_drivers = (laps.groupby('Driver')['LapNumber']
                    .count()
                    .sort_values(ascending=False)
                    .head(3)
                    .index.tolist())

driver_colors = ['crimson', 'teal', 'gold']
fig, ax = plt.subplots(figsize=(12, 6))

for driver, color in zip(top3_drivers, driver_colors):
    driver_laps = laps[laps['Driver'] == driver]
    normal_laps  = driver_laps[~driver_laps['IsAnomaly']]
    anomaly_laps = driver_laps[driver_laps['IsAnomaly']]

    # Plot normal laps as a line
    ax.plot(normal_laps['LapNumber'], normal_laps['LapTime'],
            color=color, label=driver, linewidth=1.5)

    # Plot anomaly laps as red X markers
    ax.scatter(anomaly_laps['LapNumber'], anomaly_laps['LapTime'],
               color='red', marker='x', s=100, zorder=5, linewidths=2)

ax.set_title('Anomaly Detection — Lap Time per Driver\n✕ = flagged anomaly lap',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Lap Number', fontsize=12)
ax.set_ylabel('Lap Time (seconds)', fontsize=12)
ax.legend()
plt.tight_layout()
plt.savefig('plots/anomaly_detection.png', dpi=150)
plt.close()
print("✅ Saved: plots/anomaly_detection.png")
print("\n✅ Phase 3 Task 10 Complete!")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("🏁 ALL PHASES COMPLETE — SUMMARY")
print("=" * 60)
print(f"Total clean laps analysed:     {len(laps)}")
print(f"Model MAE:                     {mae:.4f}s")
print(f"Model RMSE:                    {rmse:.4f}s")
print(f"Model R²:                      {r2:.4f}")
print(f"Top feature:                   {importances.index[-1]}")
print(f"Total anomaly laps flagged:    {laps['IsAnomaly'].sum()}")
print("\nPlots saved in plots/ folder:")
plots = [
    'lap_distribution.png', 'compound_boxplot.png',
    'sector_comparison.png', 'speed_correlation.png',
    'predicted_vs_actual.png', 'feature_importance.png',
    'anomaly_detection.png'
]
for p in plots:
    print(f"  ✅ plots/{p}")
print("\n🎉 F1 AI/ML Analysis Complete! Push to GitHub and submit report.")