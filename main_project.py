import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Setup aesthetics
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (10, 6)

print("="*50)
print("[*] Starting Advanced Exploratory Data Analysis (EDA) on the original dataset...")

# 2. Read the original data
file_path = "crop_data.csv"
if not os.path.exists(file_path):
    print("Error: The original crop_data.csv file was not found.")
    exit()

df = pd.read_csv(file_path)
print(f"Success! Loaded original dataset with {df.shape[0]} rows and {df.shape[1]} columns.")

# 3. Quick Stats
print("\n[*] Descriptive Statistics:")
print(df.describe().round(2))
print("="*50)

# ==========================================
# 4. Data Visualization Plots
# ==========================================
print("\n[*] Generating Professional Charts...")

# Chart 1: Correlation Matrix
plt.figure(figsize=(10, 8))
# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title("Correlation Matrix (Original Data)", fontsize=16, fontweight='bold')
plt.savefig("correlation_heatmap.png", bbox_inches='tight', dpi=300)
print("1 -> Saved: correlation_heatmap.png")
plt.close()

# Chart 2: Humidity Requirements for Top 5 Crops
sample_crops = ['rice', 'maize', 'coffee', 'cotton', 'apple']
sample_df = df[df['label'].isin(sample_crops)]

plt.figure(figsize=(12, 6))
sns.boxplot(x='label', y='humidity', data=sample_df, palette="magma")
plt.title("Humidity Requirements (Top 5 Crops)", fontsize=15, fontweight='bold')
plt.xlabel("Crop Type")
plt.ylabel("Humidity (%)")
plt.savefig("humidity_analysis.png", bbox_inches='tight', dpi=300)
print("2 -> Saved: humidity_analysis.png")
plt.close()

print("\nEDA Completed! Check out the generated images.")
print("="*50)
