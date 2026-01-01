import csv
import sys
import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from helpers import *
from datetime import datetime

# Logs will be saved in 'logs' directory
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Logging into text file
class Logs:
    def __init__(self, *files):
        self.files = files

    def write(self, message):
        for f in self.files:
            f.write(message)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

log_filename = os.path.join(
    log_dir,
    f"data_preprocessing_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
)

log_file = open(log_filename, "w", encoding="utf-8")

sys.stdout = Logs(sys.stdout, log_file)
sys.stderr = Logs(sys.stderr, log_file)

print(f"Logging started: {log_filename}")

# Initialize tqdm for pandas
tqdm.pandas()

# Import Dataset
df = pd.read_csv('data.csv')

# Exploratory Data Analysis 
print("\n" + "="*50)
print("EXPLORATORY DATA ANALYSIS (EDA)")
print("="*50)

# Dataset overview
print("\nDataset Shape:", df.shape)
print("\nColumn Names:", df.columns.tolist())
print("\nData Types:\n", df.dtypes)
print("\nFirst 10 rows:\n", df.head(10))
print("\nLast 10 rows:\n", df.tail(10))

# Identify categorical and numeric columns
categorical_cols = df.select_dtypes(include=['object']).columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(f"\nCategorical columns: {categorical_cols.tolist()}")
print(f"Numerical columns: {numeric_cols.tolist()}")

# Convert 'content' column to lowercase
df_lower = df.copy()
df_lower['content'] = df_lower['content'].astype(str).str.lower().str.strip().str.replace(r'[\r\n]+', '', regex=True)

# Check for duplicates in 'content' column
duplicates = df_lower[df_lower['content'].duplicated(keep=False)].sort_values(by='content')
print("\nDuplicate content rows:", df_lower['content'].duplicated().sum())
print("\nFirst 5 duplicate rows:")
print(duplicates.head(5))

# Check for missing values
df_check = df_lower.copy()
for col in df_check.select_dtypes(include=['object']).columns:
    df_check[col] = df_check[col].str.strip().str.lower()
df_check = df_check.replace(missing_strings, np.nan)
missing_rows = df_check[df_check.isnull().any(axis=1)]
print("\nMissing values per column:\n", df_check.isnull().sum())
print("\nSample rows with missing values:\n", missing_rows.head())

# Detect noise
noise_rows = df[df['content'].apply(noise_detection)]
total_noises = df['content'].apply(noise_detection).sum()
print(f"\nTotal noise rows in dataset: {total_noises}")
print("\nSample noise rows:")
print(noise_rows.head(5))

# Class distribution
class_counts = df_check['sentiment'].value_counts()
print("\nTotal Record Count:")
print(class_counts)

# Plot class distribution
plt.figure(figsize=(6,4))
sns.barplot(x=class_counts.index, y=class_counts.values, hue=class_counts.index, palette=['#004080', '#0059b3', '#0073e6'] , legend=False)
plt.title("Class Distribution of Sentiment")
plt.xlabel("Sentiment")
plt.ylabel("Count")
print("\nPlot graph for Sentiment has generated. \nClose the plot window to continue the process.")
plt.show()

print("\n" + "="*50)
print("END OF EXPLORATORY DATA ANALYSIS")
print("="*50)

time.sleep(3)

# Data Preprocessing
print("\n" + "="*50)
print("PERFORMING DATA PREPROCESSING")
print("="*50)
    
df_clean = df.copy()

# Normalize text (case-insensitive)
for col in df_clean.select_dtypes(include=['object']).columns:
    df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()
    
# Remove noise rows
before = df_clean.shape[0]
noise_mask = df_clean['content'].apply(noise_detection)
df_clean = df_clean[~noise_mask].reset_index(drop=True)
removed_noise = before - df_clean.shape[0]
print(f"Noise rows removed: {removed_noise}")

# Apply text preprocessing
print("\nApplying text preprocessing...")
df_clean['content'] = df_clean['content'].progress_apply(text_preprocess)

# Remove duplicates
before = df_clean.shape[0]
df_clean = df_clean.drop_duplicates(subset=['content']).reset_index(drop=True)
removed_duplicates_after = before - df_clean.shape[0]
print(f"Duplicate content rows removed after text preprocessing: {removed_duplicates_after}")
    
# Remove NaN / empty rows
before = df_clean.shape[0]
df_clean = df_clean.replace(missing_strings, np.nan).dropna()
removed_nan = before - df_clean.shape[0]
print(f"\nRows removed due to NaN: {removed_nan}")
    
# Evaluate dfset shape count after cleaning
total_removed = removed_nan + removed_noise
print(f"\nTotal rows removed in all steps: {total_removed}")
print(f"Final dataset shape: {df_clean.shape}")

# Class distribution
print("\nPreprocessed Class Distribution:")
print(df_clean['sentiment'].value_counts())

# Find the least class size
min_size = df_clean['sentiment'].value_counts().min()
print(f"Least class size: {min_size}")

# Resample each class to match the least class size
balanced_frames = []
for sentiment_val in [-1, 0, 1]:
    df_class = df_clean[df_clean['sentiment'] == sentiment_val]
    if not df_class.empty:
        df_resampled = df_class.sample(n=min_size, random_state=30)
        balanced_frames.append(df_resampled)

# Combine and shuffle the balanced dataset
df_balanced = pd.concat(balanced_frames).sample(frac=1, random_state=30).reset_index(drop=True)

# Display new class distribution
print("\nBalanced Class Distribution:")
print(df_balanced['sentiment'].value_counts())

# Export balanced dataset
df_balanced.to_csv('preprocessed_data.csv', index=False, quoting=csv.QUOTE_ALL)
print("\nBalanced preprocessed data saved to 'preprocessed_data.csv'")
    
print("\n" + "="*50)
print("END OF DATA PREPROCESSING")
print("="*50)

log_file.close()