# Generated script for BPIC15-Homonymous - Run 3
# Generated on: 2025-11-18T21:50:56.174132
# Model: grok-4-fast

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import Counter
import re

# Configuration parameters
input_file = 'data/bpic15/BPIC15-Homonymous.csv'
output_file = 'data/bpic15/bpic15_homonymous_cleaned_run3.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
homonymous_suffix = ':homonymous'
activity_suffix_pattern = r'(_signed\d*|_\d+)$'
similarity_threshold = 0.8
linkage_method = 'average'

# Load the data
df = pd.read_csv(input_file)
print(f"Run 3: Original dataset shape: {df.shape}")

# Ensure required columns exist
required_cols = [case_column, activity_column, timestamp_column]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' not found in the dataset.")

# Normalize CaseID to Case if needed
if 'Case ID' in df.columns:
    df[case_column] = df['Case ID']

# Step 2: Identify Homonymous Activities
df['ishomonymous'] = df[activity_column].str.endswith(homonymous_suffix).astype(int)
df['BaseActivity'] = df[activity_column].str.replace(homonymous_suffix + '$', '', regex=True)

# Step 3: Preprocess Activity Names
df['CleanBase'] = df['BaseActivity'].str.replace(activity_suffix_pattern, '', regex=True)
df['ProcessedActivity'] = df['CleanBase'].str.lower()
df['ProcessedActivity'] = df['ProcessedActivity'].str.replace(r'[_-]', ' ', regex=True)
df['ProcessedActivity'] = df['ProcessedActivity'].str.replace(r'\s+', ' ', regex=True).str.strip()

# Step 4: Vectorize Activities
unique_processed = df['ProcessedActivity'].dropna().unique()
if len(unique_processed) > 0:
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
    X = vectorizer.fit_transform(unique_processed).toarray()
else:
    X = []
    unique_processed = []

# Step 5: Cluster Similar Activities
if len(X) > 0:
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric='cosine',
        linkage=linkage_method,
        distance_threshold=1 - similarity_threshold
    )
    cluster_labels = clustering.fit_predict(X)
    cluster_map = dict(zip(unique_processed, cluster_labels))
    df['cluster'] = df['ProcessedActivity'].map(cluster_map)
    df['cluster'] = df['cluster'].fillna(-1)  # For any unmapped (should not happen)
else:
    df['cluster'] = -1

# Step 6: Majority Voting Within Clusters
canonical = {}
if len(df['cluster'].unique()) > 1 or df['cluster'].iloc[0] != -1:
    for cluster_id in df['cluster'].unique():
        if cluster_id == -1:
            continue
        mask = df['cluster'] == cluster_id
        if mask.sum() > 0:
            clean_bases = df.loc[mask, 'CleanBase'].value_counts()
            most_freq = clean_bases.index[0]
            canonical[cluster_id] = most_freq

# Assign Activity_fixed
df['Activity_fixed'] = df[activity_column].copy()
homo_mask = df['ishomonymous'] == 1
df.loc[homo_mask, 'Activity_fixed'] = df.loc[homo_mask, 'cluster'].map(canonical).fillna(df.loc[homo_mask, 'CleanBase'])

# Step 7: Calculate Detection Metrics (BEFORE FIXING)
if label_column in df.columns:
    y_true = (df[label_column].notna() & (df[label_column] != '')).astype(int)
    y_pred = df['ishomonymous']
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print("=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    met = "met" if precision >= 0.6 else "not met"
    print(f"Precision threshold (≥ 0.6) {met}")
else:
    print("=== Detection Performance Metrics ===")
    print("Precision: 0.0000")
    print("Recall: 0.0000")
    print("F1-Score: 0.0000")
    print("Precision threshold (≥ 0.6) not met")
    print("No labels available for metric calculation.")

# Step 8: Integrity Check
clean_modified = ((df['ishomonymous'] == 0) & (df[activity_column] != df['Activity_fixed'])).sum()
homo_unchanged = ((df['ishomonymous'] == 1) & (df['Activity_fixed'] == df['BaseActivity'])).sum()
homo_corrected = ((df['ishomonymous'] == 1) & (df['Activity_fixed'] != df['BaseActivity'])).sum()
print(f"Clean activities modified: {clean_modified} (should be 0)")
print(f"Homonymous activities unchanged: {homo_unchanged}")
print(f"Homonymous activities corrected: {homo_corrected}")

# Step 9: Save Output
base_cols = [case_column, timestamp_column]
if 'Resource' in df.columns:
    base_cols.append('Resource')
if 'Variant' in df.columns:
    base_cols.append('Variant')
base_cols += [activity_column, 'Activity_fixed', 'ishomonymous']
if label_column in df.columns:
    base_cols.append(label_column)

df_output = df[base_cols].copy()
df_output[timestamp_column] = pd.to_datetime(df_output[timestamp_column], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
df_output.to_csv(output_file, index=False)

# Step 10: Summary Statistics
print(f"Total number of events: {len(df)}")
print(f"Number of homonymous events detected: {df['ishomonymous'].sum()}")
print(f"Unique activities before fixing: {df[activity_column].nunique()}")
print(f"Unique activities after fixing: {df['Activity_fixed'].nunique()}")
print(f"Output file path: {output_file}")

# Required prints
print(f"Run 3: Processed dataset saved to: {output_file}")
print(f"Run 3: Final dataset shape: {df_output.shape}")
print(f"Run 3: Dataset: bpic15")
print(f"Run 3: Task type: homonymous")