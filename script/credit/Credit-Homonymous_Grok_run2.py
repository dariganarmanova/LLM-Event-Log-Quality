# Generated script for Credit-Homonymous - Run 2
# Generated on: 2025-11-18T19:27:28.939404
# Model: grok-4-fast

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import Counter
import re

# Configuration
input_file = 'data/credit/Credit-Homonymous.csv'
output_file = 'data/credit/credit_homonymous_cleaned_run2.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
homonymous_suffix = ':homonymous'
similarity_threshold = 0.8
linkage_method = 'average'
activity_suffix_pattern = r'(_signed\d*|_\d+)$'

# Load data
df = pd.read_csv(input_file)
print(f"Run 2: Original dataset shape: {df.shape}")

# Ensure required columns exist
required = [case_column, activity_column, timestamp_column]
for col in required:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# Normalize CaseID to Case if present
if 'CaseID' in df.columns:
    df[case_column] = df['CaseID']

# Step 2: Identify homonymous activities
df['ishomonymous'] = df[activity_column].str.endswith(homonymous_suffix, na=False).astype(int)
df['BaseActivity'] = df[activity_column].str.replace(homonymous_suffix, '', regex=False)

# Step 3: Preprocess activity names
df['CleanedBase'] = df['BaseActivity']
mask_homo = df['ishomonymous'] == 1
df.loc[mask_homo, 'CleanedBase'] = df.loc[mask_homo, 'BaseActivity'].str.replace(activity_suffix_pattern, '', regex=True)
df['ProcessedActivity'] = (
    df['CleanedBase']
    .astype(str)
    .str.lower()
    .str.replace(r'[_-]', ' ', regex=True)
    .str.replace(r'\s+', ' ', regex=True)
    .str.strip()
)

# Step 7: Calculate detection metrics (before fixing)
if label_column in df.columns:
    y_true = (df[label_column].notna() & (df[label_column].astype(str) != '')).astype(int)
    y_pred = df['ishomonymous']
    if len(y_true) > 0 and y_true.sum() > 0 and y_pred.sum() > 0:
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
    else:
        prec = rec = f1 = 0.0
else:
    print("No labels available for metric calculation.")
    prec = rec = f1 = 0.0

print("=== Detection Performance Metrics ===")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-Score: {f1:.4f}")
if prec >= 0.6:
    print("Precision threshold (>= 0.6) met")
else:
    print("Precision threshold (>= 0.6) not met")

# Steps 4-6: Vectorize, cluster, and fix (only for homonymous)
homo_mask = df['ishomonymous'] == 1
unique_processed_homo = df.loc[homo_mask, 'ProcessedActivity'].dropna().unique()
if len(unique_processed_homo) > 0:
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
    vectors = vectorizer.fit_transform(unique_processed_homo)
    dense_vectors = vectors.toarray()

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric='cosine',
        linkage=linkage_method,
        distance_threshold=1 - similarity_threshold
    )
    cluster_labels = clustering.fit_predict(dense_vectors)

    cluster_map = dict(zip(unique_processed_homo, cluster_labels))
    df['cluster_id'] = -1
    valid_map_mask = df.loc[homo_mask, 'ProcessedActivity'].isin(cluster_map.keys())
    df.loc[homo_mask & valid_map_mask, 'cluster_id'] = df.loc[homo_mask & valid_map_mask, 'ProcessedActivity'].map(cluster_map)

    canonicals = {}
    unique_clusters = df.loc[homo_mask, 'cluster_id'].dropna().unique()
    for cid in unique_clusters:
        cluster_rows_mask = (df['cluster_id'] == cid)
        proc_counter = Counter(df.loc[cluster_rows_mask, 'ProcessedActivity'])
        if proc_counter:
            most_common_proc = proc_counter.most_common(1)[0][0]
            canonicals[cid] = most_common_proc

    df['Activity_fixed'] = df[activity_column]
    for cid, canon in canonicals.items():
        fix_mask = (df['ishomonymous'] == 1) & (df['cluster_id'] == cid)
        df.loc[fix_mask, 'Activity_fixed'] = canon

    # For any unmapped homonymous (e.g., NaN processed), keep original BaseActivity
    unmapped_mask = (df['ishomonymous'] == 1) & (df['cluster_id'] == -1)
    df.loc[unmapped_mask, 'Activity_fixed'] = df.loc[unmapped_mask, 'BaseActivity']
else:
    df['Activity_fixed'] = df[activity_column]
    df['cluster_id'] = -1

# Step 8: Integrity check
clean_modified = ((df['ishomonymous'] == 0) & (df['Activity'] != df['Activity_fixed'])).sum()
homo_total = df['ishomonymous'].sum()
homo_unchanged_mask = (df['ishomonymous'] == 1) & (df['BaseActivity'] == df['Activity_fixed'])
homo_unchanged = homo_unchanged_mask.sum()
homo_corrected = homo_total - homo_unchanged

print("\n=== Integrity Check ===")
print(f"Clean activities modified: {clean_modified} (should be 0)")
print(f"Homonymous activities unchanged: {homo_unchanged}")
print(f"Homonymous activities corrected: {homo_corrected}")

# Step 9: Prepare and save output
df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')

output_columns = [case_column, timestamp_column, activity_column, 'Activity_fixed', 'ishomonymous']
if label_column in df.columns:
    output_columns.append(label_column)
if 'Resource' in df.columns:
    output_columns.insert(2, 'Resource')
if 'Variant' in df.columns:
    output_columns.append('Variant')

df_output = df[output_columns]
df_output.to_csv(output_file, index=False)

# Step 10: Summary statistics
total_events = len(df)
homo_detected = df['ishomonymous'].sum()
unique_before = df[activity_column].nunique()
unique_after = df['Activity_fixed'].nunique()

print("\n=== Summary Statistics ===")
print(f"Total number of events: {total_events}")
print(f"Number of homonymous events detected: {homo_detected}")
print(f"Unique activities before fixing: {unique_before}")
print(f"Unique activities after fixing: {unique_after}")
print(f"Output file path: {output_file}")

# Required prints
print(f"Run 2: Processed dataset saved to: {output_file}")
print(f"Run 2: Final dataset shape: {df_output.shape}")
print(f"Run 2: Dataset: credit")
print(f"Run 2: Task type: homonymous")