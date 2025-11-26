# Generated script for Pub-Homonymous - Run 2
# Generated on: 2025-11-18T18:43:05.053753
# Model: grok-4-fast

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import defaultdict

# Configuration parameters
time_threshold = 2.0
require_same_resource = False
min_matching_events = 2
max_mismatches = 1
activity_suffix_pattern = r'(_signed\d*|_\d+)$'
similarity_threshold = 0.8
case_sensitive = False
use_fuzzy_matching = False
linkage_method = 'average'

# File paths and columns
input_file = 'data/pub/Pub-Homonymous.csv'
output_file = 'data/pub/pub_homonymous_cleaned_run2.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
homonymous_suffix = ':homonymous'

# Load the data
df = pd.read_csv(input_file)
print(f"Run 2: Original dataset shape: {df.shape}")

# Check required columns
required = [case_column, activity_column, timestamp_column]
for col in required:
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' not found in the dataset.")

# Normalize CaseID to Case if needed
if 'CaseID' in df.columns:
    df = df.rename(columns={'CaseID': case_column})

# Create ishomonymous
df['ishomonymous'] = df[activity_column].apply(lambda x: 1 if isinstance(x, str) and x.endswith(homonymous_suffix) else 0)

# Create BaseActivity
def get_base(act):
    if isinstance(act, str) and act.endswith(homonymous_suffix):
        return act[:-len(homonymous_suffix)]
    return act

df['BaseActivity'] = df[activity_column].apply(get_base)

# Create CleanBase for homonymous
mask_hom = df['ishomonymous'] == 1
df['CleanBase'] = df['BaseActivity']
df.loc[mask_hom, 'CleanBase'] = df.loc[mask_hom, 'BaseActivity'].apply(
    lambda x: re.sub(activity_suffix_pattern, '', str(x), flags=re.I)
)

# Function to normalize to processed
def get_processed(text):
    if not isinstance(text, str):
        return ''
    norm = re.sub(r'[_-]', ' ', text)
    norm = re.sub(r'\s+', ' ', norm.strip())
    return norm.lower()

# Create ProcessedActivity (only meaningful for homonymous)
df['ProcessedActivity'] = df['CleanBase'].apply(get_processed)

# Clustering for homonymous
df_hom = df[mask_hom]
if len(df_hom) > 0:
    unique_hom_processed = sorted(df_hom['ProcessedActivity'].dropna().unique())
    unique_hom_processed = [p for p in unique_hom_processed if p]
    df['cluster_id'] = -1
    if len(unique_hom_processed) > 0:
        vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
        X = vectorizer.fit_transform(unique_hom_processed).toarray()
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric='cosine',
            linkage=linkage_method,
            distance_threshold=1 - similarity_threshold
        )
        cluster_labels = clustering.fit_predict(X)
        proc_to_cid = dict(zip(unique_hom_processed, cluster_labels))
        df.loc[mask_hom, 'cluster_id'] = df.loc[mask_hom, 'ProcessedActivity'].map(proc_to_cid)
else:
    df['cluster_id'] = -1

# Determine canonical per qualifying cluster
cluster_to_canonical = {}
for cid in df.loc[mask_hom, 'cluster_id'].dropna().unique():
    cluster_mask = mask_hom & (df['cluster_id'] == cid)
    num_events = cluster_mask.sum()
    if num_events >= min_matching_events:
        cleanbase_counts = df.loc[cluster_mask, 'CleanBase'].value_counts()
        if len(cleanbase_counts) > 0:
            most_freq_clean = cleanbase_counts.index[0]
            canonical = get_processed(most_freq_clean)
            cluster_to_canonical[cid] = canonical

# Assign Activity_fixed
df['Activity_fixed'] = np.where(~mask_hom, df['Activity'], df['BaseActivity'])
for cid, can in cluster_to_canonical.items():
    cluster_mask = mask_hom & (df['cluster_id'] == cid)
    df.loc[cluster_mask, 'Activity_fixed'] = can

# 7. Calculate Detection Metrics (BEFORE FIXING, but independent)
print("=== Detection Performance Metrics ===")
if label_column in df.columns:
    y_true = (df[label_column].notna() & (df[label_column].astype(str) != '')).astype(int)
    y_pred = df['ishomonymous'].astype(int)
    if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
    else:
        prec = rec = f1 = 0.0
else:
    prec = rec = f1 = 0.0
    print("No labels available for metric calculation.")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-Score: {f1:.4f}")
threshold_met = "met" if prec >= 0.6 else "not met"
print(f"Precision threshold (≥ 0.6) {threshold_met}")

# 8. Integrity Check
clean_modified = ((df['ishomonymous'] == 0) & (df['BaseActivity'] != df['Activity_fixed'])).sum()
total_hom = df['ishomonymous'].sum()
hom_unchanged = ((df['ishomonymous'] == 1) & (df['BaseActivity'] == df['Activity_fixed'])).sum()
hom_corrected = total_hom - hom_unchanged
print(f"Clean activities modified: {clean_modified} (should be 0)")
print(f"Homonymous activities unchanged: {hom_unchanged}")
print(f"Homonymous activities corrected: {hom_corrected}")

# 9. Save Output
# Format timestamp
df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')

# Prepare output df: keep all original + new columns, drop temps
df_output = df.drop(columns=['BaseActivity', 'CleanBase', 'ProcessedActivity', 'cluster_id'], errors='ignore')

df_output.to_csv(output_file, index=False)

# 10. Summary Statistics
print(f"Total number of events: {len(df)}")
print(f"Number of homonymous events detected: {total_hom}")
print(f"Unique activities before fixing: {df[activity_column].nunique()}")
print(f"Unique activities after fixing: {df['Activity_fixed'].nunique()}")
print(f"Output file path: {output_file}")

# Required prints
print(f"Run 2: Processed dataset saved to: {output_file}")
print(f"Run 2: Final dataset shape: {df_output.shape}")
print(f"Run 2: Dataset: pub")
print(f"Run 2: Task type: homonymous")