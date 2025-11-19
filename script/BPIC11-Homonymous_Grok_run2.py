# Generated script for BPIC11-Homonymous - Run 2
# Generated on: 2025-11-18T22:37:51.505381
# Model: grok-4-fast

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import precision_score, recall_score, f1_score
import re

# Configuration parameters
time_threshold = 2.0
require_same_resource = False
min_matching_events = 2
max_mismatches = 1
activity_suffix_pattern = r"(_signed\d*|_\d+)$"
similarity_threshold = 0.8
case_sensitive = False
use_fuzzy_matching = False
linkage_method = 'average'
homonymous_suffix = ':homonymous'
input_file = 'data/bpic11/BPIC11-Homonymous.csv'
output_file = 'data/bpic11/bpic11_homonymous_cleaned_run2.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'

# Load the data
df = pd.read_csv(input_file)

# Normalize Case column if needed
if 'Case ID' in df.columns:
    df[case_column] = df['Case ID']
    df = df.drop('Case ID', axis=1)

# Ensure required columns exist
required_cols = [case_column, activity_column, timestamp_column]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' not found in the dataset.")

print(f"Run 2: Original dataset shape: {df.shape}")

# #2. Identify Homonymous Activities
df['ishomonymous'] = df[activity_column].str.endswith(homonymous_suffix).astype(int)
df['BaseActivity'] = df[activity_column]
mask = df['ishomonymous'] == 1
df.loc[mask, 'BaseActivity'] = df.loc[mask, activity_column].str.rstrip(homonymous_suffix)

# #7. Calculate Detection Metrics (BEFORE FIXING)
if label_column in df.columns:
    y_true = (df[label_column].fillna('').astype(str) != '').astype(int)
    y_pred = df['ishomonymous']
    if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
    else:
        precision = recall = f1 = 0.0
else:
    precision = recall = f1 = 0.0
    print("No labels available for metric calculation.")

print("=== Detection Performance Metrics ===")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Precision threshold (>= 0.6) {'met' if precision >= 0.6 else 'not met'}")

# #3. Preprocess Activity Names
def preprocess_activity(act):
    if pd.isna(act):
        return act
    # Remove the activity suffix pattern
    act = re.sub(activity_suffix_pattern, '', str(act))
    # Lowercase if not case_sensitive
    if not case_sensitive:
        act = act.lower()
    # Replace underscores and hyphens with spaces
    act = re.sub(r'[_-]', ' ', act)
    # Collapse multiple spaces
    act = re.sub(r'\s+', ' ', act).strip()
    return act

df['ProcessedActivity'] = df.apply(
    lambda row: preprocess_activity(row['BaseActivity']) if row['ishomonymous'] == 1 else row['BaseActivity'],
    axis=1
)

# #4. Vectorize Activities (only for homonymous)
homo_mask = df['ishomonymous'] == 1
homo_acts = df.loc[homo_mask, 'ProcessedActivity'].dropna().unique()
if len(homo_acts) > 0:
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3), lowercase=False)
    vectors = vectorizer.fit_transform(homo_acts).toarray()
else:
    vectors = np.array([]).reshape(0, 0)

# #5. Cluster Similar Activities
cluster_map = {}
if len(homo_acts) > 1:
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric='cosine',
        linkage=linkage_method,
        distance_threshold=1 - similarity_threshold
    )
    cluster_ids = clustering.fit_predict(vectors)
    cluster_map = dict(zip(homo_acts, cluster_ids))
elif len(homo_acts) == 1:
    cluster_map[homo_acts[0]] = 0

# Map clusters to df
df['cluster'] = -1
if len(cluster_map) > 0:
    df.loc[homo_mask, 'cluster'] = df.loc[homo_mask, 'ProcessedActivity'].map(cluster_map)

# #6. Majority Voting Within Clusters
homo_df = df[homo_mask].copy()
canonical = {}
if len(homo_acts) > 0:
    for cid in homo_df['cluster'].unique():
        if cid != -1:
            cluster_proc = homo_df[homo_df['cluster'] == cid]['ProcessedActivity'].value_counts()
            if len(cluster_proc) > 0:
                canonical[cid] = cluster_proc.index[0]

# Assign Activity_fixed
df['Activity_fixed'] = df[activity_column]
mask_homo = df['ishomonymous'] == 1
df.loc[mask_homo, 'Activity_fixed'] = df.loc[mask_homo, 'cluster'].map(canonical)
# For unmapped (e.g., singletons or low similarity), keep original BaseActivity
unmapped_mask = mask_homo & df['Activity_fixed'].isna()
df.loc[unmapped_mask, 'Activity_fixed'] = df.loc[unmapped_mask, 'BaseActivity']

# #8. Integrity Check
clean_modified = ((df['ishomonymous'] == 0) & (df['BaseActivity'] != df['Activity_fixed'])).sum()
homo_unchanged = ((df['ishomonymous'] == 1) & (df['BaseActivity'] == df['Activity_fixed'])).sum()
homo_corrected = ((df['ishomonymous'] == 1) & (df['BaseActivity'] != df['Activity_fixed'])).sum()

print("Integrity Check:")
print(f"Clean activities modified: {clean_modified} (should be 0)")
print(f"Homonymous activities unchanged: {homo_unchanged}")
print(f"Homonymous activities corrected: {homo_corrected}")

# Standardize timestamp format
df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')

# #9. Save Output
output_columns = [case_column, timestamp_column]
if 'Resource' in df.columns:
    output_columns.append('Resource')
if 'Variant' in df.columns:
    output_columns.append('Variant')
output_columns.extend([activity_column, 'Activity_fixed', 'ishomonymous'])
if label_column in df.columns:
    output_columns.append(label_column)

df[output_columns].to_csv(output_file, index=False)

# #10. Summary Statistics
total_events = len(df)
homonymous_detected = df['ishomonymous'].sum()
unique_before = df[activity_column].nunique()
unique_after = df['Activity_fixed'].nunique()

print(f"Total number of events: {total_events}")
print(f"Number of homonymous events detected: {homonymous_detected}")
print(f"Unique activities before fixing: {unique_before}")
print(f"Unique activities after fixing: {unique_after}")
print(f"Output file path: {output_file}")

# REQUIRED: Print summary
print(f"Run 2: Processed dataset saved to: data/bpic11/bpic11_homonymous_cleaned_run2.csv")
print(f"Run 2: Final dataset shape: {df.shape}")
print(f"Run 2: Dataset: bpic11")
print(f"Run 2: Task type: homonymous")