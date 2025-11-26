# Generated script for BPIC11-Homonymous - Run 1
# Generated on: 2025-11-18T22:37:07.029054
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
activity_suffix_pattern = r"(_signed\d*|_\d+)$"
similarity_threshold = 0.8
case_sensitive = False
use_fuzzy_matching = False
linkage_method = 'average'
homonymous_suffix = ':homonymous'
run_number = 1
dataset_name = 'bpic11'
task_type = 'homonymous'
input_file = 'data/bpic11/BPIC11-Homonymous.csv'
output_file = 'data/bpic11/bpic11_homonymous_cleaned_run1.csv'

# Load the data
df = pd.read_csv(input_file)
original_shape = df.shape
print(f"Run {run_number}: Original dataset shape: {original_shape}")

# Ensure required columns exist
required_columns = ['Case', 'Activity', 'Timestamp']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# Normalize CaseID to Case if needed
if 'Case ID' in df.columns:
    df.rename(columns={'Case ID': 'Case'}, inplace=True)

# #2. Identify Homonymous Activities
df['ishomonymous'] = df['Activity'].str.endswith(homonymous_suffix).astype(int)
df['BaseActivity'] = df['Activity']
mask_hom = df['Activity'].str.endswith(homonymous_suffix)
df.loc[mask_hom, 'BaseActivity'] = df.loc[mask_hom, 'Activity'].str[:-len(homonymous_suffix)]

# #3. Preprocess Activity Names
def preprocess_activity(activity):
    cleaned = re.sub(activity_suffix_pattern, '', activity)
    if not case_sensitive:
        cleaned = cleaned.lower()
    cleaned = re.sub(r'[_-]', ' ', cleaned)
    cleaned = ' '.join(cleaned.split())
    return cleaned

df['ProcessedActivity'] = df['BaseActivity'].apply(preprocess_activity)

# #7. Calculate Detection Metrics (BEFORE FIXING)
has_label = 'label' in df.columns
if has_label:
    y_true = (df['label'].notna() & (df['label'] != '')).astype(int)
    y_pred = df['ishomonymous'].astype(int)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
else:
    prec = 0.0
    rec = 0.0
    f1 = 0.0
    print("No labels available for metric calculation.")

print("=== Detection Performance Metrics ===")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Precision threshold (>= 0.6) {'met' if prec >= 0.6 else 'not met'}")

# #4. Vectorize Activities
unique_processed = df['ProcessedActivity'].unique()
if len(unique_processed) == 0:
    unique_processed = np.array([''])
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
vectors = vectorizer.fit_transform(unique_processed).toarray()

# #5. Cluster Similar Activities
clustering = AgglomerativeClustering(
    n_clusters=None,
    metric='cosine',
    linkage=linkage_method,
    distance_threshold=1 - similarity_threshold
)
cluster_ids = clustering.fit_predict(vectors)

# Map processed to cluster
processed_to_cluster = dict(zip(unique_processed, cluster_ids))

# #6. Majority Voting Within Clusters
cluster_to_canonical = {}
for cluster in set(cluster_ids):
    mask = (df['ishomonymous'] == 1) & (df['ProcessedActivity'].map(processed_to_cluster) == cluster)
    if mask.sum() == 0:
        continue
    bases_in_cluster = df.loc[mask, 'BaseActivity'].value_counts()
    if len(bases_in_cluster) > 0:
        mode_base = bases_in_cluster.index[0]
        canonical = preprocess_activity(mode_base)
        cluster_to_canonical[cluster] = canonical

# Assign Activity_fixed
df['Activity_fixed'] = df['Activity']
mask_hom = df['ishomonymous'] == 1
df.loc[mask_hom, 'Activity_fixed'] = df.loc[mask_hom, 'ProcessedActivity'].map(processed_to_cluster).map(cluster_to_canonical)

# For clusters without canonical, they remain as original (but shouldn't happen for hom)
df.loc[(mask_hom) & (df['Activity_fixed'] == df['Activity']), 'Activity_fixed'] = df.loc[(mask_hom) & (df['Activity_fixed'] == df['Activity']), 'BaseActivity']

# #8. Integrity Check
clean_modified = ((df['ishomonymous'] == 0) & (df['Activity'] != df['Activity_fixed'])).sum()
hom_unchanged = ((df['ishomonymous'] == 1) & (df['BaseActivity'] == df['Activity_fixed'])).sum()
hom_corrected = ((df['ishomonymous'] == 1) & (df['BaseActivity'] != df['Activity_fixed'])).sum()

print("Integrity Check:")
print(f"Clean activities modified: {clean_modified} (should be 0)")
print(f"Homonymous activities unchanged: {hom_unchanged}")
print(f"Homonymous activities corrected: {hom_corrected}")

# #9. Save Output
# Format timestamp
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')

# Select columns, preserve optional
output_cols = ['Case', 'Timestamp']
if 'Resource' in df.columns:
    output_cols.append('Resource')
if 'Variant' in df.columns:
    output_cols.append('Variant')
output_cols += ['Activity', 'Activity_fixed', 'ishomonymous']
if 'label' in df.columns:
    output_cols.append('label')

output_df = df[output_cols].copy()

# #10. Summary Statistics
print(f"Total number of events: {len(df)}")
print(f"Number of homonymous events detected: {df['ishomonymous'].sum()}")
print(f"Unique activities before fixing: {df['Activity'].nunique()}")
print(f"Unique activities after fixing: {df['Activity_fixed'].nunique()}")
print(f"Output file path: {output_file}")

# Save
output_df.to_csv(output_file, index=False)

# Final prints
print(f"Run {run_number}: Processed dataset saved to: {output_file}")
print(f"Run {run_number}: Final dataset shape: {output_df.shape}")
print(f"Run {run_number}: Dataset: {dataset_name}")
print(f"Run {run_number}: Task type: {task_type}")