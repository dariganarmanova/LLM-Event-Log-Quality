# Generated script for Credit-Homonymous - Run 1
# Generated on: 2025-11-18T19:26:36.679486
# Model: grok-4-fast

import pandas as pd
import re
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

# Configuration parameters
input_file = 'data/credit/Credit-Homonymous.csv'
output_file = 'data/credit/credit_homonymous_cleaned_run1.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
homonymous_suffix = ':homonymous'
time_threshold = 2.0
require_same_resource = False
min_matching_events = 2
max_mismatches = 1
activity_suffix_pattern = r"(_signed\d*|_\d+)$"
similarity_threshold = 0.8
case_sensitive = False
use_fuzzy_matching = False
linkage_method = 'average'

print(f"Run 1: Original dataset shape: {pd.read_csv(input_file).shape}")

# Load the data
df = pd.read_csv(input_file)

# Normalize column names if needed
if 'Case ID' in df.columns:
    df.rename(columns={'Case ID': case_column}, inplace=True)

# Ensure required columns exist
required_cols = [case_column, activity_column, timestamp_column]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' not found in the dataset.")

# Format timestamp
df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')

# Step 2: Identify Homonymous Activities
df['ishomonymous'] = 0
df.loc[df[activity_column].str.endswith(homonymous_suffix, na=False), 'ishomonymous'] = 1
df['BaseActivity'] = df[activity_column]
df.loc[df['ishomonymous'] == 1, 'BaseActivity'] = df.loc[df['ishomonymous'] == 1, activity_column].str.replace(homonymous_suffix, '', regex=False)

# Step 3: Preprocess Activity Names
def normalize_activity(act, case_sensitive_local=False):
    if pd.isna(act):
        return act
    act_lower = act if case_sensitive_local else act.lower()
    act_norm = re.sub(r'[_-]', ' ', act_lower)
    act_norm = re.sub(r'\s+', ' ', act_norm).strip()
    return act_norm

df['ProcessedActivity'] = df['BaseActivity'].apply(lambda x: normalize_activity(x, case_sensitive))

# Step 4: Vectorize Activities (only for homonymous)
hom_df = df[df['ishomonymous'] == 1].copy()
if len(hom_df) > 0:
    unique_list = list(hom_df['ProcessedActivity'].dropna().unique())
    if len(unique_list) > 0:
        vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
        X = vectorizer.fit_transform(unique_list).toarray()
        # Step 5: Cluster Similar Activities
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric='cosine',
            linkage=linkage_method,
            distance_threshold=1 - similarity_threshold
        )
        cluster_id = clustering.fit_predict(X)
        unique_to_cluster = dict(zip(unique_list, cluster_id))
        # Step 6: Majority Voting Within Clusters
        cluster_bases = defaultdict(list)
        for _, row in hom_df.iterrows():
            if pd.notna(row['ProcessedActivity']):
                cl = unique_to_cluster[row['ProcessedActivity']]
                cluster_bases[cl].append(row['BaseActivity'])
        cl_to_canonical = {}
        for cl, bases in cluster_bases.items():
            base_counts = Counter(bases)
            mode_base = base_counts.most_common(1)[0][0]
            stripped = re.sub(activity_suffix_pattern, '', mode_base, flags=re.IGNORECASE if not case_sensitive else 0)
            canonical = normalize_activity(stripped, case_sensitive)
            cl_to_canonical[cl] = canonical
        # Assign to df
        df['cluster'] = df['ProcessedActivity'].map(unique_to_cluster)
        df['Activity_fixed'] = df[activity_column]
        mask_hom = df['ishomonymous'] == 1
        df.loc[mask_hom & df['cluster'].notna(), 'Activity_fixed'] = df.loc[mask_hom & df['cluster'].notna(), 'cluster'].map(cl_to_canonical)
        # For homonymous without cluster (e.g., NaN processed), set to normalized BaseActivity
        df.loc[mask_hom & df['cluster'].isna(), 'Activity_fixed'] = df.loc[mask_hom & df['cluster'].isna(), 'BaseActivity'].apply(lambda x: normalize_activity(re.sub(activity_suffix_pattern, '', x, flags=re.IGNORECASE if not case_sensitive else 0), case_sensitive))
    else:
        # No unique processed, just normalize and strip for homonymous
        df['Activity_fixed'] = df[activity_column]
        mask_hom = df['ishomonymous'] == 1
        df.loc[mask_hom, 'Activity_fixed'] = df.loc[mask_hom, 'BaseActivity'].apply(lambda x: normalize_activity(re.sub(activity_suffix_pattern, '', x, flags=re.IGNORECASE if not case_sensitive else 0), case_sensitive))
else:
    df['Activity_fixed'] = df[activity_column]

# For non-homonymous, ensure Activity_fixed == Activity
mask_clean = df['ishomonymous'] == 0
df.loc[mask_clean, 'Activity_fixed'] = df.loc[mask_clean, activity_column]

# Drop temporary columns
df.drop(['ProcessedActivity', 'cluster'], axis=1, inplace=True, errors='ignore')

# Step 7: Calculate Detection Metrics (BEFORE FIXING - but since ishomonymous is detection)
print("=== Detection Performance Metrics ===")
if label_column in df.columns:
    y_true = ((df[label_column].notna()) & (df[label_column] != '')).astype(int)
    y_pred = df['ishomonymous']
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
print(f"Precision threshold (≥ 0.6) {'met' if prec >= 0.6 else 'not met'}")

# Step 8: Integrity Check
clean_modified = ((df['ishomonymous'] == 0) & (df['Activity_fixed'] != df[activity_column])).sum()
hom_unchanged = ((df['ishomonymous'] == 1) & (df['Activity_fixed'] == df['BaseActivity'])).sum()
hom_corrected = df['ishomonymous'].sum() - hom_unchanged
print("=== Integrity Check ===")
print(f"Clean activities modified: {clean_modified} (should be 0)")
print(f"Homonymous activities unchanged: {hom_unchanged}")
print(f"Homonymous activities corrected: {hom_corrected}")

# Step 9: Save Output (preserve all original columns + new)
df.to_csv(output_file, index=False)

# Step 10: Summary Statistics
total_events = len(df)
homonymous_detected = df['ishomonymous'].sum()
unique_before = df[activity_column].nunique()
unique_after = df['Activity_fixed'].nunique()
print("=== Summary Statistics ===")
print(f"Total number of events: {total_events}")
print(f"Number of homonymous events detected: {homonymous_detected}")
print(f"Unique activities before fixing: {unique_before}")
print(f"Unique activities after fixing: {unique_after}")
print(f"Output file path: {output_file}")

# Drop BaseActivity as it's temporary
df.drop('BaseActivity', axis=1, inplace=True, errors='ignore')

# REQUIRED prints
print(f"Run 1: Processed dataset saved to: data/credit/credit_homonymous_cleaned_run1.csv")
print(f"Run 1: Final dataset shape: {df.shape}")
print(f"Run 1: Dataset: credit")
print(f"Run 1: Task type: homonymous")