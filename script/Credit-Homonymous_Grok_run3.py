# Generated script for Credit-Homonymous - Run 3
# Generated on: 2025-11-18T19:28:10.318773
# Model: grok-4-fast

import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import defaultdict

# Configuration parameters
input_file = 'data/credit/Credit-Homonymous.csv'
output_file = 'data/credit/credit_homonymous_cleaned_run3.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
homonymous_suffix = ':homonymous'
time_threshold = 2.0
require_same_resource = False
min_matching_events = 2
max_mismatches = 1
activity_suffix_pattern = r'(_signed\d*|_\d+)$'
similarity_threshold = 0.8
case_sensitive = False
use_fuzzy_matching = False
linkage_method = 'average'

# Load the data
df = pd.read_csv(input_file)
original_shape = df.shape
print(f"Run 3: Original dataset shape: {original_shape}")

# Ensure required columns exist (assume they do, but check)
required_columns = [case_column, activity_column, timestamp_column]
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' not found in the dataset.")

# Normalize CaseID to Case if needed
if 'CaseID' in df.columns:
    df = df.rename(columns={'CaseID': case_column})

# Step 2: Identify Homonymous Activities
df['ishomonymous'] = df[activity_column].str.endswith(homonymous_suffix).astype(int)
df['BaseActivity'] = df[activity_column].str.rsplit(homonymous_suffix, n=1).str[0]

# Step 3: Preprocess Activity Names
def preprocess_activity(base_act, pattern, case_sensitive_flag):
    # Remove activity suffix pattern if it matches at the end
    if re.search(pattern + '$', base_act):
        base_act = re.sub(pattern + '$', '', base_act)
    # Replace underscores and hyphens with spaces
    base_act = re.sub(r'[_-]', ' ', base_act)
    # Collapse multiple spaces and strip
    base_act = re.sub(r'\s+', ' ', base_act).strip()
    # Lowercase if not case sensitive
    if not case_sensitive_flag:
        base_act = base_act.lower()
    return base_act

df['ProcessedActivity'] = df['BaseActivity'].apply(
    lambda x: preprocess_activity(x, activity_suffix_pattern, case_sensitive)
)

# Step 7: Calculate Detection Metrics (BEFORE FIXING)
if label_column in df.columns:
    y_true = ((df[label_column].notna()) & (df[label_column].astype(str).str.strip() != '')).astype(int)
    y_pred = df['ishomonymous']
    if y_true.sum() > 0 and y_pred.sum() > 0:
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
    else:
        prec = rec = f1 = 0.0
else:
    prec = rec = f1 = 0.0
    print("No labels available for metric calculation.")

print("=== Detection Performance Metrics ===")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-Score: {f1:.4f}")
met = "met" if prec >= 0.6 else "not met"
print(f"Precision threshold (≥ 0.6) {met}")

# Step 4 & 5: Vectorize and Cluster Homonymous Activities Only
hom_mask = df['ishomonymous'] == 1
hom_proc_unique = df.loc[hom_mask, 'ProcessedActivity'].dropna().unique()
df['Activity_fixed'] = df['Activity']  # Initialize with original

if len(hom_proc_unique) > 0:
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3), lowercase=False)
    X = vectorizer.fit_transform(hom_proc_unique).toarray()
    
    if X.shape[0] > 0:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric='cosine',
            linkage=linkage_method,
            distance_threshold=1 - similarity_threshold
        )
        cluster_labels = clustering.fit_predict(X)
        
        # Map processed to cluster
        processed_to_cluster = dict(zip(hom_proc_unique, cluster_labels))
        
        # Get homonymous rows
        hom_df = df[hom_mask].copy()
        hom_df['cluster'] = hom_df['ProcessedActivity'].map(processed_to_cluster)
        
        # Step 6: Majority Voting Within Clusters (using most frequent ProcessedActivity as canonical)
        cluster_canonical = {}
        for cluster_id in hom_df['cluster'].unique():
            if pd.isna(cluster_id):
                continue
            cluster_data = hom_df[hom_df['cluster'] == cluster_id]['ProcessedActivity']
            if len(cluster_data) > 0:
                mode_proc = cluster_data.mode()
                canonical = mode_proc[0] if len(mode_proc) > 0 else cluster_data.iloc[0]
                cluster_canonical[cluster_id] = canonical
        
        # Assign canonical to homonymous rows
        hom_df['Activity_fixed'] = hom_df['cluster'].map(cluster_canonical)
        
        # Update original df
        df.loc[hom_mask, 'Activity_fixed'] = hom_df['Activity_fixed']

# Step 8: Integrity Check
clean_modified = ((df['ishomonymous'] == 0) & (df['BaseActivity'] != df['Activity_fixed'])).sum()
hom_unchanged = ((df['ishomonymous'] == 1) & (df['BaseActivity'] == df['Activity_fixed'])).sum()
hom_corrected = ((df['ishomonymous'] == 1) & (df['BaseActivity'] != df['Activity_fixed'])).sum()

print("Integrity Check:")
print(f"Clean activities modified: {clean_modified} (should be 0)")
print(f"Homonymous activities unchanged: {hom_unchanged}")
print(f"Homonymous activities corrected: {hom_corrected}")

# Step 9: Prepare Output (Standardize Timestamp)
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')

# Select output columns
output_columns = [case_column, timestamp_column, activity_column, 'Activity_fixed', 'ishomonymous']
if 'Resource' in df.columns:
    output_columns.insert(2, 'Resource')
if 'Variant' in df.columns:
    res_idx = 3 if 'Resource' in df.columns else 2
    output_columns.insert(res_idx, 'Variant')
if label_column in df.columns:
    output_columns.append(label_column)

df_output = df[output_columns].copy()
df_output.to_csv(output_file, index=False)

# Step 10: Summary Statistics
print(f"Total number of events: {len(df)}")
print(f"Number of homonymous events detected: {df['ishomonymous'].sum()}")
print(f"Unique activities before fixing: {df[activity_column].nunique()}")
print(f"Unique activities after fixing: {df['Activity_fixed'].nunique()}")
print(f"Output file path: {output_file}")

# Required prints
print(f"Run 3: Processed dataset saved to: {output_file}")
print(f"Run 3: Final dataset shape: {df.shape}")
print(f"Run 3: Dataset: credit")
print(f"Run 3: Task type: homonymous")