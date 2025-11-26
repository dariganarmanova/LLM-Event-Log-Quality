# Generated script for Credit-Distorted - Run 1
# Generated on: 2025-11-18T19:17:15.007367
# Model: grok-4-fast

import pandas as pd
import re
from collections import defaultdict
from itertools import combinations
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

# Configuration
input_file = 'data/credit/Credit-Distorted.csv'
output_file = 'data/credit/credit_distorted_cleaned_run1.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
distorted_suffix = ':distorted'
ngram_size = 3
similarity_threshold = 0.8
min_length = 4

df = pd.read_csv(input_file)
print(f"Run 1: Original dataset shape: {df.shape}")

# Normalize column names if needed
if 'Case ID' in df.columns:
    df = df.rename(columns={'Case ID': case_column})

# Ensure required columns exist
required = [case_column, activity_column, timestamp_column]
for col in required:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# Optional columns
has_variant = 'Variant' in df.columns
has_resource = 'Resource' in df.columns
has_label = label_column in df.columns

# Store original activity
df['original_activity'] = df[activity_column].copy()

# Step 2: Identify distorted activities
df['isdistorted'] = df[activity_column].str.endswith(distorted_suffix, na=False).astype(int)
df['BaseActivity'] = df[activity_column].copy()
mask_dist = df['isdistorted'] == 1
df.loc[mask_dist, 'BaseActivity'] = df.loc[mask_dist, 'BaseActivity'].str.replace(distorted_suffix, '', regex=False)

# Step 3: Preprocess activity names
def preprocess(text):
    if pd.isna(text):
        return np.nan
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['ProcessedActivity'] = df['BaseActivity'].apply(preprocess)

# Step 4: Calculate Jaccard N-gram Similarity
def generate_ngrams(text, n):
    if pd.isna(text) or len(text) < n:
        return set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}

def jaccard_similarity(text1, text2, n):
    set1 = generate_ngrams(text1, n)
    set2 = generate_ngrams(text2, n)
    if not set1 or not set2:
        return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))

unique_pro = df['ProcessedActivity'].dropna().unique()
unique_pro = [p for p in unique_pro if len(p) >= min_length]
print(f"Number of unique processed activities (>= {min_length} chars): {len(unique_pro)}")

similar_pairs = []
num_comps = 0
for i, j in combinations(range(len(unique_pro)), 2):
    sim = jaccard_similarity(unique_pro[i], unique_pro[j], ngram_size)
    if sim >= similarity_threshold:
        similar_pairs.append((unique_pro[i], unique_pro[j]))
    num_comps += 1
    if num_comps % 100 == 0:
        print(f"Progress: {num_comps} comparisons done")
print(f"Found {len(similar_pairs)} similar pairs above threshold {similarity_threshold}")

# Step 5: Cluster Similar Activities (Union-Find)
parent = {act: act for act in unique_pro}
rank = {act: 0 for act in unique_pro}

def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(x, y):
    px = find(x)
    py = find(y)
    if px != py:
        if rank[px] > rank[py]:
            parent[py] = px
        elif rank[px] < rank[py]:
            parent[px] = py
        else:
            parent[py] = px
            rank[px] += 1

for a, b in similar_pairs:
    union(a, b)

clusters = defaultdict(list)
for act in unique_pro:
    clusters[find(act)].append(act)

# Step 6: Majority Voting Within Clusters
canonical_map = {}
for root, proc_list in clusters.items():
    mask = df['ProcessedActivity'].isin(proc_list)
    if mask.sum() == 0:
        continue
    originals_count = df.loc[mask, 'original_activity'].value_counts()
    if not originals_count.empty:
        canonical = originals_count.index[0]
        for p in proc_list:
            canonical_map[p] = canonical

df['canonical_activity'] = df['ProcessedActivity'].map(canonical_map)
mask_no_can = df['canonical_activity'].isna()
df.loc[mask_no_can, 'canonical_activity'] = df.loc[mask_no_can, 'original_activity']
df['is_distorted'] = (df['original_activity'] != df['canonical_activity']).astype(int)

# Step 7: Calculate Detection Metrics (BEFORE FIXING)
print("=== Detection Performance Metrics ===")
if has_label:
    def normalize_label(l):
        if pd.isna(l):
            return 0
        return 1 if 'distorted' in str(l).lower() else 0
    y_true = df[label_column].apply(normalize_label).values
    y_pred = df['is_distorted'].values
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    if prec >= 0.6:
        print("✓ Precision threshold (≥ 0.6) met")
    else:
        print("✗ Precision threshold (≥ 0.6) not met")
else:
    print("Precision: 0.0000")
    print("Recall: 0.0000")
    print("F1-Score: 0.0000")
    print("No labels available for metric calculation")

# Step 8: Integrity Check
multi_cluster_count = sum(1 for cl in clusters.values() if len(cl) > 1)
total_distorted = df['is_distorted'].sum()
total_events = len(df)
clean_unchanged = total_events - total_distorted
print("\n=== Integrity Check ===")
print(f"Total distortion clusters detected: {multi_cluster_count}")
print(f"Total activities marked as distorted: {total_distorted}")
print(f"Clean activities that were NOT modified: {clean_unchanged}")

# Step 9: Fix Activities
df['Activity_fixed'] = df['canonical_activity']

# Step 10: Save Output
df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')

output_cols = [case_column, timestamp_column, activity_column, 'Activity_fixed']
if has_variant:
    output_cols.insert(2, 'Variant')
if has_resource:
    pos = 3 if has_variant else 2
    output_cols.insert(pos, 'Resource')
if has_label:
    output_cols.append(label_column)
df_output = df[output_cols].copy()
df_output.to_csv(output_file, index=False)

# Step 11: Summary Statistics
print("\n=== Summary Statistics ===")
print(f"Total number of events: {total_events}")
print(f"Number of distorted events detected: {total_distorted}")
unique_before = df['original_activity'].nunique()
unique_after = df['Activity_fixed'].nunique()
print(f"Unique activities before fixing: {unique_before}")
print(f"Unique activities after fixing: {unique_after}")
reduction = unique_before - unique_after
perc = (reduction / unique_before * 100) if unique_before > 0 else 0
print(f"Activity reduction count and percentage: {reduction} ({perc:.2f}%)")
print(f"Output file path: {output_file}")

print("\nSample of up to 10 transformations:")
changed_samples = df[df['is_distorted'] == 1][['original_activity', 'Activity_fixed']].drop_duplicates().head(10)
if len(changed_samples) > 0:
    for _, row in changed_samples.iterrows():
        print(f"- {row['original_activity']} → {row['Activity_fixed']}")
else:
    print("No transformations applied.")

print(f"\nRun 1: Processed dataset saved to: {output_file}")
print(f"Run 1: Final dataset shape: {df_output.shape}")
print(f"Run 1: Dataset: credit")
print(f"Run 1: Task type: distorted")