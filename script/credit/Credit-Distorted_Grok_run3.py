# Generated script for Credit-Distorted - Run 3
# Generated on: 2025-11-18T19:18:47.918496
# Model: grok-4-fast

import pandas as pd
import re
from itertools import combinations
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration
input_file = 'data/credit/Credit-Distorted.csv'
output_file = 'data/credit/credit_distorted_cleaned_run3.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
distorted_suffix = ':distorted'
ngram_size = 3
similarity_threshold = 0.56
min_length = 4

print(f"Run 3: Original dataset shape: {pd.read_csv(input_file).shape}")

# Step 1: Load CSV
df = pd.read_csv(input_file)

# Normalize column names if needed
if 'CaseID' in df.columns:
    df.rename(columns={'CaseID': 'Case'}, inplace=True)

# Ensure required columns exist
required_cols = [case_column, activity_column, timestamp_column]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' not found in the dataset.")

# Store original activity
df['original_activity'] = df[activity_column].copy()

# Step 2: Identify Distorted Activities
df['isdistorted'] = 0
mask_dist = df[activity_column].str.endswith(distorted_suffix, na=False)
df.loc[mask_dist, 'isdistorted'] = 1
df['BaseActivity'] = df[activity_column]
df.loc[mask_dist, 'BaseActivity'] = df.loc[mask_dist, activity_column].str.rstrip(distorted_suffix)
df['original_base'] = df['BaseActivity'].copy()

# Step 3: Preprocess Activity Names
def preprocess_activity(text):
    if pd.isna(text):
        return ''
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = ' '.join(text.split())
    return text.strip()

df['ProcessedActivity'] = df['original_base'].apply(preprocess_activity)

# Step 4: Calculate Jaccard N-gram Similarity
def generate_ngrams(text, n=ngram_size):
    if len(text) < n:
        return set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}

def jaccard_similarity(set1, set2):
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

# Get unique processed activities for comparison
unique_processed = [p for p in df['ProcessedActivity'].unique() if len(p) >= min_length and p]
similar_pairs = []
num_comparisons = 0
for i, j in combinations(range(len(unique_processed)), 2):
    a = unique_processed[i]
    b = unique_processed[j]
    set_a = generate_ngrams(a)
    set_b = generate_ngrams(b)
    sim = jaccard_similarity(set_a, set_b)
    if sim >= similarity_threshold:
        similar_pairs.append((a, b))
    num_comparisons += 1
    if num_comparisons % 100 == 0:
        print(f"Progress: {num_comparisons} comparisons processed")

# Step 5: Cluster Similar Activities (Union-Find)
def find(parent, x):
    if parent[x] != x:
        parent[x] = find(parent, parent[x])
    return parent[x]

def union(parent, x, y):
    px = find(parent, x)
    py = find(parent, y)
    if px != py:
        parent[px] = py

if unique_processed:
    parent = {act: act for act in unique_processed}
    for a, b in similar_pairs:
        union(parent, a, b)

    # Group into clusters
    clusters = {}
    for act in unique_processed:
        root = find(parent, act)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(act)

    # Keep only clusters with 2 or more members
    multi_clusters = {k: v for k, v in clusters.items() if len(v) >= 2}
else:
    multi_clusters = {}

# Step 6: Majority Voting Within Clusters
canonical_mapping = {}
for root, cluster_procs in multi_clusters.items():
    mask = df['ProcessedActivity'].isin(cluster_procs)
    if mask.sum() == 0:
        continue
    freq = df.loc[mask, 'original_base'].value_counts()
    canonical = freq.index[0]
    for proc in cluster_procs:
        canonical_mapping[proc] = canonical

# Assign canonical for all rows
df['canonical_activity'] = df['original_base']
for proc, can in canonical_mapping.items():
    mask = df['ProcessedActivity'] == proc
    df.loc[mask, 'canonical_activity'] = can

# Mark distorted
df['is_distorted'] = (df['original_base'] != df['canonical_activity']).astype(int)

# Step 7: Calculate Detection Metrics (BEFORE FIXING)
print("=== Detection Performance Metrics ===")
if label_column in df.columns:
    def normalize_label(l):
        if pd.isna(l):
            return 0
        l_str = str(l).lower()
        if 'distorted' in l_str:
            return 1
        return 0

    y_true = df[label_column].apply(normalize_label)
    y_pred = df['is_distorted']
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    prec_met = "✓" if prec >= 0.6 else "✗"
    print(f"{prec_met} Precision threshold (≥ 0.6) met/not met")
else:
    print("Precision: 0.0000")
    print("Recall: 0.0000")
    print("F1-Score: 0.0000")
    print("No labels available for metric calculation")

# Step 8: Integrity Check
num_clusters = len(multi_clusters)
num_distorted = df['is_distorted'].sum()
num_clean = len(df) - num_distorted
num_fixed = num_distorted  # same as marked distorted
print(f"Total distortion clusters detected: {num_clusters}")
print(f"Total activities marked as distorted: {num_distorted}")
print(f"Activities to be fixed: {num_fixed}")
# Verify no changes to clean
clean_mask = df['is_distorted'] == 0
assert all(df.loc[clean_mask, 'original_base'] == df.loc[clean_mask, 'canonical_activity']), "Clean activities were modified!"

# Step 9: Fix Activities
df['Activity_fixed'] = df['canonical_activity']

# Step 10: Save Output
# Standardize timestamp
try:
    df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
except Exception as e:
    print(f"Warning: Could not standardize timestamps: {e}")

# Select output columns
output_cols = [case_column, timestamp_column, activity_column, 'Activity_fixed']
if 'Variant' in df.columns:
    output_cols.insert(2, 'Variant')
if 'Resource' in df.columns:
    insert_pos = 3 if 'Variant' in df.columns else 2
    output_cols.insert(insert_pos, 'Resource')
if label_column in df.columns:
    output_cols.append(label_column)

df_output = df[output_cols].copy()
df_output.to_csv(output_file, index=False)

# Step 11: Summary Statistics
total_events = len(df)
distorted_detected = num_distorted
unique_before = df[activity_column].nunique()
unique_after = df['Activity_fixed'].nunique()
reduction_count = unique_before - unique_after
reduction_pct = (reduction_count / unique_before * 100) if unique_before > 0 else 0

print(f"Total number of events: {total_events}")
print(f"Number of distorted events detected: {distorted_detected}")
print(f"Unique activities before fixing: {unique_before}")
print(f"Unique activities after fixing: {unique_after}")
print(f"Activity reduction count: {reduction_count}")
print(f"Activity reduction percentage: {reduction_pct:.2f}%")
print(f"Output file path: {output_file}")

# Sample transformations
print("Sample transformations (up to 10):")
changed_mask = df['is_distorted'] == 1
if changed_mask.sum() > 0:
    samples = df.loc[changed_mask, ['original_activity', 'Activity_fixed']].drop_duplicates().head(10)
    for _, row in samples.iterrows():
        print(f"{row['original_activity']} → {row['Activity_fixed']}")
else:
    print("No transformations applied.")

print(f"Run 3: Processed dataset saved to: data/credit/credit_distorted_cleaned_run3.csv")
print(f"Run 3: Final dataset shape: {df_output.shape}")
print(f"Run 3: Dataset: credit")
print(f"Run 3: Task type: distorted")