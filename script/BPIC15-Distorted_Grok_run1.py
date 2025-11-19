# Generated script for BPIC15-Distorted - Run 1
# Generated on: 2025-11-18T21:25:22.114383
# Model: grok-4-fast

import pandas as pd
import re
from collections import defaultdict
from itertools import combinations
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration parameters
time_threshold = 2.0
require_same_resource = False
min_matching_events = 2
max_mismatches = 1
activity_suffix_pattern = r"(_signed\d*|_\d+)$"
similarity_threshold = 0.56
case_sensitive = False
use_fuzzy_matching = False

# Task-specific parameters
ngram_size = 3
min_length = 4
input_file = 'data/bpic15/BPIC15-Distorted.csv'
output_file = 'data/bpic15/bpic15_distorted_cleaned_run1.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
distorted_suffix = ':distorted'

# Load the data
df = pd.read_csv(input_file)
print(f"Run 1: Original dataset shape: {df.shape}")

# Handle column name variations
column_mapping = {
    'Case ID': 'Case',
    'case': 'Case',
    'Activity ID': 'Activity',
    'activity': 'Activity',
    'Complete Timestamp': 'Timestamp',
    'timestamp': 'Timestamp'
}
df.rename(columns=column_mapping, inplace=True)

# Ensure required columns exist
required_cols = ['Case', 'Activity', 'Timestamp']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' not found in the dataset.")

# Optional columns
has_variant = 'Variant' in df.columns
has_label = label_column in df.columns

# Store original Activity
df['original_activity'] = df['Activity']

# Step 2: Identify Distorted Activities
df['isdistorted'] = df['Activity'].str.endswith(distorted_suffix, na=False).astype(int)
df['BaseActivity'] = df['Activity'].str.rstrip(distorted_suffix).str.strip()

# Step 3: Preprocess Activity Names
def preprocess_activity(activity):
    if pd.isna(activity):
        return ''
    lower = activity.lower()
    # Remove non-alphanumeric except spaces
    cleaned = re.sub(r'[^a-z0-9\s]', '', lower)
    # Collapse multiple spaces and strip
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

df['ProcessedActivity'] = df['BaseActivity'].apply(preprocess_activity)

# Get unique processed activities
unique_processed = df['ProcessedActivity'].dropna().unique()
unique_processed = [p for p in unique_processed if p]  # Remove empty strings

# Filter long processed for similarity computation
long_processed = [p for p in unique_processed if len(p) >= min_length]

# Step 4: Calculate Jaccard N-gram Similarity
def generate_ngrams(text, n):
    if len(text) < n:
        return set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}

def jaccard_similarity(text1, text2, n):
    set1 = generate_ngrams(text1, n)
    set2 = generate_ngrams(text2, n)
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

print("Computing similarities...")
similar_pairs = []
num_comparisons = 0
for i in range(len(long_processed)):
    for j in range(i + 1, len(long_processed)):
        p1 = long_processed[i]
        p2 = long_processed[j]
        sim = jaccard_similarity(p1, p2, ngram_size)
        if sim >= similarity_threshold:
            similar_pairs.append((p1, p2))
        num_comparisons += 1
        if num_comparisons % 100 == 0:
            print(f"Processed {num_comparisons} comparisons")

# Step 5: Cluster Similar Activities (Union-Find)
parent = {p: p for p in long_processed}
rank = {p: 0 for p in long_processed}

def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(x, y):
    px = find(x)
    py = find(y)
    if px == py:
        return
    if rank[px] < rank[py]:
        parent[px] = py
    elif rank[px] > rank[py]:
        parent[py] = px
    else:
        parent[py] = px
        rank[px] += 1

for p1, p2 in similar_pairs:
    union(p1, p2)

# Build clusters for long processed
clusters = defaultdict(list)
for p in long_processed:
    root = find(p)
    clusters[root].append(p)

# Singleton clusters for short processed
short_processed = [p for p in unique_processed if len(p) < min_length]
for p in short_processed:
    clusters[p].append(p)

# Step 6: Majority Voting Within Clusters
canonical_map = {}
for root, cluster_procs in clusters.items():
    mask = df['ProcessedActivity'].isin(cluster_procs)
    cluster_df = df[mask]
    if len(cluster_df) == 0:
        continue
    freq = cluster_df['BaseActivity'].value_counts()
    canonical = freq.index[0]
    for base in cluster_df['BaseActivity'].unique():
        canonical_map[base] = canonical

# Assign canonical and mark distorted
df['canonical_activity'] = df['BaseActivity'].map(canonical_map).fillna(df['BaseActivity'])
df['is_distorted'] = (df['BaseActivity'] != df['canonical_activity']).astype(int)

# Step 7: Calculate Detection Metrics (BEFORE FIXING)
print("=== Detection Performance Metrics ===")
if has_label:
    def normalize_label(l):
        if pd.isna(l):
            return 0
        return 1 if 'distorted' in str(l).lower() else 0
    y_true = df[label_column].apply(normalize_label)
    y_pred = df['is_distorted']
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    prec_ok = "✓" if precision >= 0.6 else "✗"
    print(f"{prec_ok} Precision threshold (≥ 0.6) met/not met")
else:
    print("Precision: 0.0000")
    print("Recall: 0.0000")
    print("F1-Score: 0.0000")
    print("No labels available for metric calculation")

# Step 8: Integrity Check
num_distortion_clusters = 0
for root, cluster_procs in clusters.items():
    mask = df['ProcessedActivity'].isin(cluster_procs)
    cluster_df = df[mask]
    if len(cluster_df['BaseActivity'].unique()) > 1:
        num_distortion_clusters += 1
total_distorted = df['is_distorted'].sum()
total_clean = len(df) - total_distorted
print(f"Total distortion clusters detected: {num_distortion_clusters}")
print(f"Total activities marked as distorted: {total_distorted}")
print(f"Activities to be fixed: {total_distorted}")
print(f"Clean activities that were NOT modified: {total_clean}")

# Step 9: Fix Activities
df['Activity_fixed'] = df['canonical_activity']

# Step 10: Save Output
# Format timestamp
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')

# Save full processed df
df.to_csv(output_file, index=False)

# Step 11: Summary Statistics
total_events = len(df)
num_distorted_detected = total_distorted
unique_before = len(df['Activity'].unique())
unique_after = len(df['Activity_fixed'].unique())
reduction = unique_before - unique_after
percentage = (reduction / unique_before * 100) if unique_before > 0 else 0.0
print(f"Total number of events: {total_events}")
print(f"Number of distorted events detected: {num_distorted_detected}")
print(f"Unique activities before fixing: {unique_before}")
print(f"Unique activities after fixing: {unique_after}")
print(f"Activity reduction count: {reduction}")
print(f"Activity reduction percentage: {percentage:.2f}%")
print(f"Output file path: {output_file}")

# Sample transformations
transformations = []
for base in sorted(canonical_map.keys()):
    can = canonical_map[base]
    if base != can:
        transformations.append(f"{base} → {can}")
print("Sample transformations (up to 10):")
for t in transformations[:10]:
    print(t)
if len(transformations) > 10:
    print(f"... and {len(transformations) - 10} more")

print(f"Run 1: Processed dataset saved to: {output_file}")
print(f"Run 1: Final dataset shape: {df.shape}")
print(f"Run 1: Dataset: bpic15")
print(f"Run 1: Task type: distorted")