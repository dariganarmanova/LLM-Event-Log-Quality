# Generated script for Pub-Distorted - Run 1
# Generated on: 2025-11-18T18:30:00.693483
# Model: grok-4-fast

import pandas as pd
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration
input_file = 'data/pub/Pub-Distorted.csv'
output_file = 'data/pub/pub_distorted_cleaned_run1.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
distorted_suffix = ':distorted'
ngram_size = 3
similarity_threshold = 0.8
min_length = 4

print(f"Run 1: Original dataset shape: {pd.read_csv(input_file).shape}")

# Step 1: Load CSV
df = pd.read_csv(input_file)

# Handle column naming variations
if 'CaseID' in df.columns and case_column not in df.columns:
    df[case_column] = df.pop('CaseID')

required_cols = [case_column, activity_column, timestamp_column]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' not found in the dataset.")

# Store original Activity
df['original_activity'] = df[activity_column]

# Step 2: Identify Distorted Activities
df['isdistorted'] = df[activity_column].str.endswith(distorted_suffix).astype(int)
df['BaseActivity'] = df[activity_column].str.rstrip(distorted_suffix)

# Step 3: Preprocess Activity Names
def preprocess_activity(activity):
    if pd.isna(activity):
        return ''
    activity = str(activity).lower()
    # Keep only alphanumeric and spaces
    activity = ''.join(c for c in activity if c.isalnum() or c == ' ')
    # Collapse multiple spaces and strip
    activity = ' '.join(activity.split())
    return activity

df['ProcessedActivity'] = df['BaseActivity'].apply(preprocess_activity)

# Step 4: Calculate Jaccard N-gram Similarity
def generate_ngrams(text, n=ngram_size):
    if len(text) < n:
        return set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}

def jaccard_similarity(text1, text2):
    set1 = generate_ngrams(text1)
    set2 = generate_ngrams(text2)
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

# Get unique processed activities
unique_processed = df['ProcessedActivity'].dropna().unique()
unique_long = [u for u in unique_processed if len(u) >= min_length]

# Find similar pairs
similar_pairs = []
num_comparisons = 0
for i in range(len(unique_long)):
    for j in range(i + 1, len(unique_long)):
        sim = jaccard_similarity(unique_long[i], unique_long[j])
        if sim >= similarity_threshold:
            similar_pairs.append((unique_long[i], unique_long[j]))
        num_comparisons += 1
        if num_comparisons % 100 == 0:
            print(f"Progress: {num_comparisons} comparisons processed")

# Step 5: Cluster Similar Activities (Union-Find)
parent = {act: act for act in unique_long}
rank = {act: 0 for act in unique_long}

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

for pair in similar_pairs:
    union(pair[0], pair[1])

# Build clusters for long activities
clusters = defaultdict(list)
for act in unique_long:
    root = find(act)
    clusters[root].append(act)

all_clusters = list(clusters.values())

# Add short processed as singletons
short_processed = [u for u in unique_processed if len(u) < min_length and pd.notna(u)]
for sp in short_processed:
    all_clusters.append([sp])

# Step 6: Majority Voting Within Clusters
canonical_map = {}
distortion_clusters = 0

for cluster_procs in all_clusters:
    processed_set = set(cluster_procs)
    cluster_df = df[df['ProcessedActivity'].isin(processed_set)]
    if len(cluster_df) == 0:
        continue
    original_counts = cluster_df['original_activity'].value_counts()
    num_originals = len(original_counts)
    num_procs = len(cluster_procs)
    if num_procs > 1 or num_originals > 1:
        distortion_clusters += 1
        canonical = original_counts.index[0]
        for orig in original_counts.index:
            canonical_map[orig] = canonical

# Assign canonical for all rows
df['canonical_activity'] = df['original_activity'].map(canonical_map).fillna(df['original_activity'])
df['is_distorted'] = (df['canonical_activity'] != df['original_activity']).astype(int)

# Step 7: Calculate Detection Metrics (BEFORE FIXING)
has_labels = label_column in df.columns and not df[label_column].isna().all()
if has_labels:
    def normalize_label(label):
        if pd.isna(label):
            return 0
        return 1 if 'distorted' in str(label).lower() else 0

    y_true = df[label_column].apply(normalize_label)
    y_pred = df['is_distorted']
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
else:
    prec = rec = f1 = 0.0

print("=== Detection Performance Metrics ===")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-Score: {f1:.4f}")
if prec >= 0.6:
    print("✓ Precision threshold (≥ 0.6) met")
else:
    print("✗ Precision threshold (≥ 0.6) not met")
if not has_labels:
    print("No labels available for metric calculation")

# Step 8: Integrity Check
print(f"Total distortion clusters detected: {distortion_clusters}")
print(f"Total activities marked as distorted: {df['is_distorted'].sum()}")
print(f"Activities to be fixed: {df['is_distorted'].sum()}")

# Step 9: Fix Activities
df['Activity_fixed'] = df['canonical_activity']

# Step 10: Save Output
# Standardize timestamp
df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')

# Prepare output columns
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
distorted_events = df['is_distorted'].sum()
unique_before = df['original_activity'].nunique()
unique_after = df['Activity_fixed'].nunique()
reduction = unique_before - unique_after
reduction_pct = (reduction / unique_before * 100) if unique_before > 0 else 0

print(f"Total number of events: {total_events}")
print(f"Number of distorted events detected: {distorted_events}")
print(f"Unique activities before fixing: {unique_before}")
print(f"Unique activities after fixing: {unique_after}")
print(f"Activity reduction count: {reduction} ({reduction_pct:.2f}%)")
print(f"Output file path: {output_file}")

# Sample transformations
changes = df[df['is_distorted'] == 1][['original_activity', 'canonical_activity']].drop_duplicates()
print("Sample transformations (up to 10):")
for i in range(min(10, len(changes))):
    orig = changes.iloc[i]['original_activity']
    can = changes.iloc[i]['canonical_activity']
    print(f"{orig} → {can}")

print(f"Run 1: Processed dataset saved to: data/pub/pub_distorted_cleaned_run1.csv")
print(f"Run 1: Final dataset shape: {df_output.shape}")
print(f"Run 1: Dataset: pub")
print(f"Run 1: Task type: distorted")