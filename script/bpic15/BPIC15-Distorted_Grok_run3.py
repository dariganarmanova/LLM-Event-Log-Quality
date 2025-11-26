# Generated script for BPIC15-Distorted - Run 3
# Generated on: 2025-11-18T21:27:03.850045
# Model: grok-4-fast

import pandas as pd
import re
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration parameters
input_file = 'data/bpic15/BPIC15-Distorted.csv'
output_file = 'data/bpic15/bpic15_distorted_cleaned_run3.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
distorted_suffix = ':distorted'
ngram_size = 3
similarity_threshold = 0.56
min_length = 4

# Load the data
df = pd.read_csv(input_file)
print(f"Run 3: Original dataset shape: {df.shape}")

# Handle potential column name variations for Case
if 'Case ID' in df.columns:
    df[case_column] = df.pop('Case ID')

# Store original Activity
df['original_activity'] = df[activity_column].copy()

# Step 2: Identify Distorted Activities
df['BaseActivity'] = df[activity_column].str.replace(rf'{distorted_suffix}$', '', regex=True)

# Step 3: Preprocess Activity Names
def preprocess(text):
    if pd.isna(text):
        return ''
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['ProcessedActivity'] = df['BaseActivity'].apply(preprocess)

# Step 4: Calculate Jaccard N-gram Similarity
def generate_ngrams(text, n):
    if len(text) < n:
        return set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}

def jaccard_similarity(a, b, n):
    set_a = generate_ngrams(a, n)
    set_b = generate_ngrams(b, n)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

unique_processed = [p for p in df['ProcessedActivity'].unique() if len(p) >= min_length and p]
print(f"Number of unique processed activities for clustering: {len(unique_processed)}")

similar_pairs = []
num_comparisons = 0
for i in range(len(unique_processed)):
    for j in range(i + 1, len(unique_processed)):
        num_comparisons += 1
        if num_comparisons % 1000 == 0:
            print(f"Processed {num_comparisons} comparisons...")
        sim = jaccard_similarity(unique_processed[i], unique_processed[j], ngram_size)
        if sim >= similarity_threshold:
            similar_pairs.append((unique_processed[i], unique_processed[j]))
print(f"Found {len(similar_pairs)} similar pairs above threshold {similarity_threshold}")

# Step 5: Cluster Similar Activities (Union-Find)
parent = {act: act for act in unique_processed}
rank = {act: 0 for act in unique_processed}

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

clusters = {}
for act in unique_processed:
    root = find(act)
    clusters.setdefault(root, []).append(act)

multi_clusters = {k: v for k, v in clusters.items() if len(v) >= 2}
print(f"Found {len(multi_clusters)} clusters with 2+ members")

# Step 6: Majority Voting Within Clusters
canonical_map = {}
for root, cluster_procs in multi_clusters.items():
    mask = df['ProcessedActivity'].isin(cluster_procs)
    cluster_df = df[mask]
    if len(cluster_df) == 0:
        continue
    freq = cluster_df['original_activity'].value_counts()
    canonical = freq.index[0]
    for orig in cluster_df['original_activity'].unique():
        canonical_map[orig] = canonical

# Handle singletons
all_originals = df['original_activity'].unique()
for orig in all_originals:
    if orig not in canonical_map:
        canonical_map[orig] = orig

# Set canonical and is_distorted
df['canonical_activity'] = df['original_activity'].map(canonical_map)
df['is_distorted'] = (df['original_activity'] != df['canonical_activity']).astype(int)

# Step 7: Calculate Detection Metrics (BEFORE FIXING)
print("=== Detection Performance Metrics ===")
if label_column in df.columns:
    def normalize_label(l):
        if pd.isna(l):
            return 0
        return 1 if 'distorted' in str(l).lower() else 0
    y_true = df[label_column].apply(normalize_label)
    y_pred = df['is_distorted']
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
num_clusters = len(multi_clusters)
num_distorted = df['is_distorted'].sum()
num_clean = len(df) - num_distorted
num_to_fix = num_distorted
print("\n=== Integrity Check ===")
print(f"Total distortion clusters detected: {num_clusters}")
print(f"Total activities marked as distorted: {num_distorted}")
print(f"Clean activities that were NOT modified: {num_clean}")
print(f"Activities to be fixed: {num_to_fix}")

# Step 9: Fix Activities
df['Activity_fixed'] = df['canonical_activity']

# Step 10: Save Output
# Format timestamp
if timestamp_column in df.columns:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')

# Select columns
output_cols = [case_column, activity_column, timestamp_column, 'Activity_fixed']
if 'Variant' in df.columns:
    output_cols.insert(3, 'Variant')
if label_column in df.columns:
    output_cols.append(label_column)
if 'Resource' in df.columns:
    output_cols.append('Resource')

df_output = df[output_cols].copy()
df_output.to_csv(output_file, index=False)

# Step 11: Summary Statistics
total_events = len(df)
distorted_detected = num_distorted
unique_before = df[activity_column].nunique()
unique_after = df['Activity_fixed'].nunique()
reduction_count = unique_before - unique_after
reduction_pct = (reduction_count / unique_before * 100) if unique_before > 0 else 0.0

print("\n=== Summary Statistics ===")
print(f"Total number of events: {total_events}")
print(f"Number of distorted events detected: {distorted_detected}")
print(f"Unique activities before fixing: {unique_before}")
print(f"Unique activities after fixing: {unique_after}")
print(f"Activity reduction count: {reduction_count}")
print(f"Activity reduction percentage: {reduction_pct:.2f}%")
print(f"Output file path: {output_file}")

print("\nSample transformations (up to 10):")
changes = df[df[activity_column] != df['Activity_fixed']][[activity_column, 'Activity_fixed']].drop_duplicates().head(10)
if len(changes) == 0:
    print("No transformations needed.")
else:
    for _, row in changes.iterrows():
        print(f"{row[activity_column]} → {row['Activity_fixed']}")

print(f"\nRun 3: Processed dataset saved to: {output_file}")
print(f"Run 3: Final dataset shape: {df_output.shape}")
print(f"Run 3: Dataset: bpic15")
print(f"Run 3: Task type: distorted")