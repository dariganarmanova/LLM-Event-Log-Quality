# Generated script for BPIC11-Distorted - Run 1
# Generated on: 2025-11-18T22:19:57.653504
# Model: grok-4-fast

import pandas as pd
import re
from sklearn.metrics import precision_recall_fscore_support
from collections import defaultdict

# Configuration parameters
input_file = 'data/bpic11/BPIC11-Distorted.csv'
output_file = 'data/bpic11/bpic11_distorted_cleaned_run1.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
distorted_suffix = ':distorted'
ngram_size = 3
similarity_threshold = 0.56
min_length = 4
time_threshold = 2.0
require_same_resource = False
min_matching_events = 2
max_mismatches = 1
activity_suffix_pattern = r'(_signed\d*|_\d+)$'
case_sensitive = False
use_fuzzy_matching = False

# Load the data
df = pd.read_csv(input_file)
print(f"Run 1: Original dataset shape: {df.shape}")

# Step 1: Store original Activity
df['original_activity'] = df[activity_column].copy()

# Step 2: Identify distorted activities
df['isdistorted'] = df[activity_column].str.endswith(distorted_suffix, na=False).astype(int)
df['BaseActivity'] = df[activity_column].str.replace(f'{distorted_suffix}$', '', regex=True)

# Step 3: Preprocess Activity Names
def preprocess_activity(text):
    if pd.isna(text) or text == '':
        return ''
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = ' '.join(text.split())  # Collapse multiple spaces and strip
    return text

df['ProcessedActivity'] = df['BaseActivity'].apply(preprocess_activity)

# Step 4: Calculate Jaccard N-gram Similarity
def generate_ngrams(text, n):
    if pd.isna(text) or len(text) < n:
        return set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}

def jaccard_similarity(text1, text2, n):
    set1 = generate_ngrams(text1, n)
    set2 = generate_ngrams(text2, n)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0

unique_processed = [p for p in df['ProcessedActivity'].unique() if len(p) >= min_length and p != '']
print(f"Number of unique processed activities for clustering: {len(unique_processed)}")

similar_pairs = []
for i, p1 in enumerate(unique_processed):
    for p2 in unique_processed[i + 1:]:
        sim = jaccard_similarity(p1, p2, ngram_size)
        if sim >= similarity_threshold:
            similar_pairs.append((p1, p2))
    if i % 10 == 0:
        print(f"Progress: Processed {i}/{len(unique_processed)} activities")

print(f"Found {len(similar_pairs)} similar pairs above threshold {similarity_threshold}")

# Step 5: Cluster Similar Activities (Union-Find)
parent = {p: p for p in unique_processed}
rank = {p: 0 for p in unique_processed}

def find(p):
    if parent[p] != p:
        parent[p] = find(parent[p])
    return parent[p]

def union(p1, p2):
    pp1 = find(p1)
    pp2 = find(p2)
    if pp1 == pp2:
        return
    if rank[pp1] > rank[pp2]:
        parent[pp2] = pp1
    elif rank[pp1] < rank[pp2]:
        parent[pp1] = pp2
    else:
        parent[pp2] = pp1
        rank[pp1] += 1

for p1, p2 in similar_pairs:
    union(p1, p2)

clusters = defaultdict(list)
for p in unique_processed:
    clusters[find(p)].append(p)

large_clusters = {k: v for k, v in clusters.items() if len(v) >= 2}

# Step 6: Majority Voting Within Clusters
canonical_map = {}
for root, procs in large_clusters.items():
    cluster_mask = df['ProcessedActivity'].isin(procs)
    cluster_variants = df.loc[cluster_mask, 'BaseActivity'].value_counts()
    if len(cluster_variants) > 0:
        canonical = cluster_variants.index[0]
        for var in cluster_variants.index:
            canonical_map[var] = canonical

df['canonical_activity'] = df['BaseActivity'].replace(canonical_map)
df['is_distorted'] = (df['BaseActivity'] != df['canonical_activity']).astype(int)

# Step 7: Calculate Detection Metrics (BEFORE FIXING)
if label_column in df.columns:
    def normalize_label(label):
        if pd.isna(label):
            return 0
        return 1 if 'distorted' in str(label).lower() else 0
    y_true = df[label_column].apply(normalize_label)
    y_pred = df['is_distorted']
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    print("=== Detection Performance Metrics ===")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    if prec >= 0.6:
        print("✓ Precision threshold (≥ 0.6) met")
    else:
        print("✗ Precision threshold (≥ 0.6) not met")
else:
    print("=== Detection Performance Metrics ===")
    print("Precision: 0.0000")
    print("Recall: 0.0000")
    print("F1-Score: 0.0000")
    print("No labels available for metric calculation")

# Step 8: Integrity Check
num_clusters = len(large_clusters)
num_distorted = int(df['is_distorted'].sum())
num_to_fix = num_distorted
num_clean_unchanged = len(df) - num_distorted
print(f"Total distortion clusters detected: {num_clusters}")
print(f"Total activities marked as distorted: {num_distorted}")
print(f"Activities to be fixed: {num_to_fix}")
print(f"Clean activities that were NOT modified: {num_clean_unchanged}")

# Step 9: Fix Activities
df['Activity_fixed'] = df['canonical_activity']

# Step 10: Save Output
# Format timestamp
df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')

# Select columns
output_columns = [case_column, timestamp_column]
if 'Variant' in df.columns:
    output_columns.append('Variant')
if 'Resource' in df.columns:
    output_columns.append('Resource')
output_columns += [activity_column, 'Activity_fixed']
if label_column in df.columns:
    output_columns.append(label_column)

df_output = df[output_columns].copy()
df_output.to_csv(output_file, index=False)

# Step 11: Summary Statistics
total_events = len(df_output)
num_distorted_detected = num_distorted
unique_before = df[activity_column].nunique()
unique_after = df_output['Activity_fixed'].nunique()
reduction = unique_before - unique_after
percentage = (reduction / unique_before * 100) if unique_before > 0 else 0
print(f"Total number of events: {total_events}")
print(f"Number of distorted events detected: {num_distorted_detected}")
print(f"Unique activities before fixing: {unique_before}")
print(f"Unique activities after fixing: {unique_after}")
print(f"Activity reduction count and percentage: {reduction} ({percentage:.2f}%)")
print(f"Output file path: {output_file}")

# Sample transformations
print("Sample of up to 10 transformations showing: original → canonical")
changed_mask = df['BaseActivity'] != df['canonical_activity']
if changed_mask.any():
    samples = df.loc[changed_mask, ['BaseActivity', 'canonical_activity']].drop_duplicates().head(10)
    for _, row in samples.iterrows():
        print(f"- {row['BaseActivity']} → {row['canonical_activity']}")
else:
    print("- No transformations needed")

# Required prints
print(f"Run 1: Processed dataset saved to: data/bpic11/bpic11_distorted_cleaned_run1.csv")
print(f"Run 1: Final dataset shape: {df_output.shape}")
print(f"Run 1: Dataset: bpic11")
print(f"Run 1: Task type: distorted")