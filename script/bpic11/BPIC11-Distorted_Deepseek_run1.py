# Generated script for BPIC11-Distorted - Run 1
# Generated on: 2025-11-13T11:22:21.916486
# Model: deepseek-ai/DeepSeek-V3-0324

import pandas as pd
import re
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration parameters
time_threshold = 2.0
require_same_resource = False
min_matching_events = 2
max_mismatches = 1
activity_suffix_pattern = r"(_signed\d*|_\d+)$"
similarity_threshold = 0.8
case_sensitive = False
use_fuzzy_matching = False
ngram_size = 3
min_length = 4
distorted_suffix = ':distorted'

# Load the data
input_file = 'data/bpic11/BPIC11-Distorted.csv'
output_file = 'data/bpic11/bpic11_distorted_cleaned_run1.csv'
df = pd.read_csv(input_file)
print(f"Run 1: Original dataset shape: {df.shape}")

# Ensure required columns exist
required_columns = ['Case', 'Activity', 'Timestamp']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' not found in the input file")

# Store original activity
df['original_activity'] = df['Activity']

# Identify distorted activities
df['isdistorted'] = df['Activity'].str.endswith(distorted_suffix).astype(int)
df['BaseActivity'] = df['Activity'].str.replace(distorted_suffix, '', regex=False)

# Preprocess activity names
def preprocess_activity(activity):
    if not isinstance(activity, str):
        return ''
    activity = activity.lower() if not case_sensitive else activity
    activity = re.sub(r'[^a-zA-Z0-9\s]', '', activity)
    activity = re.sub(r'\s+', ' ', activity).strip()
    return activity

df['ProcessedActivity'] = df['BaseActivity'].apply(preprocess_activity)
df = df[df['ProcessedActivity'].str.len() >= min_length].copy()

# Generate n-grams
def get_ngrams(text, n):
    return set([text[i:i+n] for i in range(len(text)-n+1)])

# Calculate Jaccard similarity
def jaccard_similarity(a, b, n):
    a_ngrams = get_ngrams(a, n)
    b_ngrams = get_ngrams(b, n)
    if not a_ngrams and not b_ngrams:
        return 0.0
    intersection = len(a_ngrams & b_ngrams)
    union = len(a_ngrams | b_ngrams)
    return intersection / union

# Find similar pairs
unique_activities = df['ProcessedActivity'].unique()
similar_pairs = []

for i in range(len(unique_activities)):
    for j in range(i+1, len(unique_activities)):
        a = unique_activities[i]
        b = unique_activities[j]
        sim = jaccard_similarity(a, b, ngram_size)
        if sim >= similarity_threshold:
            similar_pairs.append((a, b))

# Union-Find implementation
parent = {}

def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(x, y):
    parent[find(x)] = find(y)

# Initialize parent
for activity in unique_activities:
    parent[activity] = activity

# Build clusters
for a, b in similar_pairs:
    union(a, b)

# Get clusters
clusters = defaultdict(list)
for activity in unique_activities:
    root = find(activity)
    clusters[root].append(activity)

# Filter clusters with at least 2 members
clusters = {k: v for k, v in clusters.items() if len(v) >= 2}

# Majority voting for canonical forms
activity_to_canonical = {}
for cluster in clusters.values():
    # Get all original activities in this cluster
    cluster_activities = df[df['ProcessedActivity'].isin(cluster)]['BaseActivity']
    # Find most frequent original form
    canonical = cluster_activities.value_counts().idxmax()
    # Map all variants to canonical form
    for activity in cluster:
        activity_to_canonical[activity] = canonical

# Apply canonical forms
def get_canonical(row):
    processed = row['ProcessedActivity']
    if processed in activity_to_canonical:
        return activity_to_canonical[processed]
    return row['BaseActivity']

df['canonical_activity'] = df.apply(get_canonical, axis=1)
df['Activity_fixed'] = df['canonical_activity']

# Mark distorted activities
df['is_distorted'] = df.apply(lambda row: 0 if row['BaseActivity'] == row['canonical_activity'] else 1, axis=1)

# Calculate metrics if label column exists
if 'label' in df.columns:
    def normalize_label(label):
        if pd.isna(label):
            return 0
        return 1 if 'distorted' in str(label).lower() else 0
    
    y_true = df['label'].apply(normalize_label)
    y_pred = df['is_distorted']
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print("=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("✓ Precision threshold (≥ 0.6) met" if precision >= 0.6 else "✗ Precision threshold (≥ 0.6) not met")
else:
    print("=== Detection Performance Metrics ===")
    print("No labels available for metric calculation")

# Integrity check
total_clusters = len(clusters)
total_distorted = df['is_distorted'].sum()
total_clean = len(df) - total_distorted

print("\n=== Integrity Check ===")
print(f"Total distortion clusters detected: {total_clusters}")
print(f"Total activities marked as distorted: {total_distorted}")
print(f"Activities to be fixed: {total_distorted}")

# Save output
output_columns = ['Case', 'Timestamp', 'Activity', 'Activity_fixed']
if 'label' in df.columns:
    output_columns.append('label')
df_output = df[output_columns]
df_output.to_csv(output_file, index=False)

# Summary statistics
unique_before = df['Activity'].nunique()
unique_after = df['Activity_fixed'].nunique()
reduction = unique_before - unique_after
reduction_pct = (reduction / unique_before) * 100

print("\n=== Summary Statistics ===")
print(f"Total number of events: {len(df)}")
print(f"Number of distorted events detected: {total_distorted}")
print(f"Unique activities before fixing: {unique_before}")
print(f"Unique activities after fixing: {unique_after}")
print(f"Activity reduction count: {reduction} ({reduction_pct:.2f}%)")
print(f"Output file path: {output_file}")

# Print sample transformations
sample_size = min(10, len(df))
sample = df[['Activity', 'Activity_fixed']].drop_duplicates().head(sample_size)
print("\nSample transformations:")
for _, row in sample.iterrows():
    print(f"{row['Activity']} → {row['Activity_fixed']}")

print(f"\nRun 1: Processed dataset saved to: {output_file}")
print(f"Run 1: Final dataset shape: {df.shape}")
print(f"Run 1: Dataset: bpic11")
print(f"Run 1: Task type: distorted")