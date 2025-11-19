# Generated script for BPIC11-Distorted - Run 2
# Generated on: 2025-11-13T11:23:26.939751
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
output_file = 'data/bpic11/bpic11_distorted_cleaned_run2.csv'
df = pd.read_csv(input_file)
print(f"Run 2: Original dataset shape: {df.shape}")

# Step 1: Ensure required columns exist
required_columns = ['Case', 'Activity', 'Timestamp']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' not found in the input file")

# Store original activity
df['original_activity'] = df['Activity']

# Step 2: Identify distorted activities
df['isdistorted'] = df['Activity'].str.endswith(distorted_suffix).astype(int)
df['BaseActivity'] = df['Activity'].str.replace(distorted_suffix, '', regex=False)

# Step 3: Preprocess activity names
def preprocess_activity(activity):
    if pd.isna(activity):
        return ''
    # Convert to lowercase if case insensitive
    if not case_sensitive:
        activity = activity.lower()
    # Remove non-alphanumeric except spaces
    activity = re.sub(r'[^a-zA-Z0-9\s]', '', activity)
    # Collapse whitespace
    activity = ' '.join(activity.split())
    return activity

df['ProcessedActivity'] = df['BaseActivity'].apply(preprocess_activity)
# Filter out short activities
valid_activities = df[df['ProcessedActivity'].str.len() >= min_length].copy()

# Step 4: Jaccard similarity functions
def get_ngrams(text, n):
    return set([text[i:i+n] for i in range(len(text)-n+1)])

def jaccard_similarity(a, b, n):
    ngrams_a = get_ngrams(a, n)
    ngrams_b = get_ngrams(b, n)
    if not ngrams_a and not ngrams_b:
        return 1.0
    intersection = len(ngrams_a & ngrams_b)
    union = len(ngrams_a | ngrams_b)
    return intersection / union

# Find similar pairs
unique_activities = valid_activities['ProcessedActivity'].unique()
similar_pairs = []
for i in range(len(unique_activities)):
    for j in range(i+1, len(unique_activities)):
        a = unique_activities[i]
        b = unique_activities[j]
        sim = jaccard_similarity(a, b, ngram_size)
        if sim >= similarity_threshold:
            similar_pairs.append((a, b))

# Step 5: Union-Find clustering
parent = {}

def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(x, y):
    root_x = find(x)
    root_y = find(y)
    if root_x != root_y:
        parent[root_y] = root_x

# Initialize parent pointers
for activity in unique_activities:
    parent[activity] = activity

# Union similar pairs
for a, b in similar_pairs:
    union(a, b)

# Group activities by cluster
clusters = defaultdict(list)
for activity in unique_activities:
    root = find(activity)
    clusters[root].append(activity)

# Filter clusters with at least 2 members
clusters = {k: v for k, v in clusters.items() if len(v) >= 2}

# Step 6: Majority voting for canonical forms
canonical_mapping = {}
activity_to_canonical = {}

for cluster_rep, cluster_members in clusters.items():
    # Get original activities for this cluster
    cluster_df = valid_activities[valid_activities['ProcessedActivity'].isin(cluster_members)]
    # Count original activity frequencies
    freq = cluster_df['BaseActivity'].value_counts()
    if not freq.empty:
        canonical = freq.idxmax()
        for member in cluster_members:
            member_activities = cluster_df[cluster_df['ProcessedActivity'] == member]['BaseActivity'].unique()
            for ma in member_activities:
                activity_to_canonical[ma] = canonical

# Apply canonical mapping
def get_canonical(activity):
    return activity_to_canonical.get(activity, activity)

df['canonical_activity'] = df['BaseActivity'].apply(get_canonical)
df['is_distorted'] = (df['BaseActivity'] != df['canonical_activity']).astype(int)

# Step 7: Detection metrics
if 'label' in df.columns:
    def normalize_label(label):
        if pd.isna(label):
            return 0
        if 'distorted' in str(label).lower():
            return 1
        return 0
    
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
    print("Precision: 0.0000")
    print("Recall: 0.0000")
    print("F1-Score: 0.0000")

# Step 8: Integrity check
num_clusters = len(clusters)
num_distorted = df['is_distorted'].sum()
num_clean = len(df) - num_distorted

print("\n=== Integrity Check ===")
print(f"Total distortion clusters detected: {num_clusters}")
print(f"Total activities marked as distorted: {num_distorted}")
print(f"Activities to be fixed: {num_distorted}")

# Step 9: Fix activities
df['Activity_fixed'] = df['canonical_activity']

# Step 10: Save output
output_columns = ['Case', 'Timestamp', 'Activity', 'Activity_fixed']
if 'label' in df.columns:
    output_columns.append('label')
df_output = df[output_columns]
df_output.to_csv(output_file, index=False)

# Step 11: Summary statistics
unique_before = df['Activity'].nunique()
unique_after = df['Activity_fixed'].nunique()
reduction = unique_before - unique_after
reduction_pct = (reduction / unique_before) * 100 if unique_before > 0 else 0

print("\n=== Summary Statistics ===")
print(f"Total number of events: {len(df)}")
print(f"Number of distorted events detected: {num_distorted}")
print(f"Unique activities before fixing: {unique_before}")
print(f"Unique activities after fixing: {unique_after}")
print(f"Activity reduction: {reduction} ({reduction_pct:.2f}%)")
print(f"Output file path: {output_file}")

# Print sample transformations
sample_size = min(10, len(df))
sample = df.sample(sample_size)[['Activity', 'Activity_fixed']].drop_duplicates()
print("\nSample transformations (original → canonical):")
for _, row in sample.iterrows():
    print(f"{row['Activity']} → {row['Activity_fixed']}")

print(f"\nRun 2: Processed dataset saved to: {output_file}")
print(f"Run 2: Final dataset shape: {df.shape}")
print(f"Run 2: Dataset: bpic11")
print(f"Run 2: Task type: distorted")