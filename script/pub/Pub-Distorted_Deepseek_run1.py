# Generated script for Pub-Distorted - Run 1
# Generated on: 2025-11-13T17:44:40.991030
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
input_file = 'data/pub/Pub-Distorted.csv'
output_file = 'data/pub/pub_distorted_cleaned_run1.csv'
df = pd.read_csv(input_file)
print(f"Run 1: Original dataset shape: {df.shape}")

# Ensure required columns exist
required_columns = {'Case', 'Activity', 'Timestamp'}
assert required_columns.issubset(set(df.columns)), f"Missing required columns: {required_columns - set(df.columns)}"

# Step 1: Store original activities
df['original_activity'] = df['Activity']

# Step 2: Identify distorted activities
df['isdistorted'] = df['Activity'].str.endswith(distorted_suffix).astype(int)
df['BaseActivity'] = df['Activity'].str.replace(distorted_suffix, '', regex=False)

# Step 3: Preprocess activity names
def preprocess_activity(activity):
    if pd.isna(activity):
        return ''
    activity = str(activity)
    if not case_sensitive:
        activity = activity.lower()
    activity = re.sub(r'[^a-zA-Z0-9\s]', '', activity)
    activity = re.sub(r'\s+', ' ', activity).strip()
    return activity

df['ProcessedActivity'] = df['BaseActivity'].apply(preprocess_activity)
valid_activities = df[df['ProcessedActivity'].str.len() >= min_length]

# Helper functions for Jaccard similarity
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

# Step 4: Find similar pairs
unique_activities = valid_activities['ProcessedActivity'].unique()
similar_pairs = []

for i in range(len(unique_activities)):
    for j in range(i+1, len(unique_activities)):
        a = unique_activities[i]
        b = unique_activities[j]
        sim = jaccard_similarity(a, b, ngram_size)
        if sim >= similarity_threshold:
            similar_pairs.append((a, b))

# Step 5: Cluster similar activities (Union-Find)
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

# Filter clusters with at least min_matching_events
final_clusters = {k: v for k, v in clusters.items() if len(v) >= min_matching_events}

# Step 6: Majority voting within clusters
canonical_mapping = {}
activity_to_cluster = {}

for cluster_name, activities in final_clusters.items():
    original_forms = []
    for activity in activities:
        mask = (df['ProcessedActivity'] == activity)
        original_forms.extend(df.loc[mask, 'BaseActivity'].tolist())
    
    if not original_forms:
        continue
    
    # Find most frequent original form
    freq = pd.Series(original_forms).value_counts()
    canonical = freq.index[0]
    
    # Map all processed activities in cluster to canonical form
    for activity in activities:
        activity_to_cluster[activity] = canonical

# Create canonical_activity column
def get_canonical(row):
    processed = row['ProcessedActivity']
    if processed in activity_to_cluster:
        return activity_to_cluster[processed]
    return row['BaseActivity']

df['canonical_activity'] = df.apply(get_canonical, axis=1)

# Mark distorted activities
df['is_distorted'] = (df['BaseActivity'] != df['canonical_activity']).astype(int)

# Step 7: Calculate detection metrics (if label column exists)
if 'label' in df.columns:
    def normalize_label(label):
        if pd.isna(label):
            return 0
        label = str(label).lower()
        return 1 if 'distorted' in label else 0
    
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
total_clusters = len(final_clusters)
total_distorted = df['is_distorted'].sum()
total_clean = len(df) - total_distorted

print("\n=== Integrity Check ===")
print(f"Total distortion clusters detected: {total_clusters}")
print(f"Total activities marked as distorted: {total_distorted}")
print(f"Activities to be fixed: {total_distorted}")

# Step 9: Fix activities
df['Activity_fixed'] = df['canonical_activity']

# Step 10: Save output
output_columns = ['Case', 'Timestamp', 'Activity', 'Activity_fixed']
if 'Variant' in df.columns:
    output_columns.append('Variant')
if 'Resource' in df.columns:
    output_columns.append('Resource')
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
print(f"Number of distorted events detected: {total_distorted}")
print(f"Unique activities before fixing: {unique_before}")
print(f"Unique activities after fixing: {unique_after}")
print(f"Activity reduction count: {reduction} ({reduction_pct:.2f}%)")
print(f"Output file path: {output_file}")

# Print sample transformations
sample_size = min(10, len(df))
sample = df.sample(sample_size)[['Activity', 'Activity_fixed']].drop_duplicates()
print("\nSample transformations (original → canonical):")
for _, row in sample.iterrows():
    print(f"{row['Activity']} → {row['Activity_fixed']}")

# Final required prints
print(f"\nRun 1: Processed dataset saved to: {output_file}")
print(f"Run 1: Final dataset shape: {df.shape}")
print(f"Run 1: Dataset: pub")
print(f"Run 1: Task type: distorted")