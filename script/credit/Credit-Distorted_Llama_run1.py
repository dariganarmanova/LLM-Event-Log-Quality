# Generated script for Credit-Distorted - Run 1
# Generated on: 2025-11-13T15:59:46.002314
# Model: meta-llama/Llama-3.1-8B-Instruct

import pandas as pd
import re
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import defaultdict

# Load the data
input_file = 'data/credit/Credit-Distorted.csv'
output_file = 'data/credit/credit_distorted_cleaned_run1.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
distorted_suffix = ':distorted'

# Load the data
df = pd.read_csv(input_file)

# Configuration parameters
time_threshold = 2.0
require_same_resource = False
min_matching_events = 2
max_mismatches = 1
activity_suffix_pattern = r"(_signed\d*|_\d+)$"
similarity_threshold = 0.8
case_sensitive = False
use_fuzzy_matching = False

# Normalize CaseID → Case if needed (handle naming variations)
df[case_column] = df[case_column].str.lower()

# Store original Activity values in `original_activity` column for reference
df['original_activity'] = df[activity_column]

# Create `isdistorted` column: 1 if Activity ends with `distorted_suffix`, else 0
df['isdistorted'] = df[activity_column].str.endswith(distorted_suffix)

# Create `BaseActivity` column: Activity with distorted suffix removed
df['BaseActivity'] = df[activity_column].str.removesuffix(distorted_suffix)

# Preprocess Activity Names
def preprocess_activity(activity):
    if case_sensitive:
        return activity
    else:
        activity = activity.lower()
    activity = re.sub(r'[^a-z0-9\s]', '', activity)
    activity = ' '.join(activity.split())
    return activity

df['ProcessedActivity'] = df[activity_column].apply(preprocess_activity)

# Filter out activities shorter than min_length characters (too short for meaningful comparison)
min_length = 4
df = df[df['ProcessedActivity'].str.len() >= min_length]

# Calculate Jaccard N-gram Similarity
def jaccard_similarity(val1, val2):
    ngram_size = 3
    val1_ngrams = set(''.join(val1[i:i+ngram_size]) for i in range(len(val1)-ngram_size+1))
    val2_ngrams = set(''.join(val2[i:i+ngram_size]) for i in range(len(val2)-ngram_size+1))
    intersection = val1_ngrams.intersection(val2_ngrams)
    union = val1_ngrams.union(val2_ngrams)
    if not union:
        return 0.0
    return len(intersection) / len(union)

# Find Similar Pairs
similar_pairs = []
unique_activities = df['ProcessedActivity'].unique()
for i in range(len(unique_activities)):
    for j in range(i+1, len(unique_activities)):
        val1 = unique_activities[i]
        val2 = unique_activities[j]
        similarity = jaccard_similarity(val1, val2)
        if similarity >= similarity_threshold:
            similar_pairs.append((val1, val2))

# Cluster Similar Activities (Union-Find)
def find(parent, i):
    if parent[i] != i:
        parent[i] = find(parent, parent[i])
    return parent[i]

def union(parent, rank, x, y):
    x_root = find(parent, x)
    y_root = find(parent, y)
    if x_root == y_root:
        return
    if rank[x_root] < rank[y_root]:
        parent[x_root] = y_root
    elif rank[x_root] > rank[y_root]:
        parent[y_root] = x_root
    else:
        parent[y_root] = x_root
        rank[x_root] += 1

parent = {i: i for i in range(len(unique_activities))}
rank = {i: 0 for i in range(len(unique_activities))}
for val1, val2 in similar_pairs:
    union(parent, rank, unique_activities.index(val1), unique_activities.index(val2))

# Group activities by their root parent
clusters = defaultdict(list)
for i, activity in enumerate(unique_activities):
    clusters[find(parent, i)].append(activity)

# Keep only clusters with 2 or more members (single activities don't need fixing)
clusters = {k: v for k, v in clusters.items() if len(v) >= 2}

# Majority Voting Within Clusters
canonical_form = {}
for cluster in clusters.values():
    frequencies = {}
    for activity in cluster:
        frequencies[activity] = frequencies.get(activity, 0) + 1
    canonical_activity = max(frequencies, key=frequencies.get)
    canonical_form[canonical_activity] = canonical_activity

# Mark distorted activities
df['canonical_activity'] = df['original_activity'].map(canonical_form)
df['is_distorted'] = df['original_activity'].apply(lambda x: 1 if x != df['canonical_activity'].iloc[i] else 0)

# Calculate Detection Metrics (BEFORE FIXING)
if label_column in df.columns:
    def normalize_label(label):
        if pd.isnull(label) or label == np.nan:
            return 0
        elif 'distorted' in label.lower():
            return 1
        else:
            return 0

    y_true = df[label_column].apply(normalize_label)
    y_pred = df['is_distorted']
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"✓/✗ Precision threshold (≥ 0.6) met/not met")

# Integrity Check
total_distortion_clusters = len(clusters)
total_distorted_activities = df['is_distorted'].sum()
total_clean_activities = (df['is_distorted'] == 0).sum()
print(f"Total distortion clusters detected: {total_distortion_clusters}")
print(f"Total activities marked as distorted: {total_distorted_activities}")
print(f"Activities to be fixed: {total_distorted_activities}")

# Fix Activities
df['Activity_fixed'] = df['canonical_activity']

# Save Output
output_columns = [case_column, timestamp_column, 'Variant', 'Activity', 'Activity_fixed']
if label_column in df.columns:
    output_columns.append(label_column)
df_output = df[output_columns]
df_output[timestamp_column] = df_output[timestamp_column].dt.strftime('%Y-%m-%d %H:%M:%S')
df_output.to_csv(output_file, index=False)

# Summary Statistics
print(f"Total number of events: {df.shape[0]}")
print(f"Number of distorted events detected: {total_distorted_activities}")
print(f"Unique activities before fixing: {len(unique_activities)}")
print(f"Unique activities after fixing: {len(canonical_form)}")
activity_reduction_count = len(unique_activities) - len(canonical_form)
activity_reduction_percentage = (activity_reduction_count / len(unique_activities)) * 100
print(f"Activity reduction count: {activity_reduction_count}")
print(f"Activity reduction percentage: {activity_reduction_percentage:.2f}%")
print(f"Output file path: {output_file}")
print(f"Sample of up to 10 transformations: {df.head(10)}")

print(f"Run 1: Processed dataset saved to: {output_file}")
print(f"Run 1: Final dataset shape: {df.shape}")
print(f"Run 1: Dataset: credit")
print(f"Run 1: Task type: distorted")