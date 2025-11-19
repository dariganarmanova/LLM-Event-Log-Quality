# Generated script for Credit-Distorted - Run 2
# Generated on: 2025-11-13T15:59:49.137726
# Model: meta-llama/Llama-3.1-8B-Instruct

import pandas as pd
import re
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk import ngrams as nltk_ngrams
from nltk import word_tokenize as nltk_tokenize
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Configuration parameters
time_threshold = 2.0
require_same_resource = False
min_matching_events = 2
max_mismatches = 1
activity_suffix_pattern = r"(_signed\d*|_\d+)$"
similarity_threshold = 0.8
case_sensitive = False
use_fuzzy_matching = False

# Load the data
input_file = 'data/credit/Credit-Distorted.csv'
output_file = 'data/credit/credit_distorted_cleaned_run2.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
distorted_suffix = ':distorted'

# Load data
df = pd.read_csv(input_file)

# Normalize CaseID → Case if needed (handle naming variations)
df[case_column] = df[case_column].str.lower()

# Store original Activity values in `original_activity` column for reference
df['original_activity'] = df[activity_column]

# Create `isdistorted` column: 1 if Activity ends with `distorted_suffix`, else 0
df['isdistorted'] = df[activity_column].str.endswith(distorted_suffix)

# Create `BaseActivity` column: Activity with distorted suffix removed
df['BaseActivity'] = df[activity_column].str.replace(distorted_suffix, '')

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
    val1_ngrams = set(' '.join(map(str, ngrams(val1, ngram_size))).split())
    val2_ngrams = set(' '.join(map(str, ngrams(val2, ngram_size))).split())
    intersection = val1_ngrams.intersection(val2_ngrams)
    union = val1_ngrams.union(val2_ngrams)
    return len(intersection) / len(union) if union else 0.0

# Find Similar Pairs
similar_pairs = []
unique_activities = df['ProcessedActivity'].unique()
for i in range(len(unique_activities)):
    for j in range(i + 1, len(unique_activities)):
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

parent = {i: i for i in unique_activities}
rank = {i: 0 for i in unique_activities}

for val1, val2 in similar_pairs:
    union(parent, rank, val1, val2)

# Build clusters
clusters = defaultdict(list)
for activity in unique_activities:
    clusters[find(parent, activity)].append(activity)

# Majority Voting Within Clusters
canonical_mapping = {}
for cluster_id, cluster in clusters.items():
    if len(cluster) > 1:
        # Map normalized ProcessedActivity values back to their original Activity forms
        original_activities = df.loc[df['ProcessedActivity'].isin(cluster), 'original_activity'].unique()
        # Count the frequency of each original activity variant in the cluster
        frequency = {activity: cluster.count(activity) for activity in original_activities}
        # The most frequent original activity becomes the canonical form
        canonical_activity = max(frequency, key=frequency.get)
        canonical_mapping[canonical_activity] = canonical_activity
        # Mark distorted activities
        for activity in cluster:
            if activity != canonical_activity:
                canonical_mapping[activity] = canonical_activity

# Replace Activity column with canonical_activity values
df['canonical_activity'] = df['original_activity'].map(canonical_mapping)

# Calculate Detection Metrics (BEFORE FIXING)
if label_column in df.columns:
    # Normalize label values
    def normalize_label(label):
        if label is None or label == np.nan:
            return 0
        elif 'distorted' in label.lower():
            return 1
        else:
            return 0

    y_true = df[label_column].apply(normalize_label)
    y_pred = df['isdistorted']
    # Calculate precision, recall, and F1-score using sklearn metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"✓/✗ Precision threshold (≥ 0.6) met/not met")

# Integrity Check
total_distortion_clusters = len(clusters) - sum(len(cluster) == 1 for cluster in clusters.values())
total_distorted_activities = df['isdistorted'].sum()
clean_activities = (df['isdistorted'] == 0).sum()
print(f"Total distortion clusters detected: {total_distortion_clusters}")
print(f"Total activities marked as distorted: {total_distorted_activities}")
print(f"Activities to be fixed: {total_distorted_activities - clean_activities}")

# Fix Activities
df['Activity_fixed'] = df['canonical_activity']

# Save Output
selected_columns = [case_column, timestamp_column, 'Variant', 'original_activity', 'Activity_fixed']
if label_column in df.columns:
    selected_columns.append(label_column)
df_output = df[selected_columns]
df_output[timestamp_column] = pd.to_datetime(df_output[timestamp_column])
df_output[timestamp_column] = df_output[timestamp_column].dt.strftime('%Y-%m-%d %H:%M:%S')
df_output.to_csv(output_file, index=False)

# Summary Statistics
print(f"Run 2: Processed dataset saved to: {output_file}")
print(f"Run 2: Final dataset shape: {df_output.shape}")
print(f"Run 2: Dataset: credit")
print(f"Run 2: Task type: distorted")
print(f"Run 2: Total number of events: {df_output.shape[0]}")
print(f"Run 2: Number of distorted events detected: {total_distorted_activities}")
print(f"Run 2: Unique activities before fixing: {len(unique_activities)}")
print(f"Run 2: Unique activities after fixing: {len(canonical_mapping)}")
print(f"Run 2: Activity reduction count and percentage: {(len(unique_activities) - len(canonical_mapping))} ({((len(unique_activities) - len(canonical_mapping)) / len(unique_activities)) * 100:.2f}%)")
print(f"Run 2: Output file path: {output_file}")
print(f"Run 2: Sample of up to 10 transformations showing: original → canonical")
print(df_output.head(10))