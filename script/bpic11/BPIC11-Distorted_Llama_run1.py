# Generated script for BPIC11-Distorted - Run 1
# Generated on: 2025-11-13T11:27:34.575282
# Model: meta-llama/Llama-3.1-8B-Instruct

import pandas as pd
import re
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import defaultdict

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
df = pd.read_csv('data/bpic11/BPIC11-Distorted.csv')

# Normalize CaseID → Case if needed (handle naming variations)
df['Case'] = df['Case'].apply(lambda x: x.lower() if case_sensitive else x)

# Store original Activity values in `original_activity` column for reference
df['original_activity'] = df['Activity']

# Identify Distorted Activities
df['isdistorted'] = df['Activity'].str.endswith(activity_suffix_pattern)
df['BaseActivity'] = df['Activity'].str.replace(activity_suffix_pattern, '')

# Preprocess Activity Names
df['ProcessedActivity'] = df['Activity'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x).lower().strip().replace(' ', ' '))

# Filter out activities shorter than min_length characters (too short for meaningful comparison)
min_length = 4
df = df[df['ProcessedActivity'].str.len() >= min_length]

# Calculate Jaccard N-gram Similarity
def generate_ngrams(text, ngram_size):
    return set([text[i:i+ngram_size] for i in range(len(text) - ngram_size + 1)])

def jaccard_similarity(val1, val2):
    ngrams1 = generate_ngrams(val1, ngram_size=3)
    ngrams2 = generate_ngrams(val2, ngram_size=3)
    intersection = ngrams1.intersection(ngrams2)
    union = ngrams1.union(ngrams2)
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
        if jaccard_similarity(val1, val2) >= similarity_threshold:
            similar_pairs.append((val1, val2))

# Cluster Similar Activities (Union-Find)
parent = {activity: activity for activity in unique_activities}
def find(activity):
    if parent[activity] != activity:
        parent[activity] = find(parent[activity])
    return parent[activity]

def union(val1, val2):
    root1 = find(val1)
    root2 = find(val2)
    if root1 != root2:
        parent[root1] = root2

for pair in similar_pairs:
    union(pair[0], pair[1])

# Build clusters
clusters = defaultdict(list)
for activity in unique_activities:
    clusters[find(activity)].append(activity)

# Keep only clusters with 2 or more members (single activities don't need fixing)
clusters = {cluster: activities for cluster, activities in clusters.items() if len(activities) > 1}

# Majority Voting Within Clusters
canonical_activities = {}
for cluster, activities in clusters.items():
    frequency = {}
    for activity in activities:
        frequency[activity] = frequency.get(activity, 0) + 1
    canonical_activity = max(frequency, key=frequency.get)
    canonical_activities[cluster] = canonical_activity

# Mark distorted activities
df['canonical_activity'] = df['original_activity']
for cluster, canonical_activity in canonical_activities.items():
    df.loc[df['ProcessedActivity'].isin(cluster), 'canonical_activity'] = canonical_activity

# Calculate Detection Metrics (BEFORE FIXING)
if 'label' in df.columns:
    def normalize_label(label):
        if pd.isnull(label) or pd.isna(label):
            return 0
        if 'distorted' in label.lower():
            return 1
        return 0

    y_true = df['label'].apply(normalize_label)
    y_pred = df['isdistorted']
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"✓/✗ Precision threshold (≥ 0.6) met/not met")
else:
    print("No labels available for metric calculation")

# Integrity Check
total_distortion_clusters = len(canonical_activities)
total_distorted_activities = df['isdistorted'].sum()
activities_to_fix = df.loc[df['isdistorted'] == 1, 'original_activity'].nunique()
print(f"Total distortion clusters detected: {total_distortion_clusters}")
print(f"Total activities marked as distorted: {total_distorted_activities}")
print(f"Activities to be fixed: {activities_to_fix}")

# Fix Activities
df['Activity_fixed'] = df['canonical_activity']

# Save Output
output_columns = ['Case', 'Timestamp', 'Variant', 'Activity', 'Activity_fixed']
if 'label' in df.columns:
    output_columns.append('label')
df[output_columns].to_csv('data/bpic11/bpic11_distorted_cleaned_run1.csv', index=False)

# Summary Statistics
print(f"Run 1: Processed dataset saved to: data/bpic11/bpic11_distorted_cleaned_run1.csv")
print(f"Run 1: Final dataset shape: {df.shape}")
print(f"Run 1: Dataset: bpic11")
print(f"Run 1: Task type: distorted")
print(f"Run 1: Total number of events: {df.shape[0]}")
print(f"Run 1: Number of distorted events detected: {total_distorted_activities}")
print(f"Run 1: Unique activities before fixing: {df['original_activity'].nunique()}")
print(f"Run 1: Unique activities after fixing: {df['canonical_activity'].nunique()}")
print(f"Run 1: Activity reduction count: {df['original_activity'].nunique() - df['canonical_activity'].nunique()}")
print(f"Run 1: Activity reduction percentage: {(df['original_activity'].nunique() - df['canonical_activity'].nunique()) / df['original_activity'].nunique() * 100:.2f}%")
print(f"Run 1: Output file path: data/bpic11/bpic11_distorted_cleaned_run1.csv")

# Print sample of up to 10 transformations showing: original → canonical
transformations = df.loc[df['isdistorted'] == 1, ['original_activity', 'canonical_activity']]
print(transformations.head(10))