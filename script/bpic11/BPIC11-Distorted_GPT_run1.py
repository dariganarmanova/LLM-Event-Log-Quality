# Generated script for BPIC11-Distorted - Run 1
# Generated on: 2025-11-13T11:25:23.662512
# Model: gpt-4o-2024-11-20

import pandas as pd
import re
from itertools import combinations
from collections import defaultdict
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
ngram_size = 3
min_length = 4

input_file = 'data/bpic11/BPIC11-Distorted.csv'
output_file = 'data/bpic11/bpic11_distorted_cleaned_run1.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
distorted_suffix = ':distorted'

# Helper functions
def preprocess_activity(activity):
    """Normalize activity names for comparison."""
    if not case_sensitive:
        activity = activity.lower()
    activity = re.sub(r'[^a-zA-Z0-9\s]', '', activity)  # Remove non-alphanumeric except spaces
    activity = re.sub(r'\s+', ' ', activity).strip()  # Collapse multiple spaces
    return activity

def generate_ngrams(text, n):
    """Generate n-grams from a given text."""
    text = preprocess_activity(text)
    if len(text) < n:
        return set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}

def jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union

def union_find_find(parent, x):
    """Find the root parent of a node with path compression."""
    if parent[x] != x:
        parent[x] = union_find_find(parent, parent[x])
    return parent[x]

def union_find_union(parent, rank, x, y):
    """Union two clusters."""
    root_x = union_find_find(parent, x)
    root_y = union_find_find(parent, y)
    if root_x != root_y:
        if rank[root_x] > rank[root_y]:
            parent[root_y] = root_x
        elif rank[root_x] < rank[root_y]:
            parent[root_x] = root_y
        else:
            parent[root_y] = root_x
            rank[root_x] += 1

# Step 1: Load CSV
try:
    df = pd.read_csv(input_file)
    print(f"Run 1: Original dataset shape: {df.shape}")
    required_columns = {case_column, activity_column, timestamp_column}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required_columns - set(df.columns)}")
    df['original_activity'] = df[activity_column]
except Exception as e:
    print(f"Error loading file: {e}")
    exit(1)

# Step 2: Identify Distorted Activities
df['isdistorted'] = df[activity_column].str.endswith(distorted_suffix).astype(int)
df['BaseActivity'] = df[activity_column].str.replace(distorted_suffix, '', regex=False)

# Step 3: Preprocess Activity Names
df['ProcessedActivity'] = df['BaseActivity'].apply(preprocess_activity)
df = df[df['ProcessedActivity'].str.len() >= min_length]

# Step 4: Calculate Jaccard N-gram Similarity
unique_activities = df['ProcessedActivity'].unique()
similar_pairs = []

for act1, act2 in combinations(unique_activities, 2):
    ngrams1 = generate_ngrams(act1, ngram_size)
    ngrams2 = generate_ngrams(act2, ngram_size)
    similarity = jaccard_similarity(ngrams1, ngrams2)
    if similarity >= similarity_threshold:
        similar_pairs.append((act1, act2))

# Step 5: Cluster Similar Activities (Union-Find)
parent = {activity: activity for activity in unique_activities}
rank = {activity: 0 for activity in unique_activities}

for act1, act2 in similar_pairs:
    union_find_union(parent, rank, act1, act2)

clusters = defaultdict(list)
for activity in unique_activities:
    root = union_find_find(parent, activity)
    clusters[root].append(activity)

# Step 6: Majority Voting Within Clusters
canonical_mapping = {}
for cluster in clusters.values():
    if len(cluster) < 2:
        continue
    original_forms = df[df['ProcessedActivity'].isin(cluster)]['original_activity']
    most_common = original_forms.value_counts().idxmax()
    for variant in cluster:
        canonical_mapping[variant] = most_common

df['canonical_activity'] = df['ProcessedActivity'].map(canonical_mapping).fillna(df['BaseActivity'])
df['is_distorted'] = (df['original_activity'] != df['canonical_activity']).astype(int)

# Step 7: Calculate Detection Metrics
if label_column in df.columns:
    df['y_true'] = df[label_column].fillna('').str.contains('distorted', case=False).astype(int)
    y_true = df['y_true']
    y_pred = df['is_distorted']
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print("=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
else:
    print("No labels available for metric calculation.")

# Step 8: Integrity Check
total_clusters = len(clusters)
distorted_count = df['is_distorted'].sum()
clean_count = len(df) - distorted_count
print(f"Total distortion clusters detected: {total_clusters}")
print(f"Total activities marked as distorted: {distorted_count}")
print(f"Clean activities: {clean_count}")

# Step 9: Fix Activities
df['Activity_fixed'] = df['canonical_activity']

# Step 10: Save Output
try:
    df.to_csv(output_file, index=False)
    print(f"Run 1: Processed dataset saved to: {output_file}")
except Exception as e:
    print(f"Error saving file: {e}")
    exit(1)

# Step 11: Summary Statistics
print(f"Run 1: Final dataset shape: {df.shape}")
print(f"Unique activities before fixing: {len(unique_activities)}")
print(f"Unique activities after fixing: {df['Activity_fixed'].nunique()}")
print(f"Activity reduction count: {len(unique_activities) - df['Activity_fixed'].nunique()}")
print(f"Output file path: {output_file}")