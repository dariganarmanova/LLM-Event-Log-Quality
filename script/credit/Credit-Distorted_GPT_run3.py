# Generated script for Credit-Distorted - Run 3
# Generated on: 2025-11-13T15:11:33.298567
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
ngram_size = 3
min_length = 4
case_sensitive = False
use_fuzzy_matching = False

# File paths
input_file = 'data/credit/Credit-Distorted.csv'
output_file = 'data/credit/credit_distorted_cleaned_run3.csv'

# Helper functions
def preprocess_activity(activity):
    """Normalize activity names for comparison."""
    activity = activity.lower() if not case_sensitive else activity
    activity = re.sub(r'[^a-zA-Z0-9\s]', '', activity)
    activity = re.sub(r'\s+', ' ', activity).strip()
    return activity

def generate_ngrams(text, n):
    """Generate character n-grams for a given text."""
    return set(text[i:i+n] for i in range(len(text) - n + 1))

def jaccard_similarity(val1, val2):
    """Calculate Jaccard similarity between two strings based on n-grams."""
    ngrams1 = generate_ngrams(val1, ngram_size)
    ngrams2 = generate_ngrams(val2, ngram_size)
    if not ngrams1 or not ngrams2:
        return 0.0
    intersection = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)
    return intersection / union

def union_find_find(parent, x):
    """Find the root parent of a node with path compression."""
    if parent[x] != x:
        parent[x] = union_find_find(parent, parent[x])
    return parent[x]

def union_find_union(parent, rank, x, y):
    """Union two sets in the union-find structure."""
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
    print(f"Run 3: Original dataset shape: {df.shape}")
    required_columns = ['Case', 'Activity', 'Timestamp']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    df['original_activity'] = df['Activity']
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Step 2: Identify Distorted Activities
df['isdistorted'] = df['Activity'].str.endswith(':distorted').astype(int)
df['BaseActivity'] = df['Activity'].str.replace(activity_suffix_pattern, '', regex=True)

# Step 3: Preprocess Activity Names
df['ProcessedActivity'] = df['BaseActivity'].apply(preprocess_activity)
df = df[df['ProcessedActivity'].str.len() >= min_length]

# Step 4: Calculate Jaccard Similarity
unique_activities = df['ProcessedActivity'].unique()
similar_pairs = []
for act1, act2 in combinations(unique_activities, 2):
    if jaccard_similarity(act1, act2) >= similarity_threshold:
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
for cluster, activities in clusters.items():
    if len(activities) > 1:
        original_forms = df[df['ProcessedActivity'].isin(activities)]['original_activity']
        canonical_form = original_forms.value_counts().idxmax()
        for activity in activities:
            canonical_mapping[activity] = canonical_form

df['canonical_activity'] = df['ProcessedActivity'].map(canonical_mapping).fillna(df['original_activity'])
df['is_distorted'] = (df['original_activity'] != df['canonical_activity']).astype(int)

# Step 7: Calculate Detection Metrics
if 'label' in df.columns:
    df['y_true'] = df['label'].fillna('').str.contains('distorted', case=False).astype(int)
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
total_clusters = len([c for c in clusters.values() if len(c) > 1])
total_distorted = df['is_distorted'].sum()
print(f"Total distortion clusters detected: {total_clusters}")
print(f"Total activities marked as distorted: {total_distorted}")

# Step 9: Fix Activities
df['Activity_fixed'] = df['canonical_activity']

# Step 10: Save Output
try:
    df.to_csv(output_file, index=False)
    print(f"Run 3: Processed dataset saved to: {output_file}")
except Exception as e:
    print(f"Error saving dataset: {e}")
    exit()

# Step 11: Summary Statistics
unique_before = df['original_activity'].nunique()
unique_after = df['Activity_fixed'].nunique()
print(f"Run 3: Total events: {len(df)}")
print(f"Run 3: Unique activities before fixing: {unique_before}")
print(f"Run 3: Unique activities after fixing: {unique_after}")
print(f"Run 3: Activity reduction: {unique_before - unique_after} ({(unique_before - unique_after) / unique_before:.2%})")