# Generated script for BPIC15-Distorted - Run 1
# Generated on: 2025-11-13T14:40:13.338058
# Model: gpt-4o-2024-11-20

import pandas as pd
import re
from itertools import combinations
from collections import defaultdict, Counter
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration parameters
input_file = 'data/bpic15/BPIC15-Distorted.csv'
output_file = 'data/bpic15/bpic15_distorted_cleaned_run1.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
distorted_suffix = ':distorted'
ngram_size = 3
similarity_threshold = 0.56
min_length = 4

# Helper functions
def generate_ngrams(text, n):
    """Generate n-grams from a string."""
    text = text.lower()
    return set([text[i:i+n] for i in range(len(text) - n + 1)])

def jaccard_similarity(set1, set2):
    """Compute Jaccard similarity between two sets."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union

def preprocess_activity(activity):
    """Normalize activity names for comparison."""
    activity = activity.lower()
    activity = re.sub(r'[^a-z0-9\s]', '', activity)
    activity = re.sub(r'\s+', ' ', activity).strip()
    return activity

def union_find_init(elements):
    """Initialize union-find structure."""
    return {e: e for e in elements}

def union_find_find(parent, element):
    """Find the root of an element with path compression."""
    if parent[element] != element:
        parent[element] = union_find_find(parent, parent[element])
    return parent[element]

def union_find_union(parent, a, b):
    """Union two elements."""
    root_a = union_find_find(parent, a)
    root_b = union_find_find(parent, b)
    if root_a != root_b:
        parent[root_b] = root_a

# Load data
try:
    df = pd.read_csv(input_file)
    print(f"Run 1: Original dataset shape: {df.shape}")
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# Ensure required columns exist
required_columns = {case_column, activity_column, timestamp_column}
if not required_columns.issubset(df.columns):
    print(f"Missing required columns: {required_columns - set(df.columns)}")
    exit()

# Step 1: Add original_activity column
df['original_activity'] = df[activity_column]

# Step 2: Identify distorted activities
df['isdistorted'] = df[activity_column].str.endswith(distorted_suffix).astype(int)
df['BaseActivity'] = df[activity_column].str.replace(distorted_suffix, '', regex=False)

# Step 3: Preprocess activity names
df['ProcessedActivity'] = df['BaseActivity'].apply(preprocess_activity)
df = df[df['ProcessedActivity'].str.len() >= min_length]

# Step 4: Calculate Jaccard n-gram similarity
unique_activities = df['ProcessedActivity'].unique()
parent = union_find_init(unique_activities)

for a, b in combinations(unique_activities, 2):
    ngrams_a = generate_ngrams(a, ngram_size)
    ngrams_b = generate_ngrams(b, ngram_size)
    similarity = jaccard_similarity(ngrams_a, ngrams_b)
    if similarity >= similarity_threshold:
        union_find_union(parent, a, b)

# Step 5: Cluster similar activities
clusters = defaultdict(list)
for activity in unique_activities:
    root = union_find_find(parent, activity)
    clusters[root].append(activity)

# Step 6: Majority voting within clusters
canonical_mapping = {}
for cluster, activities in clusters.items():
    original_activities = df[df['ProcessedActivity'].isin(activities)]['original_activity']
    most_common = Counter(original_activities).most_common(1)[0][0]
    for activity in activities:
        canonical_mapping[activity] = most_common

# Step 7: Apply canonical mapping
df['canonical_activity'] = df['ProcessedActivity'].map(canonical_mapping)
df['is_distorted'] = (df['original_activity'] != df['canonical_activity']).astype(int)

# Step 8: Calculate detection metrics
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

# Step 9: Replace activity column with canonical values
df['Activity_fixed'] = df['canonical_activity']

# Step 10: Save output
output_columns = [case_column, timestamp_column, activity_column, 'Activity_fixed']
if label_column in df.columns:
    output_columns.append(label_column)
df[output_columns].to_csv(output_file, index=False)

# Step 11: Summary statistics
print(f"Run 1: Total events processed: {len(df)}")
print(f"Run 1: Distorted events detected: {df['is_distorted'].sum()}")
print(f"Run 1: Unique activities before fixing: {len(unique_activities)}")
print(f"Run 1: Unique activities after fixing: {len(set(canonical_mapping.values()))}")
print(f"Run 1: Processed dataset saved to: {output_file}")