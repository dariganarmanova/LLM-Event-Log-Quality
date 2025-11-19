# Generated script for BPIC11-Distorted - Run 2
# Generated on: 2025-11-13T11:25:48.047366
# Model: gpt-4o-2024-11-20

import pandas as pd
import re
from itertools import combinations
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration parameters
input_file = 'data/bpic11/BPIC11-Distorted.csv'
output_file = 'data/bpic11/bpic11_distorted_cleaned_run2.csv'
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
    """Generate character n-grams from a given text."""
    return {text[i:i + n] for i in range(len(text) - n + 1)}

def jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets."""
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0

def preprocess_activity(activity):
    """Normalize activity names for comparison."""
    activity = activity.lower()
    activity = re.sub(r'[^a-z0-9\s]', '', activity)
    activity = re.sub(r'\s+', ' ', activity).strip()
    return activity

# Load the data
try:
    df = pd.read_csv(input_file)
    print(f"Run 2: Original dataset shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: File {input_file} not found.")
    exit(1)

# Ensure required columns exist
required_columns = {case_column, activity_column, timestamp_column}
if not required_columns.issubset(df.columns):
    print(f"Error: Missing required columns. Ensure {required_columns} are in the dataset.")
    exit(1)

# Step 1: Add original_activity and isdistorted columns
df['original_activity'] = df[activity_column]
df['isdistorted'] = df[activity_column].str.endswith(distorted_suffix).astype(int)
df['BaseActivity'] = df[activity_column].str.replace(distorted_suffix, '', regex=False)

# Step 2: Preprocess activities
df['ProcessedActivity'] = df['BaseActivity'].apply(preprocess_activity)
df = df[df['ProcessedActivity'].str.len() >= min_length]

# Step 3: Calculate Jaccard similarity and find similar pairs
unique_activities = df['ProcessedActivity'].unique()
similar_pairs = []

for act1, act2 in combinations(unique_activities, 2):
    sim = jaccard_similarity(generate_ngrams(act1, ngram_size), generate_ngrams(act2, ngram_size))
    if sim >= similarity_threshold:
        similar_pairs.append((act1, act2))

# Step 4: Union-Find for clustering
parent = {activity: activity for activity in unique_activities}

def find(activity):
    if parent[activity] != activity:
        parent[activity] = find(parent[activity])
    return parent[activity]

def union(activity1, activity2):
    root1 = find(activity1)
    root2 = find(activity2)
    if root1 != root2:
        parent[root2] = root1

for act1, act2 in similar_pairs:
    union(act1, act2)

clusters = defaultdict(list)
for activity in unique_activities:
    clusters[find(activity)].append(activity)

# Step 5: Determine canonical forms
canonical_mapping = {}
for cluster in clusters.values():
    if len(cluster) > 1:
        original_forms = df[df['ProcessedActivity'].isin(cluster)]['BaseActivity']
        canonical_form = original_forms.value_counts().idxmax()
        for variant in cluster:
            canonical_mapping[variant] = canonical_form

# Step 6: Apply canonical mapping
df['canonical_activity'] = df['ProcessedActivity'].map(canonical_mapping).fillna(df['BaseActivity'])
df['isdistorted'] = (df['BaseActivity'] != df['canonical_activity']).astype(int)

# Step 7: Calculate detection metrics
if label_column in df.columns:
    df['y_true'] = df[label_column].fillna('').str.contains('distorted', case=False).astype(int)
    y_true = df['y_true']
    y_pred = df['isdistorted']
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print("=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
else:
    print("No labels available for metric calculation.")

# Step 8: Save the cleaned dataset
df['Activity_fixed'] = df['canonical_activity']
output_columns = [case_column, timestamp_column, activity_column, 'Activity_fixed']
if label_column in df.columns:
    output_columns.append(label_column)

df.to_csv(output_file, columns=output_columns, index=False)

# Step 9: Print summary statistics
print(f"Run 2: Processed dataset saved to: {output_file}")
print(f"Run 2: Final dataset shape: {df.shape}")
print(f"Run 2: Total events: {len(df)}")
print(f"Run 2: Distorted events detected: {df['isdistorted'].sum()}")
print(f"Run 2: Unique activities before fixing: {len(unique_activities)}")
print(f"Run 2: Unique activities after fixing: {df['canonical_activity'].nunique()}")