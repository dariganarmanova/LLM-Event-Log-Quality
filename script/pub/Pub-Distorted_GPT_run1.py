# Generated script for Pub-Distorted - Run 1
# Generated on: 2025-11-14T13:21:17.327011
# Model: gpt-4o-2024-11-20

import pandas as pd
import re
from itertools import combinations
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support

# Configuration parameters
input_file = 'data/pub/Pub-Distorted.csv'
output_file = 'data/pub/pub_distorted_cleaned_run1.csv'
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
    return set([text[i:i+n] for i in range(len(text) - n + 1)])

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
    print(f"Run 1: Original dataset shape: {df.shape}")
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# Ensure required columns exist
required_columns = [case_column, activity_column, timestamp_column]
for col in required_columns:
    if col not in df.columns:
        print(f"Missing required column: {col}")
        exit()

# Step 1: Store original activities
df['original_activity'] = df[activity_column]

# Step 2: Identify distorted activities
df['isdistorted'] = df[activity_column].str.endswith(distorted_suffix).astype(int)
df['BaseActivity'] = df[activity_column].str.replace(distorted_suffix, '', regex=False)

# Step 3: Preprocess activities
df['ProcessedActivity'] = df['BaseActivity'].apply(preprocess_activity)
df = df[df['ProcessedActivity'].str.len() >= min_length]

# Step 4: Calculate Jaccard similarity
unique_activities = df['ProcessedActivity'].unique()
similar_pairs = []

for act1, act2 in combinations(unique_activities, 2):
    sim = jaccard_similarity(generate_ngrams(act1, ngram_size), generate_ngrams(act2, ngram_size))
    if sim >= similarity_threshold:
        similar_pairs.append((act1, act2))

# Step 5: Cluster similar activities using Union-Find
parent = {activity: activity for activity in unique_activities}

def find(activity):
    if parent[activity] != activity:
        parent[activity] = find(parent[activity])
    return parent[activity]

def union(act1, act2):
    root1 = find(act1)
    root2 = find(act2)
    if root1 != root2:
        parent[root2] = root1

for act1, act2 in similar_pairs:
    union(act1, act2)

clusters = defaultdict(list)
for activity in unique_activities:
    clusters[find(activity)].append(activity)

# Step 6: Majority voting within clusters
canonical_mapping = {}
for cluster in clusters.values():
    if len(cluster) > 1:
        original_forms = df[df['ProcessedActivity'].isin(cluster)]['BaseActivity']
        canonical_form = original_forms.value_counts().idxmax()
        for variant in cluster:
            canonical_mapping[variant] = canonical_form

df['canonical_activity'] = df['ProcessedActivity'].map(canonical_mapping).fillna(df['BaseActivity'])
df['is_distorted'] = (df['BaseActivity'] != df['canonical_activity']).astype(int)

# Step 7: Detection metrics
if label_column in df.columns:
    df['y_true'] = df[label_column].str.contains('distorted', case=False, na=False).astype(int)
    y_true = df['y_true']
    y_pred = df['is_distorted']
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    print("=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
else:
    print("No labels available for metric calculation.")

# Step 8: Integrity check
total_clusters = len([c for c in clusters.values() if len(c) > 1])
total_distorted = df['is_distorted'].sum()
print(f"Total distortion clusters detected: {total_clusters}")
print(f"Total activities marked as distorted: {total_distorted}")

# Step 9: Fix activities
df['Activity_fixed'] = df['canonical_activity']

# Step 10: Save output
output_columns = [case_column, timestamp_column, 'original_activity', 'Activity_fixed']
if label_column in df.columns:
    output_columns.append(label_column)
df[output_columns].to_csv(output_file, index=False)

# Step 11: Summary statistics
unique_before = len(df['BaseActivity'].unique())
unique_after = len(df['canonical_activity'].unique())
print(f"Run 1: Total events: {len(df)}")
print(f"Run 1: Unique activities before fixing: {unique_before}")
print(f"Run 1: Unique activities after fixing: {unique_after}")
print(f"Run 1: Activity reduction: {unique_before - unique_after} ({((unique_before - unique_after) / unique_before) * 100:.2f}%)")
print(f"Run 1: Processed dataset saved to: {output_file}")