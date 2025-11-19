# Generated script for Credit-Distorted - Run 1
# Generated on: 2025-11-13T15:11:07.072826
# Model: gpt-4o-2024-11-20

import pandas as pd
import re
from itertools import combinations
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration parameters
input_file = 'data/credit/Credit-Distorted.csv'
output_file = 'data/credit/credit_distorted_cleaned_run1.csv'
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
    return set(text[i:i + n] for i in range(len(text) - n + 1))

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

# Step 1: Load CSV
try:
    df = pd.read_csv(input_file)
    required_columns = {case_column, activity_column, timestamp_column}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required_columns - set(df.columns)}")
    df['original_activity'] = df[activity_column]
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

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
    set1, set2 = generate_ngrams(act1, ngram_size), generate_ngrams(act2, ngram_size)
    if jaccard_similarity(set1, set2) >= similarity_threshold:
        similar_pairs.append((act1, act2))

# Step 5: Cluster Similar Activities (Union-Find)
parent = {activity: activity for activity in unique_activities}

def find(activity):
    if parent[activity] != activity:
        parent[activity] = find(parent[activity])
    return parent[activity]

def union(activity1, activity2):
    root1, root2 = find(activity1), find(activity2)
    if root1 != root2:
        parent[root2] = root1

for act1, act2 in similar_pairs:
    union(act1, act2)

clusters = defaultdict(list)
for activity in unique_activities:
    clusters[find(activity)].append(activity)

# Step 6: Majority Voting Within Clusters
canonical_mapping = {}
for cluster in clusters.values():
    if len(cluster) > 1:
        original_forms = df[df['ProcessedActivity'].isin(cluster)]['original_activity']
        canonical_form = original_forms.value_counts().idxmax()
        for variant in cluster:
            canonical_mapping[variant] = canonical_form

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
total_clusters = len([c for c in clusters.values() if len(c) > 1])
total_distorted = df['is_distorted'].sum()
total_clean = len(df) - total_distorted
print(f"Total distortion clusters detected: {total_clusters}")
print(f"Total activities marked as distorted: {total_distorted}")
print(f"Clean activities: {total_clean}")

# Step 9: Fix Activities
df['Activity_fixed'] = df['canonical_activity']

# Step 10: Save Output
output_columns = [case_column, timestamp_column, activity_column, 'Activity_fixed']
if label_column in df.columns:
    output_columns.append(label_column)
df[output_columns].to_csv(output_file, index=False)

# Step 11: Summary Statistics
unique_before = df['original_activity'].nunique()
unique_after = df['Activity_fixed'].nunique()
reduction = unique_before - unique_after
reduction_pct = (reduction / unique_before) * 100
print(f"Total events: {len(df)}")
print(f"Unique activities before fixing: {unique_before}")
print(f"Unique activities after fixing: {unique_after}")
print(f"Activity reduction: {reduction} ({reduction_pct:.2f}%)")
print(f"Processed dataset saved to: {output_file}")