# Generated script for Pub-Distorted - Run 3
# Generated on: 2025-11-14T13:21:48.254822
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
input_file = 'data/pub/Pub-Distorted.csv'
output_file = 'data/pub/pub_distorted_cleaned_run3.csv'

# Helper functions
def generate_ngrams(text, n):
    """Generate character n-grams for a given text."""
    text = re.sub(r'\s+', ' ', text.strip())  # Normalize whitespace
    return set([text[i:i + n] for i in range(len(text) - n + 1)])

def jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets."""
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0

def preprocess_activity(activity):
    """Preprocess activity name for comparison."""
    activity = activity.lower() if not case_sensitive else activity
    activity = re.sub(r'[^a-zA-Z0-9\s]', '', activity)  # Remove non-alphanumeric chars
    activity = re.sub(r'\s+', ' ', activity).strip()  # Normalize whitespace
    return activity

# Load the dataset
try:
    df = pd.read_csv(input_file)
    print(f"Run 3: Original dataset shape: {df.shape}")
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# Ensure required columns exist
required_columns = ['Case', 'Activity', 'Timestamp']
optional_columns = ['Variant', 'Resource', 'label']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# Normalize column names if necessary
df.rename(columns=lambda x: x.strip(), inplace=True)

# Step 1: Store original activity values
df['original_activity'] = df['Activity']

# Step 2: Identify distorted activities
df['isdistorted'] = df['Activity'].str.endswith(':distorted').astype(int)
df['BaseActivity'] = df['Activity'].str.replace(activity_suffix_pattern, '', regex=True)

# Step 3: Preprocess activity names
df['ProcessedActivity'] = df['BaseActivity'].apply(preprocess_activity)
df = df[df['ProcessedActivity'].str.len() >= min_length]

# Step 4: Calculate Jaccard similarity
unique_activities = df['ProcessedActivity'].unique()
similar_pairs = []
for act1, act2 in combinations(unique_activities, 2):
    sim = jaccard_similarity(generate_ngrams(act1, ngram_size), generate_ngrams(act2, ngram_size))
    if sim >= similarity_threshold:
        similar_pairs.append((act1, act2))

# Step 5: Cluster similar activities using union-find
parent = {activity: activity for activity in unique_activities}

def find(activity):
    if parent[activity] != activity:
        parent[activity] = find(parent[activity])  # Path compression
    return parent[activity]

def union(activity1, activity2):
    root1, root2 = find(activity1), find(activity2)
    if root1 != root2:
        parent[root2] = root1

for act1, act2 in similar_pairs:
    union(act1, act2)

# Group activities by their root parent
clusters = defaultdict(list)
for activity in unique_activities:
    clusters[find(activity)].append(activity)

# Step 6: Majority voting within clusters
canonical_mapping = {}
for cluster, activities in clusters.items():
    original_forms = df[df['ProcessedActivity'].isin(activities)]['original_activity']
    most_common = original_forms.value_counts().idxmax()
    for activity in activities:
        canonical_mapping[activity] = most_common

# Apply canonical mapping
df['canonical_activity'] = df['ProcessedActivity'].map(canonical_mapping)
df['is_distorted'] = (df['original_activity'] != df['canonical_activity']).astype(int)

# Step 7: Calculate detection metrics
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

# Step 8: Integrity check
total_clusters = len(clusters)
total_distorted = df['is_distorted'].sum()
print(f"Total distortion clusters detected: {total_clusters}")
print(f"Total activities marked as distorted: {total_distorted}")

# Step 9: Fix activities
df['Activity_fixed'] = df['canonical_activity']

# Step 10: Save output
output_columns = ['Case', 'Timestamp', 'Variant', 'original_activity', 'Activity_fixed']
output_columns = [col for col in output_columns if col in df.columns]
df.to_csv(output_file, index=False)

# Step 11: Summary statistics
print(f"Run 3: Processed dataset saved to: {output_file}")
print(f"Run 3: Final dataset shape: {df.shape}")
print(f"Run 3: Unique activities before fixing: {len(unique_activities)}")
print(f"Run 3: Unique activities after fixing: {df['Activity_fixed'].nunique()}")
print(f"Run 3: Dataset: pub")
print(f"Run 3: Task type: distorted")