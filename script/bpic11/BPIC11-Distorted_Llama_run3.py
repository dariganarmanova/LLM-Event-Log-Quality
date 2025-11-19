# Generated script for BPIC11-Distorted - Run 3
# Generated on: 2025-11-13T11:27:49.672651
# Model: meta-llama/Llama-3.1-8B-Instruct

import pandas as pd
import re
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk import ngrams as nltk_ngrams
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import defaultdict
from typing import Dict, List, Tuple

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

# Load the data
df = pd.read_csv('data/bpic11/BPIC11-Distorted.csv')

# Normalize CaseID → Case if needed (handle naming variations)
df['Case'] = df['Case'].astype(str).str.lower()

# Store original Activity values in `original_activity` column for reference
df['original_activity'] = df['Activity']

# Identify Distorted Activities
df['isdistorted'] = df['Activity'].str.endswith(':distorted')
df['BaseActivity'] = df['Activity'].str.replace(':distorted', '')

# Preprocess Activity Names
def preprocess_activity(activity: str) -> str:
    if case_sensitive:
        return activity
    else:
        activity = activity.lower()
    activity = re.sub(r'[^a-zA-Z0-9\s]', '', activity)
    activity = ' '.join(activity.split())
    return activity

df['ProcessedActivity'] = df['Activity'].apply(preprocess_activity)

# Filter out activities shorter than min_length characters (too short for meaningful comparison)
df = df[df['ProcessedActivity'].str.len() >= min_length]

# Calculate Jaccard N-gram Similarity
def jaccard_similarity(activity1: str, activity2: str) -> float:
    if not activity1 or not activity2:
        return 0.0
    ngrams1 = set(nltk_ngrams(word_tokenize(activity1), ngram_size))
    ngrams2 = set(nltk_ngrams(word_tokenize(activity2), ngram_size))
    intersection = ngrams1.intersection(ngrams2)
    union = ngrams1.union(ngrams2)
    return len(intersection) / len(union)

# Find Similar Pairs
similar_pairs: List[Tuple[str, str]] = []
unique_activities = df['ProcessedActivity'].unique()
for i in range(len(unique_activities)):
    for j in range(i + 1, len(unique_activities)):
        activity1 = unique_activities[i]
        activity2 = unique_activities[j]
        similarity = jaccard_similarity(activity1, activity2)
        if similarity >= similarity_threshold:
            similar_pairs.append((activity1, activity2))

# Cluster Similar Activities (Union-Find)
def find(parent: Dict[str, str], activity: str) -> str:
    if parent[activity] != activity:
        parent[activity] = find(parent, parent[activity])
    return parent[activity]

def union(parent: Dict[str, str], activity1: str, activity2: str) -> None:
    root1 = find(parent, activity1)
    root2 = find(parent, activity2)
    if root1 != root2:
        parent[root1] = root2

parent = {activity: activity for activity in unique_activities}
for activity1, activity2 in similar_pairs:
    union(parent, activity1, activity2)

# Build clusters
clusters: Dict[str, List[str]] = defaultdict(list)
for activity in unique_activities:
    root = find(parent, activity)
    clusters[root].append(activity)

# Majority Voting Within Clusters
canonical_activities: Dict[str, str] = {}
for cluster, activities in clusters.items():
    if len(activities) > 1:
        activity_counts = {}
        for activity in activities:
            original_activity = df.loc[df['ProcessedActivity'] == activity, 'original_activity'].iloc[0]
            activity_counts[original_activity] = activity_counts.get(original_activity, 0) + 1
        canonical_activity = max(activity_counts, key=activity_counts.get)
        canonical_activities[cluster] = canonical_activity

# Mark distorted activities
df['is_distorted'] = 0
for index, row in df.iterrows():
    if row['original_activity'] != canonical_activities[find(parent, row['ProcessedActivity'])]:
        df.loc[index, 'is_distorted'] = 1

# Calculate Detection Metrics (BEFORE FIXING)
def normalize_label(label: str) -> int:
    if pd.isnull(label) or label == '':
        return 0
    elif 'distorted' in label.lower():
        return 1
    else:
        return 0

if 'label' in df.columns:
    y_true = df['label'].apply(normalize_label)
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
total_distortion_clusters = len(canonical_activities)
total_distorted_activities = df['is_distorted'].sum()
clean_activities = (df['is_distorted'] == 0).sum()
print(f"Total distortion clusters detected: {total_distortion_clusters}")
print(f"Total activities marked as distorted: {total_distorted_activities}")
print(f"Activities to be fixed: {total_distorted_activities - clean_activities}")

# Fix Activities
df['canonical_activity'] = df['ProcessedActivity'].apply(lambda x: canonical_activities[find(parent, x)])
df['Activity_fixed'] = df['canonical_activity']

# Save Output
output_df = df[['Case', 'Timestamp', 'Variant', 'original_activity', 'Activity_fixed']]
if 'label' in df.columns:
    output_df['label'] = df['label']
output_df['Timestamp'] = output_df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
output_df.to_csv('data/bpic11/bpic11_distorted_cleaned_run3.csv', index=False)

# Summary Statistics
print(f"Run 3: Processed dataset saved to: data/bpic11/bpic11_distorted_cleaned_run3.csv")
print(f"Run 3: Final dataset shape: {output_df.shape}")
print(f"Run 3: Dataset: bpic11")
print(f"Run 3: Task type: distorted")
print(f"Run 3: Total number of events: {output_df.shape[0]}")
print(f"Run 3: Number of distorted events detected: {total_distorted_activities}")
print(f"Run 3: Unique activities before fixing: {len(unique_activities)}")
print(f"Run 3: Unique activities after fixing: {len(canonical_activities)}")
print(f"Run 3: Activity reduction count and percentage: {(len(unique_activities) - len(canonical_activities))}/{len(unique_activities)}")
print(f"Run 3: Output file path: data/bpic11/bpic11_distorted_cleaned_run3.csv")

# Print sample of up to 10 transformations showing: original → canonical
print("Run 3: Sample transformations:")
for index, row in df.head(10).iterrows():
    print(f"{row['original_activity']} → {row['canonical_activity']}")