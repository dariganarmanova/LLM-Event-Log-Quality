# Generated script for BPIC15-Distorted - Run 2
# Generated on: 2025-11-13T14:32:58.555161
# Model: meta-llama/Llama-3.1-8B-Instruct

import pandas as pd
import re
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk import ngrams as nltk_ngrams
from nltk.tokenize import word_tokenize
from nltk import ngrams as nltk_ngrams
from collections import defaultdict
from typing import Dict, List, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score

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
df = pd.read_csv('data/bpic15/BPIC15-Distorted.csv')

# Normalize CaseID → Case if needed (handle naming variations)
df['Case'] = df['Case'].astype(str).str.lower()

# Store original Activity values in `original_activity` column for reference
df['original_activity'] = df['Activity'].copy()

# Identify Distorted Activities
df['isdistorted'] = df['Activity'].str.endswith(':distorted')
df['BaseActivity'] = df['Activity'].str.replace(':distorted', '')

# Preprocess Activity Names
def preprocess_activity(activity: str) -> str:
    activity = activity.lower()
    activity = re.sub(r'[^a-z0-9\s]', '', activity)
    activity = ' '.join(activity.split())
    return activity

df['ProcessedActivity'] = df['Activity'].apply(preprocess_activity)

# Filter out activities shorter than min_length characters (too short for meaningful comparison)
min_length = 4
df = df[df['ProcessedActivity'].str.len() >= min_length]

# Calculate Jaccard N-gram Similarity
def jaccard_similarity(set1: set, set2: set) -> float:
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if len(union) == 0:
        return 0.0
    return len(intersection) / len(union)

def generate_ngrams(text: str, n: int) -> set:
    return set(ngrams(text, n))

def find_similar_pairs(df: pd.DataFrame, n: int, threshold: float) -> List[Tuple[str, str]]:
    unique_activities = df['ProcessedActivity'].unique()
    similar_pairs = []
    for i in range(len(unique_activities)):
        for j in range(i+1, len(unique_activities)):
            activity1 = unique_activities[i]
            activity2 = unique_activities[j]
            ngram_set1 = generate_ngrams(activity1, n)
            ngram_set2 = generate_ngrams(activity2, n)
            similarity = jaccard_similarity(ngram_set1, ngram_set2)
            if similarity >= threshold:
                similar_pairs.append((activity1, activity2))
    return similar_pairs

similar_pairs = find_similar_pairs(df, 3, similarity_threshold)

# Cluster Similar Activities (Union-Find)
def find(parent: Dict[str, str], x: str) -> str:
    if parent[x] != x:
        parent[x] = find(parent, parent[x])
    return parent[x]

def union(parent: Dict[str, str], x: str, y: str) -> None:
    root_x = find(parent, x)
    root_y = find(parent, y)
    if root_x != root_y:
        parent[root_x] = root_y

parent = {activity: activity for activity in df['ProcessedActivity'].unique()}
for pair in similar_pairs:
    union(parent, pair[0], pair[1])

# Majority Voting Within Clusters
def majority_voting(cluster: List[str]) -> str:
    frequency = defaultdict(int)
    for activity in cluster:
        frequency[df.loc[df['ProcessedActivity'] == activity, 'original_activity'].iloc[0]] += 1
    return max(frequency, key=frequency.get)

clusters = defaultdict(list)
for activity in df['ProcessedActivity']:
    clusters[find(parent, activity)].append(activity)

canonical_mapping = {}
for cluster, activities in clusters.items():
    if len(activities) > 1:
        canonical_activity = majority_voting(activities)
        for activity in activities:
            canonical_mapping[df.loc[df['ProcessedActivity'] == activity, 'original_activity'].iloc[0]] = canonical_activity

# Mark distorted activities
df['is_distorted'] = df['original_activity'].apply(lambda x: 1 if x in canonical_mapping else 0)
df['canonical_activity'] = df['original_activity'].apply(lambda x: canonical_mapping[x] if x in canonical_mapping else x)

# Calculate Detection Metrics (BEFORE FIXING)
def normalize_label(label: str) -> int:
    if pd.isnull(label) or label == 'nan':
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
else:
    print("No labels available for metric calculation")

# Integrity Check
total_distortion_clusters = len([cluster for cluster in clusters.values() if len(cluster) > 1])
total_distorted_activities = df['is_distorted'].sum()
clean_activities = df['is_distorted'].sum() - total_distortion_clusters
print(f"Total distortion clusters detected: {total_distortion_clusters}")
print(f"Total activities marked as distorted: {total_distorted_activities}")
print(f"Activities to be fixed: {clean_activities}")

# Fix Activities
df['Activity_fixed'] = df['canonical_activity']

# Save Output
output_columns = ['Case', 'Timestamp', 'Variant', 'original_activity', 'Activity_fixed']
if 'label' in df.columns:
    output_columns.append('label')
df_output = df[output_columns]
df_output['Timestamp'] = pd.to_datetime(df_output['Timestamp'])
df_output['Timestamp'] = df_output['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
df_output.to_csv('data/bpic15/bpic15_distorted_cleaned_run2.csv', index=False)

# Summary Statistics
print(f"Run 2: Total number of events: {df.shape[0]}")
print(f"Run 2: Number of distorted events detected: {total_distorted_activities}")
print(f"Run 2: Unique activities before fixing: {len(df['original_activity'].unique())}")
print(f"Run 2: Unique activities after fixing: {len(df['canonical_activity'].unique())}")
print(f"Run 2: Activity reduction count and percentage: {(len(df['original_activity'].unique()) - len(df['canonical_activity'].unique())) / len(df['original_activity'].unique()) * 100:.2f}%")
print(f"Run 2: Output file path: data/bpic15/bpic15_distorted_cleaned_run2.csv")

# Print sample of up to 10 transformations showing: original → canonical
print("Run 2: Sample transformations:")
for i in range(min(10, len(df))):
    print(f"{df.loc[i, 'original_activity']} → {df.loc[i, 'canonical_activity']}")