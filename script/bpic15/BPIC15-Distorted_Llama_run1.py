# Generated script for BPIC15-Distorted - Run 1
# Generated on: 2025-11-13T14:32:55.083486
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
input_file = './data/bpic15/BPIC15-Distorted.csv'
output_file = 'data/bpic15/bpic15_distorted_cleaned_run1.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
distorted_suffix = ':distorted'

df = pd.read_csv(input_file)
print(f"Run 1: Original dataset shape: {df.shape}")

# Ensure required columns exist
required_columns = [case_column, activity_column, timestamp_column, label_column]
if not all(col in df.columns for col in required_columns):
    raise ValueError("Missing required columns")

# Normalize CaseID → Case if needed (handle naming variations)
df[case_column] = df[case_column].str.lower()

# Store original Activity values in `original_activity` column for reference
df['original_activity'] = df[activity_column].copy()

# Identify Distorted Activities
df['isdistorted'] = df[activity_column].str.endswith(distorted_suffix)
df['BaseActivity'] = df[activity_column].str.removesuffix(distorted_suffix)

# Preprocess Activity Names
def preprocess_activity(activity):
    if case_sensitive:
        return activity
    else:
        activity = activity.lower()
    activity = re.sub(r'[^a-z0-9\s]', '', activity)
    activity = re.sub(r'\s+', ' ', activity)
    activity = activity.strip()
    return activity

df['ProcessedActivity'] = df[activity_column].apply(preprocess_activity)
df = df[df['ProcessedActivity'].str.len() >= 4]

# Calculate Jaccard N-gram Similarity
def jaccard_similarity(set1, set2):
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if not union:
        return 0.0
    return len(intersection) / len(union)

def generate_ngrams(text, n):
    return set(text[i:i+n] for i in range(len(text) - n + 1))

def find_similar_pairs(df, threshold):
    similar_pairs = []
    unique_activities = df['ProcessedActivity'].unique()
    for i in range(len(unique_activities)):
        for j in range(i+1, len(unique_activities)):
            activity1 = unique_activities[i]
            activity2 = unique_activities[j]
            ngrams1 = generate_ngrams(activity1, ngram_size)
            ngrams2 = generate_ngrams(activity2, ngram_size)
            similarity = jaccard_similarity(ngrams1, ngrams2)
            if similarity >= threshold:
                similar_pairs.append((activity1, activity2))
    return similar_pairs

similar_pairs = find_similar_pairs(df, similarity_threshold)

# Cluster Similar Activities (Union-Find)
def find(parent, i):
    if parent[i] != i:
        parent[i] = find(parent, parent[i])
    return parent[i]

def union(parent, rank, x, y):
    x_root = find(parent, x)
    y_root = find(parent, y)
    if x_root == y_root:
        return
    if rank[x_root] < rank[y_root]:
        parent[x_root] = y_root
    elif rank[x_root] > rank[y_root]:
        parent[y_root] = x_root
    else:
        parent[y_root] = x_root
        rank[x_root] += 1

parent = {i: i for i in range(len(unique_activities))}
rank = {i: 0 for i in range(len(unique_activities))}
for pair in similar_pairs:
    union(parent, rank, unique_activities.index(pair[0]), unique_activities.index(pair[1]))

# Majority Voting Within Clusters
def majority_voting(cluster):
    frequency = defaultdict(int)
    for activity in cluster:
        frequency[df.loc[df['ProcessedActivity'] == activity, 'original_activity'].iloc[0]] += 1
    canonical_activity = max(frequency, key=frequency.get)
    return canonical_activity

clusters = defaultdict(list)
for i in range(len(unique_activities)):
    root = find(parent, i)
    clusters[root].append(unique_activities[i])
canonical_activities = {activity: majority_voting(cluster) for cluster in clusters.values() if len(cluster) > 1}

# Mark distorted activities
df['canonical_activity'] = df['ProcessedActivity'].map(canonical_activities)
df['is_distorted'] = df['original_activity'].ne(df['canonical_activity']).astype(int)

# Calculate Detection Metrics (BEFORE FIXING)
def normalize_label(label):
    if label is None or pd.isna(label):
        return 0
    elif 'distorted' in label.lower():
        return 1
    else:
        return 0

if label_column in df.columns:
    y_true = df[label_column].apply(normalize_label)
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
total_distortion_clusters = len([cluster for cluster in clusters.values() if len(cluster) > 1])
total_activities_marked_as_distorted = df['is_distorted'].sum()
activities_to_be_fixed = df[~df['original_activity'].isin(canonical_activities.values)].shape[0]
print(f"Total distortion clusters detected: {total_distortion_clusters}")
print(f"Total activities marked as distorted: {total_activities_marked_as_distorted}")
print(f"Activities to be fixed: {activities_to_be_fixed}")

# Fix Activities
df['Activity_fixed'] = df['canonical_activity']

# Save Output
output_columns = [case_column, timestamp_column, label_column, 'original_activity', 'Activity_fixed']
if label_column in df.columns:
    output_columns.append(label_column)
df_output = df[output_columns]
df_output[timestamp_column] = pd.to_datetime(df_output[timestamp_column])
df_output[timestamp_column] = df_output[timestamp_column].dt.strftime('%Y-%m-%d %H:%M:%S')
df_output.to_csv(output_file, index=False)

# Summary Statistics
print(f"Run 1: Processed dataset saved to: {output_file}")
print(f"Run 1: Final dataset shape: {df_output.shape}")
print(f"Run 1: Dataset: bpic15")
print(f"Run 1: Task type: distorted")
print(f"Run 1: Total number of events: {df.shape[0]}")
print(f"Run 1: Number of distorted events detected: {total_activities_marked_as_distorted}")
print(f"Run 1: Unique activities before fixing: {len(unique_activities)}")
print(f"Run 1: Unique activities after fixing: {len(canonical_activities)}")
activity_reduction_count = len(unique_activities) - len(canonical_activities)
activity_reduction_percentage = (activity_reduction_count / len(unique_activities)) * 100
print(f"Run 1: Activity reduction count: {activity_reduction_count}")
print(f"Run 1: Activity reduction percentage: {activity_reduction_percentage:.2f}%")
print(f"Run 1: Output file path: {output_file}")
print(f"Run 1: Sample of up to 10 transformations:")
print(df_output.head(10))