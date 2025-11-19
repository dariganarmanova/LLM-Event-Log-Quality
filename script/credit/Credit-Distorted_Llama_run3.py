# Generated script for Credit-Distorted - Run 3
# Generated on: 2025-11-13T15:59:51.577426
# Model: meta-llama/Llama-3.1-8B-Instruct

import pandas as pd
import numpy as np
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
df = pd.read_csv('data/credit/Credit-Distorted.csv')

# Normalize CaseID → Case if needed (handle naming variations)
df['Case'] = df['Case'].str.lower()

# Store original Activity values in `original_activity` column for reference
df['original_activity'] = df['Activity']

# Identify Distorted Activities
df['isdistorted'] = df['Activity'].str.endswith(':distorted')
df['BaseActivity'] = df['Activity'].str.replace(':distorted', '')

# Preprocess Activity Names
df['ProcessedActivity'] = df['Activity'].apply(lambda x: ''.join(e for e in x if e.isalnum() or e.isspace()).lower().strip())
df['ProcessedActivity'] = df['ProcessedActivity'].apply(lambda x: ' '.join(x.split()))
df = df[df['ProcessedActivity'].str.len() >= 4]

# Calculate Jaccard N-gram Similarity
def generate_ngrams(text, ngram_size):
    return set(''.join(text[i:i+ngram_size]) for i in range(len(text)-ngram_size+1))

def jaccard_similarity(ngram_set1, ngram_set2):
    intersection = ngram_set1.intersection(ngram_set2)
    union = ngram_set1.union(ngram_set2)
    return len(intersection) / len(union) if union else 0.0

def find_similar_pairs(df, similarity_threshold):
    unique_activities = df['ProcessedActivity'].unique()
    similar_pairs = []
    for i in range(len(unique_activities)):
        for j in range(i+1, len(unique_activities)):
            activity1 = unique_activities[i]
            activity2 = unique_activities[j]
            ngram_set1 = generate_ngrams(activity1, 3)
            ngram_set2 = generate_ngrams(activity2, 3)
            similarity = jaccard_similarity(ngram_set1, ngram_set2)
            if similarity >= similarity_threshold:
                similar_pairs.append((activity1, activity2))
    return similar_pairs

similar_pairs = find_similar_pairs(df, similarity_threshold)

# Cluster Similar Activities (Union-Find)
def find(parent, i):
    if parent[i] != i:
        parent[i] = find(parent, parent[i])
    return parent[i]

def union(parent, rank, x, y):
    xroot = find(parent, x)
    yroot = find(parent, y)

    if rank[xroot] < rank[yroot]:
        parent[xroot] = yroot
    elif rank[xroot] > rank[yroot]:
        parent[yroot] = xroot
    else:
        parent[yroot] = xroot
        rank[xroot] += 1

def make_clusters(df, similar_pairs):
    parent = {}
    rank = {}
    for activity in df['ProcessedActivity'].unique():
        parent[activity] = activity
        rank[activity] = 0
    for pair in similar_pairs:
        union(parent, rank, pair[0], pair[1])
    clusters = defaultdict(list)
    for activity in df['ProcessedActivity'].unique():
        clusters[find(parent, activity)].append(activity)
    return clusters

clusters = make_clusters(df, similar_pairs)

# Majority Voting Within Clusters
def majority_voting(df, clusters):
    canonical_activities = {}
    for cluster in clusters.values():
        if len(cluster) > 1:
            original_activities = df[df['ProcessedActivity'].isin(cluster)]['original_activity'].value_counts().index
            canonical_activity = original_activities.iloc[0]
            canonical_activities.update({activity: canonical_activity for activity in cluster})
    df['canonical_activity'] = df['original_activity'].map(canonical_activities)
    df['is_distorted'] = df['original_activity'].apply(lambda x: 1 if x != df['canonical_activity'].iloc[0] else 0)
    return df

df = majority_voting(df, clusters)

# Calculate Detection Metrics
def normalize_labels(df, label_column):
    if label_column in df.columns:
        df[label_column] = df[label_column].apply(lambda x: 1 if 'distorted' in x.lower() else 0)
    return df

df = normalize_labels(df, 'label')

y_true = df['label'].values
y_pred = df['is_distorted'].values
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
activities_to_be_fixed = df[df['is_distorted'] == 1].shape[0]
print(f"Total distortion clusters detected: {total_distortion_clusters}")
print(f"Total activities marked as distorted: {total_activities_marked_as_distorted}")
print(f"Activities to be fixed: {activities_to_be_fixed}")

# Fix Activities
df['Activity_fixed'] = df['canonical_activity']

# Save Output
output_columns = ['Case', 'Timestamp', 'Variant', 'Activity', 'Activity_fixed']
if 'label' in df.columns:
    output_columns.append('label')
df_output = df[output_columns]
df_output['Timestamp'] = pd.to_datetime(df_output['Timestamp'])
df_output['Timestamp'] = df_output['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
df_output.to_csv('data/credit/credit_distorted_cleaned_run3.csv', index=False)

# Summary Statistics
print(f"Run 3: Total number of events: {df.shape[0]}")
print(f"Run 3: Number of distorted events detected: {total_activities_marked_as_distorted}")
print(f"Run 3: Unique activities before fixing: {len(df['original_activity'].unique())}")
print(f"Run 3: Unique activities after fixing: {len(df['canonical_activity'].unique())}")
print(f"Run 3: Activity reduction count: {len(df['original_activity'].unique()) - len(df['canonical_activity'].unique())}")
print(f"Run 3: Activity reduction percentage: {(len(df['original_activity'].unique()) - len(df['canonical_activity'].unique())) / len(df['original_activity'].unique()) * 100:.2f}%")
print(f"Run 3: Output file path: data/credit/credit_distorted_cleaned_run3.csv")

# Print sample of up to 10 transformations
transformations = df[df['is_distorted'] == 1][['original_activity', 'canonical_activity']].values
print(f"Run 3: Sample of transformations:")
for i in range(min(10, transformations.shape[0])):
    print(f"Original: {transformations[i, 0]} → Canonical: {transformations[i, 1]}")