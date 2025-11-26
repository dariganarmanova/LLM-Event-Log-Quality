# Generated script for Pub-Distorted - Run 2
# Generated on: 2025-11-18T17:31:08.856470
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
df = pd.read_csv('data/pub/Pub-Distorted.csv')

# Normalize CaseID → Case if needed (handle naming variations)
df['Case'] = df['Case'].astype(str).str.lower()

# Store original Activity values in `original_activity` column for reference
df['original_activity'] = df['Activity']

# Identify Distorted Activities
df['isdistorted'] = df['Activity'].str.endswith(activity_suffix_pattern)
df['BaseActivity'] = df['Activity'].str.replace(activity_suffix_pattern, '')

# Preprocess Activity Names
df['ProcessedActivity'] = df['Activity'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x.lower()))
df['ProcessedActivity'] = df['ProcessedActivity'].apply(lambda x: re.sub(r'\s+', ' ', x))
df = df[df['ProcessedActivity'].str.len() >= 4]

# Calculate Jaccard N-gram Similarity
def generate_ngrams(text, ngram_size):
    return set([text[i:i+ngram_size] for i in range(len(text) - ngram_size + 1)])

def jaccard_similarity(set1, set2):
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if union else 0.0

def find_similar_pairs(df, similarity_threshold):
    similar_pairs = []
    unique_activities = df['ProcessedActivity'].unique()
    for i in range(len(unique_activities)):
        for j in range(i+1, len(unique_activities)):
            activity1 = unique_activities[i]
            activity2 = unique_activities[j]
            similarity = jaccard_similarity(generate_ngrams(activity1, 3), generate_ngrams(activity2, 3))
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

def build_clusters(df, similar_pairs):
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

clusters = build_clusters(df, similar_pairs)

# Majority Voting Within Clusters
def majority_voting(df, clusters):
    canonical_forms = {}
    for cluster in clusters.values():
        if len(cluster) >= 2:
            counts = {}
            for activity in cluster:
                original_activity = df.loc[df['ProcessedActivity'] == activity, 'original_activity'].iloc[0]
                counts[original_activity] = counts.get(original_activity, 0) + 1
            canonical_form = max(counts, key=counts.get)
            canonical_forms[canonical_form] = canonical_form
    df['canonical_activity'] = df['original_activity']
    for canonical_form, activity in canonical_forms.items():
        df.loc[df['original_activity'] == activity, 'canonical_activity'] = canonical_form
    return df

df = majority_voting(df, clusters)

# Mark Distorted Activities
def mark_distorted(df):
    df['is_distorted'] = df['original_activity'].apply(lambda x: 1 if x in df['canonical_activity'].values else 0)
    return df

df = mark_distorted(df)

# Calculate Detection Metrics
def calculate_detection_metrics(df):
    if 'label' in df.columns:
        y_true = df['label'].apply(lambda x: 1 if 'distorted' in x.lower() else 0)
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

calculate_detection_metrics(df)

# Integrity Check
def integrity_check(df):
    distortion_clusters = len([cluster for cluster in clusters.values() if len(cluster) > 1])
    distorted_activities = df['is_distorted'].sum()
    clean_activities = (df['is_distorted'] == 0).sum()
    print(f"Total distortion clusters detected: {distortion_clusters}")
    print(f"Total activities marked as distorted: {distorted_activities}")
    print(f"Activities to be fixed: {distortion_clusters}")

integrity_check(df)

# Fix Activities
def fix_activities(df):
    df['Activity_fixed'] = df['canonical_activity']
    return df

df = fix_activities(df)

# Save Output
def save_output(df):
    df.to_csv('data/pub/pub_distorted_cleaned_run2.csv', index=False)

save_output(df)

# Summary Statistics
print(f"Run 2: Processed dataset saved to: data/pub/pub_distorted_cleaned_run2.csv")
print(f"Run 2: Final dataset shape: {df.shape}")
print(f"Run 2: Dataset: pub")
print(f"Run 2: Task type: distorted")