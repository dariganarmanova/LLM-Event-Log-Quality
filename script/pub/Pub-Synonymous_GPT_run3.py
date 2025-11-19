# Generated script for Pub-Synonymous - Run 3
# Generated on: 2025-11-13T17:01:49.642751
# Model: gpt-4o-2024-11-20

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
import re

# Configuration parameters
input_file = 'data/pub/Pub-Synonymous.csv'
output_file = 'data/pub/pub_synonymous_cleaned_run3.csv'
time_threshold = 2.0
require_same_resource = False
min_matching_events = 2
max_mismatches = 1
activity_suffix_pattern = r"(_signed\d*|_\d+)$"
similarity_threshold = 0.8
case_sensitive = False
use_fuzzy_matching = False
ngram_range = (1, 3)
min_synonym_group_size = 2

# Load the data
try:
    df = pd.read_csv(input_file)
    print(f"Run 3: Original dataset shape: {df.shape}")
except Exception as e:
    print(f"Error loading input file: {e}")
    exit()

# Normalize column names
if 'CaseID' in df.columns and 'Case' not in df.columns:
    df.rename(columns={'CaseID': 'Case'}, inplace=True)

if 'Activity' not in df.columns:
    raise ValueError("The 'Activity' column is required but missing in the dataset.")

# Store original activity column
df['original_activity'] = df['Activity']

# Ensure Activity is string-typed and handle missing values
df['Activity'] = df['Activity'].fillna('').astype(str)

# Normalize activities
def normalize_activity(activity):
    activity = activity.lower() if not case_sensitive else activity
    activity = re.sub(r'[^a-zA-Z0-9\s]', '', activity)  # Remove non-alphanumeric except spaces
    activity = re.sub(r'\s+', ' ', activity).strip()  # Collapse multiple spaces
    return activity if activity else 'empty_activity'

df['Activity_clean'] = df['Activity'].apply(normalize_activity)

# Extract unique activities
unique_activities = df['Activity_clean'].unique()
if len(unique_activities) < 2:
    print("Warning: Less than 2 unique activities found. Skipping clustering.")
    df['SynonymGroup'] = -1
    df['canonical_activity'] = df['Activity_clean']
    df['is_synonymous_event'] = 0
else:
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=ngram_range, lowercase=True)
    tfidf_matrix = vectorizer.fit_transform(unique_activities)
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Union-Find for clustering
    parent = list(range(len(unique_activities)))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        root_x = find(x)
        root_y = find(y)
        if root_x != root_y:
            parent[root_y] = root_x

    for i in range(len(unique_activities)):
        for j in range(i + 1, len(unique_activities)):
            if similarity_matrix[i, j] >= similarity_threshold:
                union(i, j)

    # Build clusters
    clusters = defaultdict(list)
    for idx, activity in enumerate(unique_activities):
        root = find(idx)
        clusters[root].append(activity)

    # Filter clusters by size
    valid_clusters = {k: v for k, v in clusters.items() if len(v) >= min_synonym_group_size}

    # Map activities to canonical forms
    activity_to_cluster = {}
    canonical_mapping = {}
    for cluster_id, members in valid_clusters.items():
        member_counts = Counter(df.loc[df['Activity_clean'].isin(members), 'Activity_clean'])
        canonical = member_counts.most_common(1)[0][0]
        for member in members:
            activity_to_cluster[member] = cluster_id
            canonical_mapping[member] = canonical

    # Assign cluster and canonical activity
    df['SynonymGroup'] = df['Activity_clean'].map(activity_to_cluster).fillna(-1).astype(int)
    df['canonical_activity'] = df['Activity_clean'].map(canonical_mapping).fillna(df['Activity_clean'])
    df['is_synonymous_event'] = np.where(
        (df['SynonymGroup'] != -1) & (df['Activity_clean'] != df['canonical_activity']), 1, 0
    )

# Print detection metrics if label column exists
if 'label' in df.columns:
    y_true = df['label'].notnull().astype(int)
    y_pred = df['is_synonymous_event']
    precision = np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_pred == 1) if np.sum(y_pred == 1) > 0 else 0
    recall = np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_true == 1) if np.sum(y_true == 1) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    print("=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
else:
    print("No ground-truth labels available for evaluation.")

# Integrity check
total_clusters = len(valid_clusters)
total_synonyms = df['is_synonymous_event'].sum()
total_canonical = df['canonical_activity'].nunique()
print(f"Total synonym clusters found: {total_clusters}")
print(f"Total events flagged as synonyms: {total_synonyms}")
print(f"Total canonical/clean events: {total_canonical}")

# Fix activities
df['Activity'] = df['canonical_activity']

# Save final dataset
columns_to_drop = ['original_activity', 'Activity_clean', 'canonical_activity', 'SynonymGroup', 'is_synonymous_event']
final_df = df.drop(columns=columns_to_drop, errors='ignore')
final_df.to_csv(output_file, index=False)

# Print summary
print(f"Run 3: Processed dataset saved to: {output_file}")
print(f"Run 3: Final dataset shape: {final_df.shape}")
print(f"Run 3: Unique activities before: {len(unique_activities)} → after: {final_df['Activity'].nunique()}")
print(f"Run 3: Activity reduction: {len(unique_activities) - final_df['Activity'].nunique()} "
      f"({(len(unique_activities) - final_df['Activity'].nunique()) / len(unique_activities) * 100:.2f}%)")