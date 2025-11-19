# Generated script for Credit-Synonymous - Run 1
# Generated on: 2025-11-13T16:41:47.842401
# Model: deepseek-ai/DeepSeek-V3-0324

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration parameters
time_threshold = 2.0
require_same_resource = False
min_matching_events = 2
max_mismatches = 1
activity_suffix_pattern = r'(_signed\d*|_\d+)$'
similarity_threshold = 0.8
case_sensitive = False
use_fuzzy_matching = False
min_synonym_group_size = 2
ngram_range = (1, 3)
label_column = 'label'

# Load and Validate
df = pd.read_csv('data/credit/Credit-Synonymous.csv')
print(f"Run 1: Original dataset shape: {df.shape}")
print(f"Run 1: First few rows:\n{df.head()}")
print(f"Run 1: Unique activities count: {df['Activity'].nunique()}")

# Normalize column names
if 'CaseID' in df.columns and 'Case' not in df.columns:
    df.rename(columns={'CaseID': 'Case'}, inplace=True)

# Ensure Activity column exists
if 'Activity' not in df.columns:
    raise ValueError("Activity column is missing")

# Store original values
df['original_activity'] = df['Activity'].copy()

# Ensure Activity is string-typed
df['Activity'] = df['Activity'].astype(str)

# Parse Timestamp if exists
if 'Timestamp' in df.columns:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

# Sort by Case and Timestamp if both exist
if 'Case' in df.columns and 'Timestamp' in df.columns:
    df.sort_values(['Case', 'Timestamp'], inplace=True)

# Normalize Activity Labels
def normalize_activity(activity):
    if pd.isna(activity):
        return ''
    activity = str(activity)
    if not case_sensitive:
        activity = activity.lower()
    activity = re.sub(r'[^\w\s]', '', activity)
    activity = re.sub(r'\s+', ' ', activity).strip()
    return activity

df['Activity_clean'] = df['original_activity'].apply(normalize_activity)
df['Activity_clean'] = df['Activity_clean'].replace('', 'empty_activity')

# TF-IDF Embedding and Similarity
unique_activities = df['Activity_clean'].unique()
if len(unique_activities) < 2:
    print("Warning: Not enough unique activities for clustering")
    df['is_synonymous_event'] = 0
else:
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=ngram_range, lowercase=False, min_df=1)
    tfidf_matrix = vectorizer.fit_transform(unique_activities)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    print(f"Run 1: TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Run 1: Unique activities count: {len(unique_activities)}")

    # Cluster Using Union-Find
    parent = list(range(len(unique_activities)))

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    def union(u, v):
        u_root = find(u)
        v_root = find(v)
        if u_root != v_root:
            parent[v_root] = u_root

    for i in range(len(unique_activities)):
        for j in range(i + 1, len(unique_activities)):
            if similarity_matrix[i, j] >= similarity_threshold:
                union(i, j)

    clusters = defaultdict(list)
    for i, activity in enumerate(unique_activities):
        clusters[find(i)].append(activity)

    valid_clusters = {k: v for k, v in clusters.items() if len(v) >= min_synonym_group_size}
    print(f"Run 1: Synonym clusters found: {len(valid_clusters)}")

    # Select Canonical Form (Majority/Mode)
    activity_to_cluster = {}
    canonical_mapping = {}

    for cluster_id, activities in valid_clusters.items():
        activity_counts = df[df['Activity_clean'].isin(activities)]['Activity_clean'].value_counts()
        canonical = activity_counts.idxmax()
        for activity in activities:
            canonical_mapping[activity] = canonical

    df['SynonymGroup'] = -1
    df['canonical_activity'] = df['Activity_clean']

    for i, (activity, canonical) in enumerate(canonical_mapping.items()):
        mask = df['Activity_clean'] == activity
        df.loc[mask, 'SynonymGroup'] = i
        df.loc[mask, 'canonical_activity'] = canonical

    df['is_synonymous_event'] = 0
    for activity in canonical_mapping:
        mask = (df['Activity_clean'] == activity) & (df['canonical_activity'] != activity)
        df.loc[mask, 'is_synonymous_event'] = 1

    # Calculate Detection Metrics
    if label_column in df.columns:
        y_true = (~df[label_column].isna() & (df[label_column] != '')).astype(int)
        y_pred = df['is_synonymous_event']
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        print("=== Detection Performance Metrics ===")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
    else:
        print("=== Detection Performance Metrics ===")
        print("Precision: 0.0000")
        print("Recall: 0.0000")
        print("F1-Score: 0.0000")
        print("No ground-truth labels available for evaluation")

    # Integrity Check
    print(f"Run 1: Total synonym clusters found: {len(valid_clusters)}")
    print(f"Run 1: Total events flagged as synonyms: {df['is_synonymous_event'].sum()}")
    print(f"Run 1: Total canonical/clean events: {len(df) - df['is_synonymous_event'].sum()}")

    # Fix Activities
    df['Activity'] = df['canonical_activity']

# Create Final Fixed Dataset
output_columns = [col for col in df.columns if col not in ['original_activity', 'Activity_clean', 'canonical_activity', 'SynonymGroup', 'is_synonymous_event']]
final_df = df[output_columns]

# Save Output and Summary
output_path = 'data/credit/credit_synonymous_cleaned_run1.csv'
final_df.to_csv(output_path, index=False)

# Print summary
print(f"Run 1: Total rows: {len(final_df)}")
print(f"Run 1: Synonym clusters found: {len(valid_clusters) if 'valid_clusters' in locals() else 0}")
print(f"Run 1: Synonymous events replaced: {df['is_synonymous_event'].sum() if 'is_synonymous_event' in df.columns else 0}")
print(f"Run 1: Replacement rate: {df['is_synonymous_event'].sum() / len(df) * 100 if 'is_synonymous_event' in df.columns else 0:.2f}%")
print(f"Run 1: Unique activities before: {df['original_activity'].nunique()}")
print(f"Run 1: Unique activities after: {final_df['Activity'].nunique()}")
print(f"Run 1: Activity reduction: {df['original_activity'].nunique() - final_df['Activity'].nunique()} ({((df['original_activity'].nunique() - final_df['Activity'].nunique()) / df['original_activity'].nunique() * 100):.2f}%)")
print(f"Run 1: Output file path: {output_path}")

# Print sample transformations
if 'valid_clusters' in locals():
    changed = df[df['is_synonymous_event'] == 1][['original_activity', 'canonical_activity']].drop_duplicates()
    print("Run 1: Sample transformations:")
    for _, row in changed.head(10).iterrows():
        print(f"'{row['original_activity']}' → '{row['canonical_activity']}'")

print(f"Run 1: Final dataset shape: {final_df.shape}")
print(f"Run 1: Dataset: credit")
print(f"Run 1: Task type: synonymous")