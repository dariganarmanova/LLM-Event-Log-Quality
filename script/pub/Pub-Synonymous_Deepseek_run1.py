# Generated script for Pub-Synonymous - Run 1
# Generated on: 2025-11-13T17:03:21.978859
# Model: deepseek-ai/DeepSeek-V3-0324

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import defaultdict
import re
import numpy as np

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
df = pd.read_csv('data/pub/Pub-Synonymous.csv')
print(f"Run 1: Original dataset shape: {df.shape}")

if 'CaseID' in df.columns and 'Case' not in df.columns:
    df.rename(columns={'CaseID': 'Case'}, inplace=True)

if 'Activity' not in df.columns:
    raise ValueError("Activity column is missing")

df['original_activity'] = df['Activity'].copy()
df['Activity'] = df['Activity'].astype(str).fillna('')

if 'Timestamp' in df.columns:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

if 'Case' in df.columns and 'Timestamp' in df.columns:
    df.sort_values(by=['Case', 'Timestamp'], inplace=True)

print(f"Number of unique Activity values: {df['Activity'].nunique()}")
print(df.head())

# Normalize Activity Labels
def normalize_activity(activity):
    activity = str(activity)
    if not case_sensitive:
        activity = activity.lower()
    activity = re.sub(r'[^\w\s]', '', activity)
    activity = re.sub(r'\s+', ' ', activity).strip()
    return activity if activity else 'empty_activity'

df['Activity_clean'] = df['original_activity'].apply(normalize_activity)

# TF-IDF Embedding and Similarity
unique_activities = df['Activity_clean'].unique()
if len(unique_activities) < 2:
    df['is_synonymous_event'] = 0
    print("Warning: Less than 2 unique activities, skipping clustering")
else:
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=ngram_range, lowercase=False, min_df=1)
    tfidf_matrix = vectorizer.fit_transform(unique_activities)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Unique activities count: {len(unique_activities)}")

    # Cluster Using Union-Find
    parent = list(range(len(unique_activities)))

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    for i in range(len(unique_activities)):
        for j in range(i + 1, len(unique_activities)):
            if similarity_matrix[i, j] >= similarity_threshold:
                root_i = find(i)
                root_j = find(j)
                if root_i != root_j:
                    parent[root_j] = root_i

    clusters = defaultdict(list)
    for i in range(len(unique_activities)):
        clusters[find(i)].append(i)

    valid_clusters = [cluster for cluster in clusters.values() if len(cluster) >= min_synonym_group_size]
    print(f"Number of synonym clusters discovered: {len(valid_clusters)}")

    # Select Canonical Form (Majority/Mode)
    activity_to_index = {act: idx for idx, act in enumerate(unique_activities)}
    index_to_activity = {idx: act for idx, act in enumerate(unique_activities)}
    activity_to_cluster = {act: -1 for act in unique_activities}
    canonical_mapping = {act: act for act in unique_activities}

    for cluster_id, cluster in enumerate(valid_clusters):
        cluster_activities = [index_to_activity[idx] for idx in cluster]
        activity_counts = df['Activity_clean'].value_counts()
        canonical = max(cluster_activities, key=lambda x: activity_counts.get(x, 0))
        for act in cluster_activities:
            activity_to_cluster[act] = cluster_id
            canonical_mapping[act] = canonical

    df['SynonymGroup'] = df['Activity_clean'].map(activity_to_cluster)
    df['canonical_activity'] = df['Activity_clean'].map(canonical_mapping)
    df['is_synonymous_event'] = df.apply(
        lambda row: 1 if row['SynonymGroup'] != -1 and row['Activity_clean'] != row['canonical_activity'] else 0,
        axis=1
    )

# Calculate Detection Metrics (BEFORE FIXING)
if label_column in df.columns:
    y_true = (df[label_column].notna() & (df[label_column] != '')).astype(int)
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
synonym_clusters = df['SynonymGroup'].nunique() - (1 if -1 in df['SynonymGroup'].values else 0)
synonym_events = df['is_synonymous_event'].sum()
canonical_events = len(df) - synonym_events
print(f"Total synonym clusters found: {synonym_clusters}")
print(f"Total events flagged as synonyms: {synonym_events}")
print(f"Total canonical/clean events: {canonical_events}")

# Fix Activities
df['Activity'] = df['canonical_activity']

# Create Final Fixed Dataset
output_columns = [col for col in df.columns if col not in ['original_activity', 'Activity_clean', 'canonical_activity', 'SynonymGroup', 'is_synonymous_event']]
final_df = df[output_columns].copy()

# Save Output and Summary
output_path = 'data/pub/pub_synonymous_cleaned_run1.csv'
final_df.to_csv(output_path, index=False)

unique_before = df['original_activity'].nunique()
unique_after = df['Activity'].nunique()
reduction = unique_before - unique_after
reduction_pct = (reduction / unique_before) * 100 if unique_before > 0 else 0

print("=== Summary ===")
print(f"Total rows: {len(df)}")
print(f"Synonym clusters found: {synonym_clusters}")
print(f"Synonymous events replaced: {synonym_events}")
print(f"Replacement rate: {(synonym_events / len(df)) * 100:.2f}%")
print(f"Unique activities before → after: {unique_before} → {unique_after}")
print(f"Activity reduction: {reduction} ({reduction_pct:.2f}%)")
print(f"Output file path: {output_path}")

changed = df[df['is_synonymous_event'] == 1][['original_activity', 'canonical_activity']].drop_duplicates()
print("Sample transformations:")
for _, row in changed.head(10).iterrows():
    print(f"'{row['original_activity']}' → '{row['canonical_activity']}'")

print(f"Run 1: Processed dataset saved to: {output_path}")
print(f"Run 1: Final dataset shape: {final_df.shape}")
print(f"Run 1: Dataset: pub")
print(f"Run 1: Task type: synonymous")