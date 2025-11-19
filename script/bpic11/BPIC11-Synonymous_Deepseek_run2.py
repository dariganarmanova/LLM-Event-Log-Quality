# Generated script for BPIC11-Synonymous - Run 2
# Generated on: 2025-11-13T11:50:26.473665
# Model: deepseek-ai/DeepSeek-V3-0324

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score
from datetime import datetime

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
df = pd.read_csv('data/bpic11/BPIC11-Synonymous.csv')
print(f"Run 2: Original dataset shape: {df.shape}")

if 'CaseID' in df.columns and 'Case' not in df.columns:
    df.rename(columns={'CaseID': 'Case'}, inplace=True)

if 'Activity' not in df.columns:
    raise ValueError("Activity column is missing")

df['original_activity'] = df['Activity'].astype(str)
df['Activity'] = df['Activity'].astype(str).fillna('')

if 'Timestamp' in df.columns:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

if 'Case' in df.columns and 'Timestamp' in df.columns:
    df.sort_values(by=['Case', 'Timestamp'], inplace=True)

print(f"Run 2: Number of unique activities: {df['Activity'].nunique()}")
print("Run 2: First few rows:")
print(df.head())

# Normalize Activity Labels
def normalize_activity(activity):
    activity = str(activity).lower()
    activity = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in activity)
    activity = ' '.join(activity.split())
    return activity

df['Activity_clean'] = df['original_activity'].apply(normalize_activity)
df['Activity_clean'] = df['Activity_clean'].replace('', 'empty_activity')

# TF-IDF Embedding and Similarity
unique_activities = df['Activity_clean'].unique()
if len(unique_activities) < 2:
    df['is_synonymous_event'] = 0
    print("Run 2: Warning: Less than 2 unique activities, skipping clustering")
else:
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=ngram_range, lowercase=True, min_df=1)
    tfidf_matrix = vectorizer.fit_transform(unique_activities)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    print(f"Run 2: TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Run 2: Unique activities count: {len(unique_activities)}")

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

    clusters = {}
    for i in range(len(unique_activities)):
        root = find(i)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(i)

    valid_clusters = {k: v for k, v in clusters.items() if len(v) >= min_synonym_group_size}
    print(f"Run 2: Number of synonym clusters discovered: {len(valid_clusters)}")

    # Select Canonical Form (Majority/Mode)
    activity_to_cluster = {}
    canonical_mapping = {}

    for cluster_id, members in valid_clusters.items():
        member_activities = [unique_activities[i] for i in members]
        activity_counts = df[df['Activity_clean'].isin(member_activities)]['Activity_clean'].value_counts()
        canonical = activity_counts.idxmax()
        for member in member_activities:
            canonical_mapping[member] = canonical

    df['SynonymGroup'] = df['Activity_clean'].apply(lambda x: next((k for k, v in valid_clusters.items() if unique_activities[v[0]] == canonical_mapping.get(x, x)), -1))
    df['canonical_activity'] = df['Activity_clean'].apply(lambda x: canonical_mapping.get(x, x))
    df['is_synonymous_event'] = df.apply(lambda row: 1 if row['Activity_clean'] in canonical_mapping and row['Activity_clean'] != canonical_mapping[row['Activity_clean']] else 0, axis=1)

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
total_clusters = len(valid_clusters) if 'valid_clusters' in locals() else 0
total_synonyms = df['is_synonymous_event'].sum()
total_canonical = len(df) - total_synonyms
print(f"Run 2: Total synonym clusters found: {total_clusters}")
print(f"Run 2: Total events flagged as synonyms: {total_synonyms}")
print(f"Run 2: Total canonical/clean events: {total_canonical}")

# Fix Activities
df['Activity'] = df['canonical_activity']

# Create Final Fixed Dataset
output_columns = [col for col in df.columns if col not in ['original_activity', 'Activity_clean', 'canonical_activity', 'SynonymGroup', 'is_synonymous_event']]
final_df = df[output_columns]

# Save Output and Summary
output_path = 'data/bpic11/bpic11_synonymous_cleaned_run2.csv'
final_df.to_csv(output_path, index=False)

unique_before = df['original_activity'].nunique()
unique_after = df['Activity'].nunique()
replacement_rate = (total_synonyms / len(df)) * 100

print(f"Run 2: Total rows: {len(df)}")
print(f"Run 2: Synonym clusters found: {total_clusters}")
print(f"Run 2: Synonymous events replaced: {total_synonyms}")
print(f"Run 2: Replacement rate: {replacement_rate:.2f}%")
print(f"Run 2: Unique activities before: {unique_before} → after: {unique_after}")
print(f"Run 2: Activity reduction: {unique_before - unique_after} ({((unique_before - unique_after) / unique_before) * 100:.2f}%)")
print(f"Run 2: Output file path: {output_path}")

sample_changes = df[df['is_synonymous_event'] == 1][['original_activity', 'Activity']].drop_duplicates().head(10)
print("Run 2: Sample transformations:")
for _, row in sample_changes.iterrows():
    print(f"'{row['original_activity']}' → '{row['Activity']}'")

print(f"Run 2: Final dataset shape: {final_df.shape}")
print(f"Run 2: Dataset: bpic11")
print(f"Run 2: Task type: synonymous")