# Generated script for Pub-Synonymous - Run 2
# Generated on: 2025-11-13T17:04:26.373017
# Model: deepseek-ai/DeepSeek-V3-0324

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
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
df = pd.read_csv('data/pub/Pub-Synonymous.csv')
if 'CaseID' in df.columns and 'Case' not in df.columns:
    df.rename(columns={'CaseID': 'Case'}, inplace=True)
if 'Activity' not in df.columns:
    raise ValueError("Activity column is missing")
df['original_activity'] = df['Activity'].astype(str)
df['Activity'] = df['Activity'].fillna('').astype(str)
if 'Timestamp' in df.columns:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
if 'Case' in df.columns and 'Timestamp' in df.columns:
    df = df.sort_values(['Case', 'Timestamp'])
print(f"Run 2: Original dataset shape: {df.shape}")
print(f"Run 2: First few rows:\n{df.head()}")
print(f"Run 2: Unique activities count: {df['Activity'].nunique()}")

# Normalize Activity Labels
def normalize_activity(activity):
    activity = str(activity).lower()
    activity = re.sub(r'[^a-z0-9 ]', '', activity)
    activity = re.sub(r'\s+', ' ', activity).strip()
    return activity if activity else 'empty_activity'

df['Activity_clean'] = df['original_activity'].apply(normalize_activity)
df['Activity_clean'] = df['Activity_clean'].replace('', 'empty_activity')

# TF-IDF Embedding and Similarity
unique_activities = df['Activity_clean'].unique()
if len(unique_activities) < 2:
    df['is_synonymous_event'] = 0
    print("Warning: Not enough unique activities for clustering")
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
    clusters = defaultdict(list)
    for i in range(len(unique_activities)):
        clusters[find(i)].append(i)
    valid_clusters = {k: v for k, v in clusters.items() if len(v) >= min_synonym_group_size}
    activity_to_cluster = {}
    for cluster_id, members in valid_clusters.items():
        for member_idx in members:
            activity_to_cluster[unique_activities[member_idx]] = cluster_id
    print(f"Run 2: Synonym clusters discovered: {len(valid_clusters)}")

    # Select Canonical Form (Majority/Mode)
    canonical_mapping = {}
    activity_counts = df['Activity_clean'].value_counts().to_dict()
    for cluster_id, members in valid_clusters.items():
        member_activities = [unique_activities[member_idx] for member_idx in members]
        canonical = max(member_activities, key=lambda x: activity_counts.get(x, 0))
        for member_activity in member_activities:
            canonical_mapping[member_activity] = canonical
    df['SynonymGroup'] = df['Activity_clean'].apply(lambda x: activity_to_cluster.get(x, -1))
    df['canonical_activity'] = df['Activity_clean'].apply(lambda x: canonical_mapping.get(x, x))
    df['is_synonymous_event'] = df.apply(lambda row: 1 if row['SynonymGroup'] != -1 and row['Activity_clean'] != row['canonical_activity'] else 0, axis=1)

    # Calculate Detection Metrics
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
    print(f"Run 2: Total synonym clusters found: {len(valid_clusters)}")
    print(f"Run 2: Total events flagged as synonyms: {df['is_synonymous_event'].sum()}")
    print(f"Run 2: Total canonical/clean events: {len(df) - df['is_synonymous_event'].sum()}")

    # Fix Activities
    df['Activity'] = df['canonical_activity']

# Create Final Fixed Dataset
output_columns = [col for col in df.columns if col not in ['original_activity', 'Activity_clean', 'canonical_activity', 'SynonymGroup', 'is_synonymous_event']]
final_df = df[output_columns]

# Save Output and Summary
final_df.to_csv('data/pub/pub_synonymous_cleaned_run2.csv', index=False)
unique_before = df['original_activity'].nunique()
unique_after = final_df['Activity'].nunique()
replacement_rate = (df['is_synonymous_event'].sum() / len(df)) * 100 if 'is_synonymous_event' in df.columns else 0
print(f"Run 2: Total rows: {len(final_df)}")
print(f"Run 2: Synonym clusters found: {len(valid_clusters) if 'valid_clusters' in locals() else 0}")
print(f"Run 2: Synonymous events replaced: {df['is_synonymous_event'].sum() if 'is_synonymous_event' in df.columns else 0}")
print(f"Run 2: Replacement rate: {replacement_rate:.2f}%")
print(f"Run 2: Unique activities before: {unique_before} → after: {unique_after}")
print(f"Run 2: Activity reduction: {unique_before - unique_after} ({((unique_before - unique_after) / unique_before * 100):.2f}%)")
print(f"Run 2: Output file path: data/pub/pub_synonymous_cleaned_run2.csv")
if 'is_synonymous_event' in df.columns and df['is_synonymous_event'].sum() > 0:
    changes = df[df['is_synonymous_event'] == 1][['original_activity', 'canonical_activity']].drop_duplicates().head(10)
    for _, row in changes.iterrows():
        print(f"'{row['original_activity']}' → '{row['canonical_activity']}'")
print(f"Run 2: Final dataset shape: {final_df.shape}")
print(f"Run 2: Dataset: pub")
print(f"Run 2: Task type: synonymous")