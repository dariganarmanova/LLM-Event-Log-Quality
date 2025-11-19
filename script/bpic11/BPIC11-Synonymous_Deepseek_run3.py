# Generated script for BPIC11-Synonymous - Run 3
# Generated on: 2025-11-13T11:51:06.234351
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
df = pd.read_csv('data/bpic11/BPIC11-Synonymous.csv')
if 'CaseID' in df.columns and 'Case' not in df.columns:
    df.rename(columns={'CaseID': 'Case'}, inplace=True)
if 'Activity' not in df.columns:
    raise ValueError("Activity column is missing")
df['original_activity'] = df['Activity'].copy()
df['Activity'] = df['Activity'].astype(str).fillna('')
if 'Timestamp' in df.columns:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
if 'Case' in df.columns and 'Timestamp' in df.columns:
    df.sort_values(['Case', 'Timestamp'], inplace=True)
print(f"Run 3: Original dataset shape: {df.shape}")
print(f"Run 3: First few rows:\n{df.head()}")
print(f"Run 3: Number of unique activities: {df['Activity'].nunique()}")

# Normalize Activity Labels
def normalize_activity(activity):
    activity = str(activity)
    activity = activity.lower()
    activity = re.sub(r'[^a-z0-9 ]', '', activity)
    activity = re.sub(r'\s+', ' ', activity).strip()
    return activity if activity else 'empty_activity'

df['Activity_clean'] = df['original_activity'].apply(normalize_activity)

# TF-IDF Embedding and Similarity
unique_activities = df['Activity_clean'].unique()
if len(unique_activities) < 2:
    df['is_synonymous_event'] = 0
    print("Run 3: Warning: Less than 2 unique activities, skipping clustering")
else:
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=ngram_range, lowercase=True, min_df=1)
    tfidf_matrix = vectorizer.fit_transform(unique_activities)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    print(f"Run 3: TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Run 3: Unique activities count: {len(unique_activities)}")

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
    valid_clusters = [cluster for cluster in clusters.values() if len(cluster) >= min_synonym_group_size]
    print(f"Run 3: Number of synonym clusters discovered: {len(valid_clusters)}")

    # Select Canonical Form (Majority/Mode)
    activity_to_cluster = {unique_activities[i]: -1 for i in range(len(unique_activities))}
    canonical_mapping = {}
    for cluster_id, cluster in enumerate(valid_clusters):
        activity_counts = {}
        for idx in cluster:
            activity = unique_activities[idx]
            count = len(df[df['Activity_clean'] == activity])
            activity_counts[activity] = count
        canonical = max(activity_counts.items(), key=lambda x: x[1])[0]
        for idx in cluster:
            activity = unique_activities[idx]
            activity_to_cluster[activity] = cluster_id
            canonical_mapping[activity] = canonical
    df['SynonymGroup'] = df['Activity_clean'].map(activity_to_cluster).fillna(-1).astype(int)
    df['canonical_activity'] = df['Activity_clean'].map(canonical_mapping).fillna(df['Activity_clean'])
    df['is_synonymous_event'] = df.apply(lambda row: 1 if row['SynonymGroup'] != -1 and row['Activity_clean'] != row['canonical_activity'] else 0, axis=1)

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
    print(f"Run 3: Total synonym clusters found: {len(valid_clusters)}")
    print(f"Run 3: Total events flagged as synonyms: {df['is_synonymous_event'].sum()}")
    print(f"Run 3: Total canonical/clean events: {len(df) - df['is_synonymous_event'].sum()}")

    # Fix Activities
    df['Activity'] = df['canonical_activity']

# Create Final Fixed Dataset
output_columns = [col for col in df.columns if col not in ['original_activity', 'Activity_clean', 'canonical_activity', 'SynonymGroup', 'is_synonymous_event']]
final_df = df[output_columns].copy()

# Save Output and Summary
output_path = 'data/bpic11/bpic11_synonymous_cleaned_run3.csv'
final_df.to_csv(output_path, index=False)
unique_before = df['original_activity'].nunique()
unique_after = final_df['Activity'].nunique()
synonym_count = df['is_synonymous_event'].sum() if 'is_synonymous_event' in df.columns else 0
print(f"Run 3: Summary for dataset: bpic11")
print(f"Run 3: Task type: synonymous")
print(f"Run 3: Total rows: {len(final_df)}")
print(f"Run 3: Synonym clusters found: {len(valid_clusters) if 'valid_clusters' in locals() else 0}")
print(f"Run 3: Synonymous events replaced: {synonym_count}")
print(f"Run 3: Replacement rate: {synonym_count / len(final_df) * 100:.2f}%")
print(f"Run 3: Unique activities before → after: {unique_before} → {unique_after}")
print(f"Run 3: Activity reduction: {unique_before - unique_after} ({((unique_before - unique_after) / unique_before * 100):.2f}%)")
print(f"Run 3: Output file path: {output_path}")
print(f"Run 3: Final dataset shape: {final_df.shape}")
if 'valid_clusters' in locals():
    sample_changes = df[df['is_synonymous_event'] == 1][['original_activity', 'canonical_activity']].drop_duplicates().head(10)
    for _, row in sample_changes.iterrows():
        print(f"'{row['original_activity']}' → '{row['canonical_activity']}'")