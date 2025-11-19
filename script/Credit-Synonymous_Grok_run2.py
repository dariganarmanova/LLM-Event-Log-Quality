# Generated script for Credit-Synonymous - Run 2
# Generated on: 2025-11-18T21:17:08.727455
# Model: grok-4-fast

import pandas as pd
import re
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_fscore_support

# Algorithm Configuration Parameters
time_threshold = 2.0
require_same_resource = False
min_matching_events = 2
max_mismatches = 1
activity_suffix_pattern = r"(_signed\d*|_\d+)$"
similarity_threshold = 0.8
case_sensitive = False
use_fuzzy_matching = False

# Input Configuration
input_file = 'data/credit/Credit-Synonymous.csv'
dataset_name = 'credit'
output_file = 'data/credit/credit_synonymous_cleaned_run2.csv'
label_column = 'label'
min_synonym_group_size = 2
ngram_range = (1, 3)

# Load and Validate
df = pd.read_csv(input_file)
print(f"Run 2: Original dataset shape: {df.shape}")

if 'CaseID' in df.columns and 'Case' not in df.columns:
    df = df.rename(columns={'CaseID': 'Case'})

if 'Activity' not in df.columns:
    raise ValueError("Missing 'Activity' column")

if 'Timestamp' in df.columns:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

if 'Case' in df.columns and 'Timestamp' in df.columns:
    df = df.sort_values(['Case', 'Timestamp']).reset_index(drop=True)

df['original_activity'] = df['Activity'].fillna('').astype(str)

def normalize_activity(activity):
    activity = activity.lower()
    activity = re.sub(r'[^a-z0-9\s]', ' ', activity)
    activity = re.sub(r'\s+', ' ', activity).strip()
    return activity

df['Activity_clean'] = df['original_activity'].apply(normalize_activity)
df.loc[df['Activity_clean'] == '', 'Activity_clean'] = 'empty_activity'

print(df.head())
print(f"Number of unique Activity values: {df['original_activity'].nunique()}")

# TF-IDF Embedding and Clustering Preparation
unique_activities_set = set(df['Activity_clean'].unique())
n = len(unique_activities_set)
if n < 2:
    print("Warning: Less than 2 unique activities, skipping clustering.")
    df['SynonymGroup'] = -1
    df['canonical_activity'] = df['original_activity']
    df['is_synonymous_event'] = 0
    num_clusters = 0
else:
    unique_activities = sorted(unique_activities_set)
    n = len(unique_activities)
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=ngram_range, lowercase=True, min_df=1)
    tfidf_matrix = vectorizer.fit_transform(unique_activities)
    sim_matrix = cosine_similarity(tfidf_matrix)
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}, unique activity count: {n}")

    # Union-Find
    def find(parent, i):
        if parent[i] != i:
            parent[i] = find(parent, parent[i])
        return parent[i]

    def union(parent, rank, x, y):
        px = find(parent, x)
        py = find(parent, y)
        if px != py:
            if rank[px] > rank[py]:
                parent[py] = px
            elif rank[px] < rank[py]:
                parent[px] = py
            else:
                parent[py] = px
                rank[px] += 1

    parent = list(range(n))
    rank = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= similarity_threshold:
                union(parent, rank, i, j)

    clusters = defaultdict(list)
    for i in range(n):
        root = find(parent, i)
        clusters[root].append(i)

    valid_clusters = {k: v for k, v in clusters.items() if len(v) >= min_synonym_group_size}
    print(f"Number of synonym clusters discovered: {len(valid_clusters)}")
    num_clusters = len(valid_clusters)

    # Map activities to cluster IDs (remap to 0,1,... for valid only)
    activity_to_cluster = {}
    cluster_member_map = {}
    cluster_id_counter = 0
    clustered_cleans = set()
    for root, members in valid_clusters.items():
        for i in members:
            clean = unique_activities[i]
            activity_to_cluster[clean] = cluster_id_counter
            clustered_cleans.add(clean)
        cluster_member_map[cluster_id_counter] = members
        cluster_id_counter += 1

    for clean in unique_activities:
        if clean not in activity_to_cluster:
            activity_to_cluster[clean] = -1

    df['SynonymGroup'] = df['Activity_clean'].map(activity_to_cluster).astype(int)

    # Select Canonical Forms
    cluster_to_canonical = {}
    for c_id, members in cluster_member_map.items():
        cluster_cleans = [unique_activities[i] for i in members]
        mask = df['Activity_clean'].isin(cluster_cleans)
        if mask.any():
            orig_counts = df.loc[mask, 'original_activity'].value_counts()
            if len(orig_counts) > 0:
                canonical = orig_counts.index[0]
                cluster_to_canonical[c_id] = canonical

    df['canonical_activity'] = df['original_activity']
    clustered_mask = df['SynonymGroup'] != -1
    if len(cluster_to_canonical) > 0:
        df.loc[clustered_mask, 'canonical_activity'] = df.loc[clustered_mask, 'SynonymGroup'].map(cluster_to_canonical)

    df['is_synonymous_event'] = (clustered_mask & (df['original_activity'] != df['canonical_activity'])).astype(int)

# Calculate Detection Metrics (BEFORE FIXING)
print("=== Detection Performance Metrics ===")
if label_column in df.columns:
    y_true = (df[label_column].notna() & (df[label_column].astype(str) != '')).astype(int)
    y_pred = df['is_synonymous_event']
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
else:
    print("Precision: 0.0000")
    print("Recall: 0.0000")
    print("F1-Score: 0.0000")
    print("No ground-truth labels available for evaluation")

# Integrity Check
total_flagged = df['is_synonymous_event'].sum()
total_canonical = len(df) - total_flagged
print(f"Total synonym clusters found: {num_clusters}")
print(f"Total events flagged as synonyms: {total_flagged}")
print(f"Total canonical/clean events: {total_canonical}")

# Fix Activities
df['Activity'] = df['canonical_activity']

# Create Final Fixed Dataset
drop_cols = ['original_activity', 'Activity_clean', 'canonical_activity', 'SynonymGroup', 'is_synonymous_event']
final_df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

# Save Output and Summary
final_df.to_csv(output_file, index=False)

total_rows = len(final_df)
syn_replaced = int(total_flagged)
rate = (syn_replaced / total_rows * 100) if total_rows > 0 else 0
unique_before = df['original_activity'].nunique()
unique_after = final_df['Activity'].nunique()
reduction = unique_before - unique_after
red_pct = (reduction / unique_before * 100) if unique_before > 0 else 0

print(f"Total rows: {total_rows}")
print(f"Synonym clusters found: {num_clusters}")
print(f"Synonymous events replaced: {syn_replaced}")
print(f"Replacement rate: {rate:.2f}%")
print(f"Unique activities before → after: {unique_before} → {unique_after}")
print(f"Activity reduction: {reduction} ({red_pct:.2f}%)")
print(f"Output file path: {output_file}")

# Sample transformations
changed_pairs = df[df['is_synonymous_event'] == 1][['original_activity', 'canonical_activity']].drop_duplicates().head(10)
if len(changed_pairs) > 0:
    print("Sample transformations:")
    for _, row in changed_pairs.iterrows():
        print(f"  '{row['original_activity']}' → '{row['canonical_activity']}'")
else:
    print("No transformations applied.")

print(f"Run 2: Processed dataset saved to: {output_file}")
print(f"Run 2: Final dataset shape: {final_df.shape}")
print(f"Run 2: Dataset: {dataset_name}")
print(f"Run 2: Task type: synonymous")