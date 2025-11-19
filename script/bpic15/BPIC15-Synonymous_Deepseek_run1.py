# Generated script for BPIC15-Synonymous - Run 1
# Generated on: 2025-11-13T12:39:56.346000
# Model: deepseek-ai/DeepSeek-V3-0324

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import defaultdict

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
save_detection_file = False
label_column = 'label'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'

# Load the data
df = pd.read_csv('data/bpic15/BPIC15-Synonymous.csv')
print(f"Run 1: Original dataset shape: {df.shape}")

# Step 1: Load and Validate
if 'CaseID' in df.columns and 'Case' not in df.columns:
    df.rename(columns={'CaseID': 'Case'}, inplace=True)
if activity_column not in df.columns:
    raise ValueError(f"Required column '{activity_column}' not found in the dataset.")
df['original_activity'] = df[activity_column].copy()
df[activity_column] = df[activity_column].astype(str).fillna('')
if timestamp_column in df.columns:
    df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce')
if case_column in df.columns and timestamp_column in df.columns:
    df.sort_values(by=[case_column, timestamp_column], inplace=True)
print(f"Unique activities before processing: {df[activity_column].nunique()}")
print(df.head())

# Step 2: Normalize Activity Labels
def normalize_activity(activity):
    activity = str(activity)
    activity = activity.lower() if not case_sensitive else activity
    activity = re.sub(r'[^a-zA-Z0-9 ]', '', activity)
    activity = re.sub(r'\s+', ' ', activity).strip()
    return activity if activity else 'empty_activity'

df['Activity_clean'] = df['original_activity'].apply(normalize_activity)

# Step 3: TF-IDF Embedding and Similarity
unique_activities = df['Activity_clean'].unique()
if len(unique_activities) < 2:
    print("Warning: Not enough unique activities for clustering.")
    df['is_synonymous_event'] = 0
else:
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=ngram_range, lowercase=False, min_df=1)
    tfidf_matrix = vectorizer.fit_transform(unique_activities)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Unique activities count: {len(unique_activities)}")

    # Step 4: Cluster Using Union-Find
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
    print(f"Number of synonym clusters discovered: {len(valid_clusters)}")

    # Step 5: Select Canonical Form (Majority/Mode)
    activity_to_index = {act: idx for idx, act in enumerate(unique_activities)}
    index_to_activity = {idx: act for idx, act in enumerate(unique_activities)}
    activity_to_cluster = {index_to_activity[i]: -1 for i in range(len(unique_activities))}
    canonical_mapping = {}

    for cluster_id, cluster in enumerate(valid_clusters):
        activity_counts = {}
        for idx in cluster:
            activity = index_to_activity[idx]
            count = len(df[df['Activity_clean'] == activity])
            activity_counts[activity] = count
        canonical = max(activity_counts.items(), key=lambda x: x[1])[0]
        for idx in cluster:
            activity = index_to_activity[idx]
            activity_to_cluster[activity] = cluster_id
            canonical_mapping[activity] = canonical

    df['SynonymGroup'] = df['Activity_clean'].map(activity_to_cluster).fillna(-1).astype(int)
    df['canonical_activity'] = df['Activity_clean'].apply(lambda x: canonical_mapping.get(x, x))
    df['is_synonymous_event'] = df.apply(
        lambda row: 1 if row['SynonymGroup'] != -1 and row['Activity_clean'] != row['canonical_activity'] else 0,
        axis=1
    )

# Step 6: Calculate Detection Metrics (BEFORE FIXING)
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

# Step 7: Integrity Check
synonym_clusters = df['SynonymGroup'].nunique() - (1 if -1 in df['SynonymGroup'].values else 0)
synonym_events = df['is_synonymous_event'].sum()
canonical_events = len(df) - synonym_events
print(f"Total synonym clusters found: {synonym_clusters}")
print(f"Total events flagged as synonyms: {synonym_events}")
print(f"Total canonical/clean events: {canonical_events}")

# Step 8: Fix Activities
df[activity_column] = df['canonical_activity']

# Step 10: Create Final Fixed Dataset
output_columns = [col for col in df.columns if col not in [
    'original_activity', 'Activity_clean', 'canonical_activity', 'SynonymGroup', 'is_synonymous_event'
]]
final_df = df[output_columns].copy()

# Step 11: Save Output and Summary
output_path = 'data/bpic15/bpic15_synonymous_cleaned_run1.csv'
final_df.to_csv(output_path, index=False)

unique_before = df['original_activity'].nunique()
unique_after = final_df[activity_column].nunique()
replacement_rate = (synonym_events / len(df)) * 100
reduction = unique_before - unique_after
reduction_pct = (reduction / unique_before) * 100

print("\n=== Summary ===")
print(f"Total rows: {len(final_df)}")
print(f"Synonym clusters found: {synonym_clusters}")
print(f"Synonymous events replaced: {synonym_events}")
print(f"Replacement rate: {replacement_rate:.2f}%")
print(f"Unique activities before → after: {unique_before} → {unique_after}")
print(f"Activity reduction: {reduction} ({reduction_pct:.2f}%)")
print(f"Output file path: {output_path}")

changed_samples = df[df['is_synonymous_event'] == 1][['original_activity', 'canonical_activity']].drop_duplicates().head(10)
for _, row in changed_samples.iterrows():
    print(f"'{row['original_activity']}' → '{row['canonical_activity']}'")

print(f"\nRun 1: Processed dataset saved to: {output_path}")
print(f"Run 1: Final dataset shape: {final_df.shape}")
print(f"Run 1: Dataset: bpic15")
print(f"Run 1: Task type: synonymous")