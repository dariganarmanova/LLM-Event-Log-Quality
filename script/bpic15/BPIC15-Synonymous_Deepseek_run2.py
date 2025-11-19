# Generated script for BPIC15-Synonymous - Run 2
# Generated on: 2025-11-13T12:40:54.580446
# Model: deepseek-ai/DeepSeek-V3-0324

```python
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score
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

# Task-specific parameters
min_synonym_group_size = 2
use_case_scope = False
ngram_range = (1, 3)
save_detection_file = False
label_column = 'label'
output_suffix = '_cleaned_run2'
detection_output_suffix = '_synonym_detection_run2'

# File paths
input_file = 'data/bpic15/BPIC15-Synonymous.csv'
input_directory = 'data/bpic15'
dataset_name = 'bpic15'
output_file = f'{input_directory}/{dataset_name}{output_suffix}.csv'

# Step 1: Load and Validate
df = pd.read_csv(input_file)

# Normalize column names
if 'CaseID' in df.columns and 'Case' not in df.columns:
    df = df.rename(columns={'CaseID': 'Case'})

if 'Activity' not in df.columns:
    raise ValueError("Activity column is missing in the dataset.")

df['original_activity'] = df['Activity']
df['Activity'] = df['Activity'].astype(str).fillna('')

if 'Timestamp' in df.columns:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

if 'Case' in df.columns and 'Timestamp' in df.columns:
    df = df.sort_values(['Case', 'Timestamp'])

print(f"Run 2: Original dataset shape: {df.shape}")
print(f"Run 2: First few rows:\n{df.head()}")
print(f"Run 2: Unique activities before processing: {df['Activity'].nunique()}")

# Step 2: Normalize Activity Labels
def normalize_activity(activity):
    activity = str(activity)
    if not case_sensitive:
        activity = activity.lower()
    activity = re.sub(r'[^a-zA-Z0-9\s]', '', activity)
    activity = re.sub(r'\s+', ' ', activity).strip()
    return activity if activity else 'empty_activity'

df['Activity_clean'] = df['original_activity'].apply(normalize_activity)

# Step 3: TF-IDF Embedding and Similarity
unique_activities = df['Activity_clean'].unique()

if len(unique_activities) < 2:
    df['is_synonymous_event'] = 0
    print("Run 2: Warning: Less than 2 unique activities. Skipping clustering.")
else:
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=ngram_range, lowercase=False, min_df=1)
    tfidf_matrix = vectorizer.fit_transform(unique_activities)
    similarity_matrix = cosine_similarity(tfidf_matrix)

    print(f"Run 2: TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Run 2: Unique activities count: {len(unique_activities)}")

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

    valid_clusters = {k: v for k, v in clusters.items() if len(v) >= min_synonym_group_size}
    activity_to_cluster = {unique_activities[i]: -1 for i in range(len(unique_activities))}
    for cluster_id, members in valid_clusters.items():
        for member_idx in members:
            activity_to_cluster[unique_activities[member_idx]] = cluster_id

    print(f"Run 2: Number of synonym clusters discovered: {len(valid_clusters)}")

    # Step 5: Select Canonical Form (Majority/Mode)
    cluster_to_canonical = {}
    for cluster_id, members in valid_clusters.items():
        member_activities = [unique_activities[i] for i in members]
        activity_counts = df[df['Activity_clean'].isin(member_activities)]['Activity_clean'].value_counts()
        canonical = activity_counts.idxmax()
        cluster_to_canonical[cluster_id] = canonical

    df['SynonymGroup'] = df['Activity_clean'].map(activity_to_cluster)
    df['canonical_activity'] = df['Activity_clean']
    for cluster_id, canonical in cluster_to_canonical.items():
        mask = df['SynonymGroup'] == cluster_id
        df.loc[mask, 'canonical_activity'] = canonical

    df['is_synonymous_event'] = ((df['SynonymGroup'] != -1) & (df['Activity_clean'] != df['canonical_activity'])).astype(int)

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
synonym_clusters = df['SynonymGroup'].nunique() - 1 if 'SynonymGroup' in df.columns else 0
synonym_events = df['is_synonymous_event'].sum() if 'is_synonymous_event' in df.columns else 0
canonical_events = len(df) - synonym_events

print(f"Run 2: Total synonym clusters found: {synonym_clusters}")
print(f"Run 2: Total events flagged as synonyms: {synonym_events}")
print(f"Run 2: Total canonical/clean events: {canonical_events}")

# Step 8: Fix Activities
df['Activity'] = df['canonical_activity']

# Step 10: Create Final Fixed Dataset
columns_to_drop = [col for col in ['original_activity', 'Activity_clean', 'canonical_activity', 'SynonymGroup', 'is_synonymous_event'] if col in df.columns]
final_df = df.drop(columns=columns_to_drop)

# Step 11: Save Output and Summary
final_df.to_csv(output_file, index=False)

unique_activities_before = df['original_activity'].nunique()
unique_activities_after = df['Activity'].nunique()
replacement_rate = (synonym_events / len(df)) * 100 if len(df) > 0 else 0

print(f"Run 2: Total rows: {len(df)}")
print(f"Run 2: Synonym clusters found: {synonym_clusters}")
print(f"Run 2: Synonymous events replaced: {synonym_events}")
print(f"Run 2: Replacement rate: {replacement_rate:.2f}%")
print(f"Run 2: Unique activities before → after: {unique_activities_before} → {unique_activities_after}")
print(f"Run 2: Activity reduction: {unique_activities_before - unique_activities_after} ({((unique_activities_before - unique_activities_after) / unique_activities_before * 100):.2f}%)")
print(f"Run 2: Output file path: {output_file}")

changed_samples = df[df['is_synonymous_event'] == 1][['original_activity', 'canonical_activity']].drop_duplicates().head(10)
