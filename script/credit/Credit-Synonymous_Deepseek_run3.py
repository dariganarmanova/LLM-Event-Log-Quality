# Generated script for Credit-Synonymous - Run 3
# Generated on: 2025-11-13T16:43:43.849697
# Model: deepseek-ai/DeepSeek-V3-0324

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import defaultdict
import re
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
input_file = 'data/credit/Credit-Synonymous.csv'
df = pd.read_csv(input_file)

# Normalize column names
if 'CaseID' in df.columns and 'Case' not in df.columns:
    df.rename(columns={'CaseID': 'Case'}, inplace=True)

# Ensure Activity column exists
if 'Activity' not in df.columns:
    raise ValueError("Activity column is missing in the dataset")

# Store original activity
df['original_activity'] = df['Activity'].astype(str)

# Ensure Activity is string-typed
df['Activity'] = df['Activity'].astype(str)

# Parse Timestamp if exists
if 'Timestamp' in df.columns:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

# Sort by Case and Timestamp if both exist
if 'Case' in df.columns and 'Timestamp' in df.columns:
    df.sort_values(['Case', 'Timestamp'], inplace=True)

print(f"Run 3: Original dataset shape: {df.shape}")
print(f"Run 3: First few rows:\n{df.head()}")
print(f"Run 3: Unique activities before processing: {df['Activity'].nunique()}")

# Normalize Activity Labels
def normalize_activity(activity):
    if pd.isna(activity):
        return ''
    activity = str(activity)
    # Convert to lowercase
    activity = activity.lower()
    # Remove non-alphanumeric except spaces
    activity = re.sub(r'[^a-z0-9\s]', '', activity)
    # Collapse whitespace and trim
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
    # Build TF-IDF vectorizer
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

    # Filter clusters by minimum size
    valid_clusters = {k: v for k, v in clusters.items() if len(v) >= min_synonym_group_size}
    print(f"Run 3: Number of synonym clusters discovered: {len(valid_clusters)}")

    # Map each activity to its cluster id (-1 if not in a valid cluster)
    activity_to_cluster = {unique_activities[i]: -1 for i in range(len(unique_activities))}
    for cluster_id, members in valid_clusters.items():
        for member_idx in members:
            activity_to_cluster[unique_activities[member_idx]] = cluster_id

    # Select Canonical Form (Majority/Mode)
    cluster_to_canonical = {}
    for cluster_id, members in valid_clusters.items():
        member_activities = [unique_activities[i] for i in members]
        # Count occurrences in the original data
        counts = df[df['Activity_clean'].isin(member_activities)]['Activity_clean'].value_counts()
        if not counts.empty:
            canonical = counts.idxmax()
            cluster_to_canonical[cluster_id] = canonical

    # Build canonical_mapping
    canonical_mapping = {}
    for activity in unique_activities:
        cluster_id = activity_to_cluster[activity]
        if cluster_id != -1:
            canonical_mapping[activity] = cluster_to_canonical[cluster_id]
        else:
            canonical_mapping[activity] = activity

    # Assign to DataFrame
    df['SynonymGroup'] = df['Activity_clean'].map(activity_to_cluster)
    df['canonical_activity'] = df['Activity_clean'].map(canonical_mapping)
    df['is_synonymous_event'] = df.apply(
        lambda row: 1 if row['SynonymGroup'] != -1 and row['Activity_clean'] != row['canonical_activity'] else 0,
        axis=1
    )

# Calculate Detection Metrics (BEFORE FIXING)
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
synonym_clusters = len(valid_clusters) if 'valid_clusters' in locals() else 0
synonym_events = df['is_synonymous_event'].sum()
canonical_events = len(df) - synonym_events
print(f"Run 3: Total synonym clusters found: {synonym_clusters}")
print(f"Run 3: Total events flagged as synonyms: {synonym_events}")
print(f"Run 3: Total canonical/clean events: {canonical_events}")

# Fix Activities
df['Activity'] = df['canonical_activity']

# Create Final Fixed Dataset
output_columns = [col for col in df.columns if col not in [
    'original_activity', 'Activity_clean', 'canonical_activity', 'SynonymGroup', 'is_synonymous_event'
]]
final_df = df[output_columns]

# Save Output and Summary
output_file = 'data/credit/credit_synonymous_cleaned_run3.csv'
final_df.to_csv(output_file, index=False)

# Print summary
unique_activities_before = df['original_activity'].nunique()
unique_activities_after = df['Activity'].nunique()
replacement_rate = (synonym_events / len(df)) * 100
activity_reduction = unique_activities_before - unique_activities_after
activity_reduction_pct = (activity_reduction / unique_activities_before) * 100

print(f"Run 3: Total rows: {len(df)}")
print(f"Run 3: Synonym clusters found: {synonym_clusters}")
print(f"Run 3: Synonymous events replaced: {synonym_events}")
print(f"Run 3: Replacement rate: {replacement_rate:.2f}%")
print(f"Run 3: Unique activities before → after: {unique_activities_before} → {unique_activities_after}")
print(f"Run 3: Activity reduction: {activity_reduction} ({activity_reduction_pct:.2f}%)")
print(f"Run 