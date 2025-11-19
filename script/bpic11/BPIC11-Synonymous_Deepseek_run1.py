# Generated script for BPIC11-Synonymous - Run 1
# Generated on: 2025-11-13T11:49:27.251073
# Model: deepseek-ai/DeepSeek-V3-0324

```python
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
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
use_case_scope = False
save_detection_file = False
label_column = 'label'
output_suffix = '_cleaned_run1'
detection_output_suffix = '_detection'

# Step 1: Load and Validate
input_file = 'data/bpic11/BPIC11-Synonymous.csv'
df = pd.read_csv(input_file)

# Normalize column naming
if 'CaseID' in df.columns and 'Case' not in df.columns:
    df = df.rename(columns={'CaseID': 'Case'})

# Ensure Activity column exists
if 'Activity' not in df.columns:
    raise ValueError("Activity column is missing in the dataset.")

# Store original values
df['original_activity'] = df['Activity'].copy()

# Ensure Activity is string-typed
df['Activity'] = df['Activity'].astype(str)

# Parse Timestamp if exists
if 'Timestamp' in df.columns:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

# Sort by Case and Timestamp if both exist
if 'Case' in df.columns and 'Timestamp' in df.columns:
    df = df.sort_values(['Case', 'Timestamp'])

print(f"Run 1: Original dataset shape: {df.shape}")
print(f"Run 1: First few rows:\n{df.head()}")
print(f"Run 1: Number of unique Activity values: {df['Activity'].nunique()}")

# Step 2: Normalize Activity Labels
def normalize_activity(activity):
    if pd.isna(activity):
        return ''
    activity = str(activity)
    # Convert to lowercase
    if not case_sensitive:
        activity = activity.lower()
    # Remove non-alphanumeric except spaces
    activity = re.sub(r'[^a-zA-Z0-9\s]', '', activity)
    # Collapse whitespace and trim
    activity = re.sub(r'\s+', ' ', activity).strip()
    return activity

df['Activity_clean'] = df['original_activity'].apply(normalize_activity)
df['Activity_clean'] = df['Activity_clean'].replace('', 'empty_activity')

# Step 3: TF-IDF Embedding and Similarity
unique_activities = df['Activity_clean'].unique()
if len(unique_activities) < 2:
    print("Warning: Less than 2 unique activities after normalization. Skipping clustering.")
    df['is_synonymous_event'] = 0
else:
    # Build TF-IDF vectorizer
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=ngram_range, lowercase=False, min_df=1)
    tfidf_matrix = vectorizer.fit_transform(unique_activities)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    print(f"Run 1: TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Run 1: Unique activity count: {len(unique_activities)}")

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
    print(f"Run 1: Number of synonym clusters discovered: {len(valid_clusters)}")

    # Step 5: Select Canonical Form (Majority/Mode)
    activity_to_index = {act: idx for idx, act in enumerate(unique_activities)}
    index_to_activity = {idx: act for idx, act in enumerate(unique_activities)}
    activity_to_cluster = {act: -1 for act in unique_activities}
    canonical_mapping = {}

    for cluster_id, cluster in enumerate(valid_clusters):
        cluster_activities = [index_to_activity[idx] for idx in cluster]
        # Count occurrences in original data
        activity_counts = df['Activity_clean'].value_counts().to_dict()
        cluster_activities_with_counts = [(act, activity_counts.get(act, 0)) for act in cluster_activities]
        # Select canonical as the most frequent
        canonical = max(cluster_activities_with_counts, key=lambda x: x[1])[0]
        for act in cluster_activities:
            activity_to_cluster[act] = cluster_id
            canonical_mapping[act] = canonical

    # Assign to DataFrame
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
synonym_clusters = len(valid_clusters) if 'valid_clusters' in locals() else 0
synonym_events = df['is_synonymous_event'].sum()
canonical_events = len(df) - synonym_events
print(f"Run 1: Total synonym clusters found: {synonym_clusters}")
print(f"Run 1: Total events flagged as synonyms: {synonym_events}")
print(f"Run 1: Total canonical/clean events: {canonical_events}")

# Step 8: Fix Activities
df['Activity'] = df['canonical_activity']

# Step 10: Create Final Fixed Dataset
columns_to_drop = ['original_activity', 'Activity_clean', 'canonical_activity', 'SynonymGroup', 'is_synonymous_event']
final_columns = [col for col in df.columns if col not in columns_to_drop]
final_df = df[final_columns].copy()

# Step 11: Save Output and Summary
output_file = f"data/bpic11/bpic11_synonymous_cleaned_run1.csv"
final_df.to_csv(output_file, index=False)

# Print summary
original_unique = df['original_activity'].nunique()
final_unique = final_df['Activity'].nunique()
reduction = original_unique - final_unique
reduction_pct = (reduction / original_unique) * 100 if original_unique > 0 else 0
replacement_rate = (synonym_events / len(df)) * 100 if len(df) > 0 else 0

print(f"Run 1: Summary")
print(f"Total rows: {len(df)}")
print(f"Synonym clusters found: {synonym_clusters}")
print(f"Synonymous events replaced: {synonym_events}")
print(f"Replacement rate: {replacement_rate:.2f}%")
