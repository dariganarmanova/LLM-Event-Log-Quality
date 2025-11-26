# Generated script for Credit-Synonymous - Run 3
# Generated on: 2025-11-18T21:18:05.196041
# Model: grok-4-fast

import pandas as pd
import re
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration parameters
time_threshold = 2.0
require_same_resource = False
min_matching_events = 2
max_mismatches = 1
activity_suffix_pattern = r"(_signed\d*|_\d+)$"
similarity_threshold = 0.8
case_sensitive = False
use_fuzzy_matching = False
min_synonym_group_size = 2
ngram_range = (1, 3)
save_detection_file = False
label_column = 'label'
input_file = 'data/credit/Credit-Synonymous.csv'
dataset_name = 'credit'
run_number = 3
output_file = 'data/credit/credit_synonymous_cleaned_run3.csv'

print(f"Run 3: Original dataset shape: {pd.read_csv(input_file).shape}")

# Load and Validate
df = pd.read_csv(input_file)

# Normalize column naming
if 'CaseID' in df.columns and 'Case' not in df.columns:
    df = df.rename(columns={'CaseID': 'Case'})

if 'Activity' not in df.columns:
    raise ValueError("Activity column is missing.")

# Store original values
df['original_activity'] = df['Activity'].copy()

# Ensure Activity is string-typed; fill missing with empty string
df['original_activity'] = df['original_activity'].fillna('')
df['Activity'] = df['Activity'].fillna('').astype(str)

# Print initial info
print(f"Dataset shape: {df.shape}")
print(df.head())
print(f"Number of unique Activity values: {df['original_activity'].nunique()}")

# Handle Timestamp if exists
if 'Timestamp' in df.columns:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

# Sort by Case and Timestamp if both exist
if 'Case' in df.columns and 'Timestamp' in df.columns:
    df = df.sort_values(['Case', 'Timestamp']).reset_index(drop=True)

# Normalize Activity Labels
def normalize_activity(activity):
    if pd.isna(activity):
        return ''
    activity = str(activity).lower()
    activity = re.sub(r'[^a-z0-9 ]', '', activity)
    activity = re.sub(r'\s+', ' ', activity).strip()
    return activity

df['Activity_clean'] = df['original_activity'].apply(normalize_activity)
df['Activity_clean'] = df['Activity_clean'].replace('', 'empty_activity')

# TF-IDF Embedding and Similarity
unique_activities = list(df['Activity_clean'].unique())
if len(unique_activities) < 2:
    print("Warning: Less than 2 unique activities, skipping clustering.")
    df['SynonymGroup'] = -1
    df['canonical_activity'] = df['original_activity']
    df['is_synonymous_event'] = 0
else:
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=ngram_range, lowercase=True, min_df=1)
    tfidf_matrix = vectorizer.fit_transform(unique_activities)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Unique activity count: {len(unique_activities)}")

    # Union-Find for clustering
    parent = list(range(len(unique_activities)))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px = find(x)
        py = find(y)
        if px != py:
            parent[px] = py

    for i in range(len(unique_activities)):
        for j in range(i + 1, len(unique_activities)):
            if similarity_matrix[i, j] >= similarity_threshold:
                union(i, j)

    # Build clusters
    clusters = defaultdict(list)
    for i in range(len(unique_activities)):
        root = find(i)
        clusters[root].append(i)

    # Valid synonym clusters (size >= min_synonym_group_size)
    valid_clusters = {root: indices for root, indices in clusters.items() if len(indices) >= min_synonym_group_size}
    num_synonym_clusters = len(valid_clusters)

    print(f"Number of synonym clusters discovered: {num_synonym_clusters}")

    # Prepare mappings
    original_freq = df['original_activity'].value_counts()
    original_to_clean = df.groupby('original_activity')['Activity_clean'].first().to_dict()

    clean_to_canonical = {}
    cluster_to_id = {}
    clean_to_cluster_id = {}
    cid = 0
    for root, indices in valid_clusters.items():
        cluster_to_id[root] = cid
        cid += 1
        cluster_cleans_set = {unique_activities[idx] for idx in indices}
        possible_originals = [orig for orig, cln in original_to_clean.items() if cln in cluster_cleans_set]
        if possible_originals:
            canonical = max(possible_originals, key=lambda o: original_freq.get(o, 0))
            for cl in cluster_cleans_set:
                clean_to_canonical[cl] = canonical
                clean_to_cluster_id[cl] = cluster_to_id[root]

    # Assign to DataFrame
    df['SynonymGroup'] = -1
    df['canonical_activity'] = df['original_activity']
    mask = df['Activity_clean'].isin(clean_to_canonical)
    df.loc[mask, 'canonical_activity'] = df.loc[mask, 'Activity_clean'].map(clean_to_canonical)
    df['SynonymGroup'] = df['Activity_clean'].map(clean_to_cluster_id).fillna(-1).astype(int)
    df['is_synonymous_event'] = ((df['SynonymGroup'] != -1) & (df['original_activity'] != df['canonical_activity'])).astype(int)

# Calculate Detection Metrics (BEFORE FIXING)
if label_column in df.columns:
    y_true = ((df[label_column].notna()) & (df[label_column].astype(str).str.strip() != '')).astype(int)
    y_pred = df['is_synonymous_event']
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print("=== Detection Performance Metrics ===")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
else:
    print("No ground-truth labels available for evaluation")
    print("=== Detection Performance Metrics ===")
    print("Precision: 0.0000")
    print("Recall: 0.0000")
    print("F1-Score: 0.0000")

# Integrity Check
print(f"Total synonym clusters found: {num_synonym_clusters if 'num_synonym_clusters' in locals() else 0}")
print(f"Total events flagged as synonyms: {df['is_synonymous_event'].sum()}")
print(f"Total canonical/clean events: {len(df) - df['is_synonymous_event'].sum()}")

# Fix Activities
df['Activity'] = df['canonical_activity']

# Create Final Fixed Dataset
helper_columns = ['original_activity', 'Activity_clean', 'canonical_activity', 'SynonymGroup', 'is_synonymous_event']
df_final = df.drop(columns=[col for col in helper_columns if col in df.columns])

# Save Output and Summary
df_final.to_csv(output_file, index=False)

total_rows = len(df)
synonym_events_replaced = df['is_synonymous_event'].sum()
replacement_rate = (synonym_events_replaced / total_rows * 100) if total_rows > 0 else 0
unique_before = df['original_activity'].nunique()
unique_after = df_final['Activity'].nunique()
reduction = unique_before - unique_after
reduction_pct = (reduction / unique_before * 100) if unique_before > 0 else 0
num_synonym_clusters_print = num_synonym_clusters if 'num_synonym_clusters' in locals() else 0

print(f"Total rows: {total_rows}")
print(f"Synonym clusters found: {num_synonym_clusters_print}")
print(f"Synonymous events replaced: {synonym_events_replaced}")
print(f"Replacement rate: {replacement_rate:.2f}%")
print(f"Unique activities before: {unique_before}")
print(f"after: {unique_after}")
print(f"Activity reduction: {reduction} ({reduction_pct:.2f}%)")
print(f"Output file path: {output_file}")

# Sample transformations
changed = df[df['original_activity'] != df['Activity']][['original_activity', 'Activity']].drop_duplicates().head(10)
if len(changed) > 0:
    print("Sample transformations:")
    for _, row in changed.iterrows():
        print(f"  '{row['original_activity']}' → '{row['Activity']}'")
else:
    print("No transformations applied.")

print(f"Run 3: Processed dataset saved to: data/credit/credit_synonymous_cleaned_run3.csv")
print(f"Run 3: Final dataset shape: {df_final.shape}")
print(f"Run 3: Dataset: credit")
print(f"Run 3: Task type: synonymous")