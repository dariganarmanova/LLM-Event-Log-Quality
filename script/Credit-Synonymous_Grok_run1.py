# Generated script for Credit-Synonymous - Run 1
# Generated on: 2025-11-18T21:16:27.190102
# Model: grok-4-fast

import pandas as pd
import re
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_fscore_support

# Configuration parameters
similarity_threshold = 0.8
min_synonym_group_size = 2
ngram_range = (1, 3)
activity_suffix_pattern = r"(_signed\d*|_\d+)$"
label_column = 'label'
run_number = 1
dataset_name = 'credit'
task_type = 'synonymous'
input_file = 'data/credit/Credit-Synonymous.csv'
output_file = 'data/credit/credit_synonymous_cleaned_run1.csv'

# Load the data
df = pd.read_csv(input_file)
print(f"Run {run_number}: Original dataset shape: {df.shape}")

# Normalize column naming
if 'CaseID' in df.columns and 'Case' not in df.columns:
    df = df.rename(columns={'CaseID': 'Case'})

if 'Activity' not in df.columns:
    raise ValueError("The 'Activity' column is missing from the dataset.")

# Store original and prepare
df['original_activity'] = df['Activity'].fillna('').astype(str)

# Handle timestamp if present
if 'Timestamp' in df.columns:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

# Sort if both Case and Timestamp present
if 'Case' in df.columns and 'Timestamp' in df.columns:
    df = df.sort_values(['Case', 'Timestamp']).reset_index(drop=True)

print(df.head())
print(f"Number of unique Activity values: {df['original_activity'].nunique()}")

def normalize_activity(activity: str) -> str:
    if not activity or activity.strip() == '':
        return 'empty_activity'
    activity = re.sub(activity_suffix_pattern, '', activity)
    activity = activity.lower()
    activity = ''.join([c for c in activity if c.isalnum() or c.isspace()])
    activity = ' '.join(activity.split())
    return activity if activity.strip() else 'empty_activity'

df['Activity_clean'] = df['original_activity'].apply(normalize_activity)

# Clustering setup
original_freq = Counter(df['original_activity'])
unique_originals = list(df['original_activity'].unique())
num_unique = len(unique_originals)

original_to_cluster = {}
valid_clusters = {}
canonical_mapping = {o: o for o in unique_originals}

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

if num_unique >= 2:
    normalized_for_sim = [normalize_activity(o) for o in unique_originals]
    vectorizer = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=ngram_range,
        lowercase=True,
        min_df=1
    )
    tfidf_matrix = vectorizer.fit_transform(normalized_for_sim)
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Unique activity count: {num_unique}")
    similarity_matrix = cosine_similarity(tfidf_matrix)
    parent = list(range(num_unique))
    rank = [0] * num_unique
    for i in range(num_unique):
        for j in range(i + 1, num_unique):
            if similarity_matrix[i, j] >= similarity_threshold:
                union(parent, rank, i, j)
    clusters = defaultdict(list)
    for i in range(num_unique):
        root = find(parent, i)
        clusters[root].append(i)
    valid_clusters = {k: v for k, v in clusters.items() if len(v) >= min_synonym_group_size}
    print(f"Number of synonym clusters discovered: {len(valid_clusters)}")
    for cluster_id, member_indices in valid_clusters.items():
        members_originals = [unique_originals[idx] for idx in member_indices]
        for orig in members_originals:
            original_to_cluster[orig] = cluster_id
        freqs = {orig: original_freq[orig] for orig in members_originals}
        if freqs:
            canonical = max(freqs, key=freqs.get)
            for orig in members_originals:
                canonical_mapping[orig] = canonical

# Assign to DataFrame
df['SynonymGroup'] = df['original_activity'].map(original_to_cluster).fillna(-1).astype(int)
df['canonical_activity'] = df['original_activity'].map(canonical_mapping)
df['is_synonymous_event'] = ((df['SynonymGroup'] != -1) & (df['original_activity'] != df['canonical_activity'])).astype(int)

# Calculate Detection Metrics
print("=== Detection Performance Metrics ===")
if label_column in df.columns:
    y_true = (df[label_column].notna() & (df[label_column].astype(str) != '')).astype(int)
    y_pred = df['is_synonymous_event']
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
else:
    print("Precision: 0.0000")
    print("Recall: 0.0000")
    print("F1-Score: 0.0000")
    print("No ground-truth labels available for evaluation")

# Integrity Check
print(f"Total synonym clusters found: {len(valid_clusters)}")
total_flagged = df['is_synonymous_event'].sum()
print(f"Total events flagged as synonyms: {total_flagged}")
print(f"Total canonical/clean events: {len(df) - total_flagged}")

# Fix Activities
df['Activity'] = df['canonical_activity']

# Create Final Fixed Dataset
final_df = df.drop(columns=['original_activity', 'Activity_clean', 'canonical_activity', 'SynonymGroup', 'is_synonymous_event'], errors='ignore')

# Save Output
final_df.to_csv(output_file, index=False)

# Summary
print(f"Total rows: {len(final_df)}")
print(f"Synonym clusters found: {len(valid_clusters)}")
print(f"Synonymous events replaced: {total_flagged}")
replacement_rate = (total_flagged / len(df) * 100) if len(df) > 0 else 0
print(f"Replacement rate: {replacement_rate:.2f}%")
unique_before = df['original_activity'].nunique()
unique_after = final_df['Activity'].nunique()
print(f"Unique activities before: {unique_before}")
print(f"Unique activities after: {unique_after}")
reduction = unique_before - unique_after
reduction_pct = (reduction / unique_before * 100) if unique_before > 0 else 0
print(f"Activity reduction: {reduction} ({reduction_pct:.2f}%)")
print(f"Output file path: {output_file}")

# Sample transformations
changes = df[df['is_synonymous_event'] == 1][['original_activity', 'canonical_activity']].drop_duplicates()
print("Sample transformations:")
if not changes.empty:
    for _, row in changes.head(10).iterrows():
        print(f"'{row['original_activity']}' → '{row['canonical_activity']}'")
else:
    print("No transformations applied.")

print(f"Run {run_number}: Processed dataset saved to: {output_file}")
print(f"Run {run_number}: Final dataset shape: {final_df.shape}")
print(f"Run {run_number}: Dataset: {dataset_name}")
print(f"Run {run_number}: Task type: {task_type}")