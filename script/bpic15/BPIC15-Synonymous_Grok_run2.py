# Generated script for BPIC15-Synonymous - Run 2
# Generated on: 2025-11-18T21:58:13.292645
# Model: grok-4-fast

import pandas as pd
import re
from collections import defaultdict, Counter
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
input_file = 'data/bpic15/BPIC15-Synonymous.csv'
dataset_name = 'bpic15'
output_suffix = '_synonymous_cleaned_run2'
input_directory = 'data/bpic15'

print(f"Run 2: Original dataset shape: {pd.read_csv(input_file).shape}")

# Load and Validate
df = pd.read_csv(input_file)
if 'CaseID' in df.columns and 'Case' not in df.columns:
    df = df.rename(columns={'CaseID': 'Case'})
if 'Activity' not in df.columns:
    raise ValueError("No 'Activity' column found.")
df['original_activity'] = df['Activity']
df['Activity'] = df['Activity'].astype(str).fillna('').str.strip()
df['Activity'] = df['Activity'].replace('nan', '')
if 'Timestamp' in df.columns:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
if 'Case' in df.columns and 'Timestamp' in df.columns:
    df = df.sort_values(['Case', 'Timestamp']).reset_index(drop=True)
print(f"Dataset shape: {df.shape}")
print(df.head())
print(f"Number of unique Activity values: {df['original_activity'].nunique()}")

def normalize_activity(activity):
    activity = str(activity).strip()
    if activity == 'nan' or pd.isna(activity):
        return ''
    activity = re.sub(activity_suffix_pattern, '', activity)
    activity = activity.lower()
    activity = re.sub(r'[^a-z0-9\s]', '', activity)
    activity = ' '.join(activity.split())
    return activity

df['Activity_clean'] = df['original_activity'].apply(normalize_activity)

# TF-IDF Embedding and Similarity
unique_activities = list(df['original_activity'].unique())
if len(unique_activities) < 2:
    print("Warning: Less than 2 unique activities, skipping clustering.")
    df['SynonymGroup'] = -1
    df['canonical_activity'] = df['original_activity']
    df['is_synonymous_event'] = 0
    valid_clusters = {}
else:
    cleans = [normalize_activity(act) for act in unique_activities]
    vectorizer = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=ngram_range,
        lowercase=True,
        min_df=1
    )
    tfidf_matrix = vectorizer.fit_transform(cleans)
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Unique activity count: {len(unique_activities)}")
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Cluster Using Union-Find
    class UnionFind:
        def __init__(self, size):
            self.parent = list(range(size))
            self.rank = [0] * size

        def find(self, p):
            if self.parent[p] != p:
                self.parent[p] = self.find(self.parent[p])
            return self.parent[p]

        def union(self, p, q):
            pp = self.find(p)
            pq = self.find(q)
            if pp == pq:
                return
            if self.rank[pp] < self.rank[pq]:
                self.parent[pp] = pq
            elif self.rank[pp] > self.rank[pq]:
                self.parent[pq] = pp
            else:
                self.parent[pq] = pp
                self.rank[pp] += 1

    uf = UnionFind(len(unique_activities))
    for i in range(len(unique_activities)):
        for j in range(i + 1, len(unique_activities)):
            if similarity_matrix[i, j] >= similarity_threshold:
                uf.union(i, j)

    clusters = defaultdict(list)
    for i in range(len(unique_activities)):
        root = uf.find(i)
        clusters[root].append(i)

    valid_clusters = {k: v for k, v in clusters.items() if len(v) >= min_synonym_group_size}
    print(f"Number of synonym clusters discovered: {len(valid_clusters)}")

    # Select Canonical Form
    canonical_mapping = {}
    activity_to_cluster = {}
    cluster_id = 0
    freqs = df['original_activity'].value_counts()
    for root, member_indices in valid_clusters.items():
        member_originals = [unique_activities[idx] for idx in member_indices]
        canonical_orig = max(member_originals, key=lambda o: freqs.get(o, 0))
        for orig in member_originals:
            canonical_mapping[orig] = canonical_orig
            activity_to_cluster[orig] = cluster_id
        cluster_id += 1

    for orig in unique_activities:
        if orig not in activity_to_cluster:
            activity_to_cluster[orig] = -1
            canonical_mapping[orig] = orig

    df['SynonymGroup'] = df['original_activity'].map(activity_to_cluster)
    df['canonical_activity'] = df['original_activity'].map(canonical_mapping)
    df['is_synonymous_event'] = ((df['SynonymGroup'] != -1) & (df['original_activity'] != df['canonical_activity'])).astype(int)

# Calculate Detection Metrics
print("=== Detection Performance Metrics ===")
if label_column in df.columns:
    y_true = ((df[label_column].notna()) & (df[label_column].astype(str) != '')).astype(int)
    y_pred = df['is_synonymous_event']
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1s = f1_score(y_true, y_pred, zero_division=0)
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1s:.4f}")
else:
    print("Precision: 0.0000")
    print("Recall: 0.0000")
    print("F1-Score: 0.0000")
    print("No ground-truth labels available for evaluation.")

# Integrity Check
print(f"Total synonym clusters found: {len(valid_clusters)}")
print(f"Total events flagged as synonyms: {df['is_synonymous_event'].sum()}")
print(f"Total canonical/clean events: {len(df) - df['is_synonymous_event'].sum()}")

# Fix Activities
df['Activity'] = df['canonical_activity']

# Create Final Fixed Dataset
helpers = ['original_activity', 'Activity_clean', 'canonical_activity', 'SynonymGroup', 'is_synonymous_event']
final_df = df.drop(columns=helpers)

# Save Output and Summary
output_file = f"{input_directory}/{dataset_name}{output_suffix}.csv"
final_df.to_csv(output_file, index=False)

num_replaced = (df['original_activity'] != df['canonical_activity']).sum()
replacement_rate = (num_replaced / len(df)) * 100
unique_before = df['original_activity'].nunique()
unique_after = final_df['Activity'].nunique()
reduction = unique_before - unique_after
reduction_pct = (reduction / unique_before) * 100 if unique_before > 0 else 0
print(f"Total rows: {len(final_df)}")
print(f"Synonym clusters found: {len(valid_clusters)}")
print(f"Synonymous events replaced: {num_replaced}")
print(f"Replacement rate: {replacement_rate:.2f}%")
print(f"Unique activities before → after: {unique_before} → {unique_after}")
print(f"Activity reduction: {reduction} ({reduction_pct:.2f}%)")
print(f"Output file path: {output_file}")
print("Sample transformations:")
changed = df[df['original_activity'] != df['canonical_activity']][['original_activity', 'canonical_activity']].drop_duplicates().head(10)
if len(changed) == 0:
    print("  No transformations applied.")
else:
    for _, row in changed.iterrows():
        print(f"  '{row['original_activity']}' → '{row['canonical_activity']}'")

print(f"Run 2: Processed dataset saved to: {output_file}")
print(f"Run 2: Final dataset shape: {final_df.shape}")
print(f"Run 2: Dataset: {dataset_name}")
print(f"Run 2: Task type: synonymous")