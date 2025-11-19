# Generated script for Pub-Synonymous - Run 1
# Generated on: 2025-11-18T18:52:39.494933
# Model: grok-4-fast

import pandas as pd
import re
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_fscore_support

# Configuration parameters
similarity_threshold = 0.8
min_synonym_group_size = 2
ngram_range = (1, 3)
input_file = 'data/pub/Pub-Synonymous.csv'
output_file = 'data/pub/pub_synonymous_cleaned_run1.csv'
dataset_name = 'pub'
run_number = 1
task_type = 'synonymous'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'

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
            return False
        if self.rank[pp] < self.rank[pq]:
            self.parent[pp] = pq
        elif self.rank[pp] > self.rank[pq]:
            self.parent[pq] = pp
        else:
            self.parent[pq] = pp
            self.rank[pp] += 1
        return True

def normalize_activity(activity):
    if pd.isna(activity):
        return ''
    activity = str(activity).lower()
    activity = re.sub(r'[^a-z0-9\s]', ' ', activity)
    activity = re.sub(r'\s+', ' ', activity).strip()
    return activity

# Load and Validate
df = pd.read_csv(input_file)
original_shape = df.shape
print(f"Run {run_number}: Original dataset shape: {original_shape}")
print(f"Dataset shape: {df.shape}")
print(df.head())
if 'CaseID' in df.columns and case_column not in df.columns:
    df = df.rename(columns={'CaseID': case_column})
if activity_column not in df.columns:
    raise ValueError("No 'Activity' column found.")
df[activity_column] = df[activity_column].astype(str).fillna('')
df['original_activity'] = df[activity_column].copy()
num_unique = df['original_activity'].nunique()
print(f"Number of unique Activity values: {num_unique}")
if timestamp_column in df.columns:
    df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce')
if case_column in df.columns and timestamp_column in df.columns:
    df = df.sort_values([case_column, timestamp_column]).reset_index(drop=True)

# Normalize Activity Labels
df['Activity_clean'] = df['original_activity'].apply(normalize_activity)
df['Activity_clean'] = df['Activity_clean'].replace('', 'empty_activity')
unique_cleans = sorted(df['Activity_clean'].unique())
print(f"Number of unique cleaned activities: {len(unique_cleans)}")

# Build clean_to_orig_count
clean_to_orig_count = defaultdict(lambda: defaultdict(int))
for orig, clean in zip(df['original_activity'], df['Activity_clean']):
    clean_to_orig_count[clean][orig] += 1

# TF-IDF Embedding and Similarity
num_synonym_clusters = 0
if len(unique_cleans) >= 2:
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=ngram_range, lowercase=False, min_df=1)
    tfidf_matrix = vectorizer.fit_transform(unique_cleans)
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    sim_matrix = cosine_similarity(tfidf_matrix)
    print(f"Cosine similarity matrix shape: {sim_matrix.shape}")
    uf = UnionFind(len(unique_cleans))
    for i in range(len(unique_cleans)):
        for j in range(i + 1, len(unique_cleans)):
            if sim_matrix[i, j] >= similarity_threshold:
                uf.union(i, j)
    clusters = defaultdict(list)
    for i in range(len(unique_cleans)):
        clusters[uf.find(i)].append(i)
    valid_clusters = {k: v for k, v in clusters.items() if len(v) >= min_synonym_group_size}
    num_synonym_clusters = len(valid_clusters)
    print(f"Number of synonym clusters discovered: {num_synonym_clusters}")
else:
    print("Warning: Fewer than 2 unique activities, skipping clustering.")
    valid_clusters = {}

# Select Canonical Form
clean_to_canonical_orig = {}
for clean in unique_cleans:
    orig_count = clean_to_orig_count[clean]
    if orig_count:
        most_freq_orig = max(orig_count, key=orig_count.get)
        clean_to_canonical_orig[clean] = most_freq_orig
    else:
        clean_to_canonical_orig[clean] = ''
for root, indices in valid_clusters.items():
    cluster_cleans = [unique_cleans[idx] for idx in indices]
    clean_freqs = {c: sum(clean_to_orig_count[c].values()) for c in cluster_cleans}
    max_freq_clean = max(clean_freqs, key=clean_freqs.get)
    cluster_canonical = clean_to_canonical_orig[max_freq_clean]
    for c in cluster_cleans:
        clean_to_canonical_orig[c] = cluster_canonical
df['canonical_activity'] = df['Activity_clean'].map(clean_to_canonical_orig)

# SynonymGroup
clean_to_syn_group = {clean: -1 for clean in unique_cleans}
cluster_id = 0
for root, indices in valid_clusters.items():
    for idx in indices:
        clean = unique_cleans[idx]
        clean_to_syn_group[clean] = cluster_id
    cluster_id += 1
df['SynonymGroup'] = df['Activity_clean'].map(clean_to_syn_group)

# is_synonymous_event
df['is_synonymous_event'] = ((df['SynonymGroup'] != -1) & (df['original_activity'] != df['canonical_activity'])).astype(int)

# Calculate Detection Metrics
print("\n=== Detection Performance Metrics ===")
if label_column in df.columns:
    y_true = (df[label_column].notna() & (df[label_column].astype(str).str.strip() != '')).astype(int)
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
print(f"\nTotal synonym clusters found: {num_synonym_clusters}")
print(f"Total events flagged as synonyms: {df['is_synonymous_event'].sum()}")
total_canonical = len(df) - df['is_synonymous_event'].sum()
print(f"Total canonical/clean events: {total_canonical}")

# Fix Activities
df[activity_column] = df['canonical_activity']
total_replaced = (df['original_activity'] != df['canonical_activity']).sum()

# Create Final Fixed Dataset
drop_cols = ['original_activity', 'Activity_clean', 'canonical_activity', 'SynonymGroup', 'is_synonymous_event']
final_df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# Save Output and Summary
final_df.to_csv(output_file, index=False)
before_unique = len(df['original_activity'].unique())
after_unique = len(final_df[activity_column].unique())
reduction_count = before_unique - after_unique
reduction_pct = (reduction_count / before_unique * 100) if before_unique > 0 else 0
print(f"\nTotal rows: {len(final_df)}")
print(f"Synonym clusters found: {num_synonym_clusters}")
print(f"Synonymous events replaced: {total_replaced}")
print(f"Replacement rate: {total_replaced / len(df) * 100:.2f}%")
print(f"Unique activities before: {before_unique}")
print(f"Unique activities after: {after_unique}")
print(f"Activity reduction: {reduction_count} ({reduction_pct:.2f}%)")
print(f"Output file path: {output_file}")
changes = df[df['original_activity'] != df['canonical_activity']]
if not changes.empty:
    samples = changes[['original_activity', 'canonical_activity']].head(10).values.tolist()
    print("\nSample transformations (up to 10 where changed):")
    for orig, can in samples:
        print(f"'{orig}' → '{can}'")
print(f"\nRun {run_number}: Processed dataset saved to: {output_file}")
print(f"Run {run_number}: Final dataset shape: {final_df.shape}")
print(f"Run {run_number}: Dataset: {dataset_name}")
print(f"Run {run_number}: Task type: {task_type}")