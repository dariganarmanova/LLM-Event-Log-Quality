# Generated script for BPIC15-Synonymous - Run 1
# Generated on: 2025-11-18T21:57:05.085087
# Model: grok-4-fast

import pandas as pd
import re
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

# Hardcoded paths and settings
input_file = 'data/bpic15/BPIC15-Synonymous.csv'
dataset_name = 'bpic15'
task_type = 'synonymous'
run_number = 1
output_file = 'data/bpic15/bpic15_synonymous_cleaned_run1.csv'
label_column = 'label'
min_synonym_group_size = min_matching_events
ngram_range = (1, 3)
use_case_scope = False
save_detection_file = False

# Union-Find class
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

# Load and Validate
df = pd.read_csv(input_file)
print(f"Run {run_number}: Original dataset shape: {df.shape}")

if 'CaseID' in df.columns and 'Case' not in df.columns:
    df.rename(columns={'CaseID': 'Case'}, inplace=True)

if 'Activity' not in df.columns:
    raise ValueError("Activity column is missing.")

df['original_activity'] = df['Activity']
df['Activity'] = df['Activity'].astype(str).fillna('')

if 'Timestamp' in df.columns:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

if 'Case' in df.columns and 'Timestamp' in df.columns:
    df = df.sort_values(['Case', 'Timestamp']).reset_index(drop=True)

print(f"Dataset shape: {df.shape}")
print(df.head())
print(f"Number of unique Activity values: {df['original_activity'].nunique()}")

# Normalize Activity Labels
def normalize_activity(activity):
    if pd.isna(activity):
        return ''
    activity = str(activity).lower()
    activity = re.sub(r'[^a-z0-9\s]', ' ', activity)
    activity = re.sub(r'\s+', ' ', activity).strip()
    return activity

df['Activity_clean'] = df['original_activity'].apply(normalize_activity)
empty_mask = (df['Activity_clean'] == '')
df.loc[empty_mask, 'Activity_clean'] = 'empty_activity'

# TF-IDF Embedding and Similarity
unique_cleans = sorted(df['Activity_clean'].unique())
if len(unique_cleans) < 2:
    print("Warning: Fewer than 2 unique activities, skipping synonym detection.")
    df['SynonymGroup'] = -1
    df['is_synonymous_event'] = 0
    df['canonical_activity'] = df['original_activity']
    num_clusters = 0
else:
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=ngram_range, lowercase=True, min_df=1)
    tfidf_matrix = vectorizer.fit_transform(unique_cleans)
    sim_matrix = cosine_similarity(tfidf_matrix)
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}, unique activity count: {len(unique_cleans)}")

    clean_to_index = {clean: idx for idx, clean in enumerate(unique_cleans)}
    uf = UnionFind(len(unique_cleans))
    for i in range(len(unique_cleans)):
        for j in range(i + 1, len(unique_cleans)):
            if sim_matrix[i, j] >= similarity_threshold:
                uf.union(i, j)

    clusters = {}
    for i in range(len(unique_cleans)):
        root = uf.find(i)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(unique_cleans[i])

    valid_clusters = {k: v for k, v in clusters.items() if len(v) >= min_synonym_group_size}
    num_clusters = len(valid_clusters)
    print(f"Number of synonym clusters discovered: {num_clusters}")

    activity_to_cluster = {}
    for i in range(len(unique_cleans)):
        root = uf.find(i)
        if len(clusters[root]) >= min_synonym_group_size:
            activity_to_cluster[unique_cleans[i]] = root
        else:
            activity_to_cluster[unique_cleans[i]] = -1

    # Select Canonical Form
    canonical_for_cluster = {}
    clean_value_counts = df['Activity_clean'].value_counts()
    for cluster_id, cluster_cleans in valid_clusters.items():
        clean_freqs = {clean: clean_value_counts.get(clean, 0) for clean in cluster_cleans}
        canonical_clean = max(clean_freqs, key=clean_freqs.get)
        originals_for_canonical = df[df['Activity_clean'] == canonical_clean]['original_activity'].value_counts()
        if len(originals_for_canonical) > 0:
            canonical_form = originals_for_canonical.index[0]
        else:
            canonical_form = canonical_clean  # fallback
        canonical_for_cluster[cluster_id] = canonical_form

    df['SynonymGroup'] = df['Activity_clean'].map(activity_to_cluster).fillna(-1)
    df['is_synonymous_event'] = 0
    df['canonical_activity'] = df['original_activity']
    for cluster_id, cluster_cleans in valid_clusters.items():
        canonical_form = canonical_for_cluster[cluster_id]
        mask = df['Activity_clean'].isin(cluster_cleans)
        df.loc[mask, 'canonical_activity'] = canonical_form
        df.loc[mask & (df['original_activity'] != canonical_form), 'is_synonymous_event'] = 1

# Calculate Detection Metrics (BEFORE FIXING)
print("=== Detection Performance Metrics ===")
if label_column in df.columns:
    df['y_true'] = ((df[label_column].notna()) & (df[label_column].astype(str).str.strip() != '')).astype(int)
    y_true = df['y_true']
    y_pred = df['is_synonymous_event']
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    df.drop('y_true', axis=1, inplace=True)
else:
    print("Precision: 0.0000")
    print("Recall: 0.0000")
    print("F1-Score: 0.0000")
    print("No ground-truth labels available for evaluation")

# Integrity Check
print(f"Total synonym clusters found: {num_clusters}")
print(f"Total events flagged as synonyms: {df['is_synonymous_event'].sum()}")
print(f"Total canonical/clean events: {len(df) - df['is_synonymous_event'].sum()}")

# Fix Activities
df['Activity'] = df['canonical_activity']

# Create Final Fixed Dataset
helper_columns = ['original_activity', 'Activity_clean', 'canonical_activity', 'SynonymGroup', 'is_synonymous_event']
final_df = df.drop(columns=[col for col in helper_columns if col in df.columns], errors='ignore')

# Save Output and Summary
final_df.to_csv(output_file, index=False)

total_rows = len(final_df)
synonymous_replaced = df['is_synonymous_event'].sum()
replacement_rate = (synonymous_replaced / total_rows * 100) if total_rows > 0 else 0
unique_before = df['original_activity'].nunique()
unique_after = final_df['Activity'].nunique()
reduction = unique_before - unique_after
reduction_pct = (reduction / unique_before * 100) if unique_before > 0 else 0

print(f"Total rows: {total_rows}")
print(f"Synonym clusters found: {num_clusters}")
print(f"Synonymous events replaced: {synonymous_replaced}")
print(f"Replacement rate: {replacement_rate:.2f}%")
print(f"Unique activities before → after: {unique_before} → {unique_after}")
print(f"Activity reduction: {reduction} ({reduction_pct:.2f}%)")
print(f"Output file path: {output_file}")

# Sample transformations
changed = df[df['is_synonymous_event'] == 1][['original_activity', 'canonical_activity']].drop_duplicates()
print("Sample transformations (up to 10 where changed):")
for _, row in changed.head(10).iterrows():
    print(f"'{row['original_activity']}' → '{row['canonical_activity']}'")

print(f"Run {run_number}: Processed dataset saved to: {output_file}")
print(f"Run {run_number}: Final dataset shape: {final_df.shape}")
print(f"Run {run_number}: Dataset: {dataset_name}")
print(f"Run {run_number}: Task type: {task_type}")