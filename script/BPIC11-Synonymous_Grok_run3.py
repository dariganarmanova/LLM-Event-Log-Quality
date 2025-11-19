# Generated script for BPIC11-Synonymous - Run 3
# Generated on: 2025-11-19T13:08:59.334260
# Model: grok-4-fast

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_fscore_support
from collections import defaultdict

# Configuration
run_number = 3
dataset_name = 'bpic11'
task_type = 'synonymous'
input_file = 'data/bpic11/BPIC11-Synonymous.csv'
output_path = 'data/bpic11/bpic11_synonymous_cleaned_run3.csv'
similarity_threshold = 0.8
min_synonym_group_size = 2
ngram_range = (1, 3)
label_column = 'label'

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

# Normalize function
def normalize_activity(activity):
    if pd.isna(activity):
        return 'empty_activity'
    s = str(activity).lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    if not s:
        return 'empty_activity'
    return s

# Load and validate
df = pd.read_csv(input_file)
print(f"Run {run_number}: Original dataset shape: {df.shape}")

if 'CaseID' in df.columns and 'Case' not in df.columns:
    df = df.rename(columns={'CaseID': 'Case'})

if 'Activity' not in df.columns:
    raise ValueError("Activity column is missing.")

df['original_activity'] = df['Activity'].copy()

if 'Timestamp' in df.columns:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

if 'Case' in df.columns and 'Timestamp' in df.columns:
    df = df.sort_values(['Case', 'Timestamp']).reset_index(drop=True)

print(f"Dataset shape: {df.shape}")
print(df.head())
print(f"Number of unique Activity values: {df['original_activity'].nunique()}")

# Normalize
df['Activity_clean'] = df['original_activity'].apply(normalize_activity)

# TF-IDF and clustering
unique_activities = sorted(df['Activity_clean'].unique())
num_unique = len(unique_activities)

if num_unique < 2:
    print("Warning: Fewer than 2 unique activities, skipping clustering.")
    df['SynonymGroup'] = -1
    df['canonical_activity'] = df['original_activity']
    df['is_synonymous_event'] = 0
    num_clusters = 0
else:
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=ngram_range, lowercase=True, min_df=1)
    tfidf_matrix = vectorizer.fit_transform(unique_activities)
    sim_matrix = cosine_similarity(tfidf_matrix)
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}, unique activities: {num_unique}")

    uf = UnionFind(num_unique)
    for i in range(num_unique):
        for j in range(i + 1, num_unique):
            if sim_matrix[i, j] >= similarity_threshold:
                uf.union(i, j)

    parent_to_members = defaultdict(list)
    for i in range(num_unique):
        root = uf.find(i)
        parent_to_members[root].append(unique_activities[i])

    valid_clusters = [members for members in parent_to_members.values() if len(members) >= min_synonym_group_size]
    num_clusters = len(valid_clusters)
    print(f"Number of synonym clusters discovered: {num_clusters}")

    activity_to_cluster = {}
    canonical_for_cluster = {}
    cluster_id = 0
    for members in valid_clusters:
        for act in members:
            activity_to_cluster[act] = cluster_id
        # Compute frequencies for originals in this cluster
        mask = df['Activity_clean'].isin(members)
        cid_freq = defaultdict(int)
        if mask.sum() > 0:
            for orig, count in df.loc[mask, 'original_activity'].value_counts().items():
                cid_freq[orig] += count
            if cid_freq:
                canonical = max(cid_freq, key=cid_freq.get)
                canonical_for_cluster[cluster_id] = canonical
        cluster_id += 1

    # Unclustered
    all_clustered_cleans = set(activity_to_cluster.keys())
    for act in unique_activities:
        if act not in all_clustered_cleans:
            activity_to_cluster[act] = -1

    # Assign to df
    df['SynonymGroup'] = df['Activity_clean'].map(activity_to_cluster).fillna(-1).astype(int)

    def get_canonical(clean, orig):
        cid = activity_to_cluster.get(clean, -1)
        if cid != -1 and cid in canonical_for_cluster:
            return canonical_for_cluster[cid]
        else:
            return orig

    df['canonical_activity'] = df.apply(lambda row: get_canonical(row['Activity_clean'], row['original_activity']), axis=1)
    df['is_synonymous_event'] = ((df['canonical_activity'] != df['original_activity']) & (df['SynonymGroup'] != -1)).astype(int)

# Metrics
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

# Integrity check
print(f"Total synonym clusters found: {num_clusters}")
print(f"Total events flagged as synonyms: {df['is_synonymous_event'].sum()}")
print(f"Total canonical/clean events: {len(df) - df['is_synonymous_event'].sum()}")

# Fix activities
df['Activity'] = df['canonical_activity']
df['Activity'] = df['Activity'].fillna('')

# Final dataset
helpers = ['original_activity', 'Activity_clean', 'canonical_activity', 'SynonymGroup', 'is_synonymous_event']
final_df = df.drop(columns=[col for col in helpers if col in df.columns])

# Save
final_df.to_csv(output_path, index=False)

# Summary
syn_replaced = df['is_synonymous_event'].sum()
replacement_rate = (syn_replaced / len(df) * 100) if len(df) > 0 else 0
unique_before = df['original_activity'].nunique()
unique_after = final_df['Activity'].nunique()
reduction = unique_before - unique_after
reduction_pct = (reduction / unique_before * 100) if unique_before > 0 else 0

print(f"Total rows: {len(final_df)}")
print(f"Synonym clusters found: {num_clusters}")
print(f"Synonymous events replaced: {syn_replaced}")
print(f"Replacement rate: {replacement_rate:.2f}%")
print(f"Unique activities before: {unique_before}")
print(f"Unique activities after: {unique_after}")
print(f"Activity reduction: {reduction} ({reduction_pct:.2f}%)")
print(f"Output file path: {output_path}")

# Sample transformations
print("Sample transformations:")
changed = df[df['is_synonymous_event'] == 1][['original_activity', 'canonical_activity']].drop_duplicates().head(10)
if len(changed) > 0:
    for _, row in changed.iterrows():
        print(f"'{row['original_activity']}' → '{row['canonical_activity']}'")
else:
    print("No transformations applied.")

print(f"Run {run_number}: Processed dataset saved to: {output_path}")
print(f"Run {run_number}: Final dataset shape: {final_df.shape}")
print(f"Run {run_number}: Dataset: {dataset_name}")
print(f"Run {run_number}: Task type: {task_type}")