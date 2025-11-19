# Generated script for BPIC11-Synonymous - Run 2
# Generated on: 2025-11-19T13:08:12.852690
# Model: grok-4-fast

import pandas as pd
import re
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_fscore_support

class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, p):
        if self.parent[p] != p:
            self.parent[p] = self.find(self.parent[p])
        return self.parent[p]

    def union(self, p, q):
        rootP = self.find(p)
        rootQ = self.find(q)
        if rootP == rootQ:
            return
        if self.rank[rootP] < self.rank[rootQ]:
            self.parent[rootP] = rootQ
        elif self.rank[rootP] > self.rank[rootQ]:
            self.parent[rootQ] = rootP
        else:
            self.parent[rootQ] = rootP
            self.rank[rootP] += 1

# Configuration
input_file = 'data/bpic11/BPIC11-Synonymous.csv'
output_file = 'data/bpic11/bpic11_synonymous_cleaned_run2.csv'
run_number = 2
dataset_name = 'bpic11'
similarity_threshold = 0.8
min_synonym_group_size = 2
ngram_range = (1, 3)
label_column = 'label'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'

print(f"Run {run_number}: Original dataset shape before loading: N/A")

# #1. Load and Validate
df = pd.read_csv(input_file)

if 'CaseID' in df.columns and case_column not in df.columns:
    df = df.rename(columns={'CaseID': case_column})

if activity_column not in df.columns:
    raise ValueError("No 'Activity' column found.")

df[activity_column] = df[activity_column].astype(str).fillna('')
df['original_activity'] = df[activity_column].copy()

if timestamp_column in df.columns:
    df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce')

if case_column in df.columns and timestamp_column in df.columns:
    df = df.sort_values([case_column, timestamp_column]).reset_index(drop=True)

print(f"Dataset shape: {df.shape}")
print(df.head())
print(f"Number of unique Activity values: {df['original_activity'].nunique()}")

# #2. Normalize Activity Labels
def normalize_activity(activity):
    if pd.isna(activity) or activity == '':
        return ''
    lower = str(activity).lower()
    cleaned = re.sub(r'[^\w\s]', '', lower)
    normalized = ' '.join(cleaned.split())
    return normalized

df['Activity_clean'] = df['original_activity'].apply(normalize_activity)

# #3. TF-IDF Embedding and Similarity
unique_activities = df['Activity_clean'].unique()
print(f"Run {run_number}: Unique activities count: {len(unique_activities)}")

if len(unique_activities) < 2:
    df['SynonymGroup'] = -1
    df['is_synonymous_event'] = 0
    df['canonical_activity'] = df['original_activity']
    num_clusters = 0
    print("Warning: Fewer than 2 unique activities; skipping clustering.")
else:
    vectorizer = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=ngram_range,
        lowercase=True,
        min_df=1
    )
    tfidf_matrix = vectorizer.fit_transform(unique_activities)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Unique activity count: {len(unique_activities)}")

    # #4. Cluster Using Union-Find
    uf = UnionFind(len(unique_activities))
    for i in range(len(unique_activities)):
        for j in range(i + 1, len(unique_activities)):
            if similarity_matrix[i, j] >= similarity_threshold:
                uf.union(i, j)

    clusters = defaultdict(list)
    for i, act in enumerate(unique_activities):
        root = uf.find(i)
        clusters[root].append(act)

    valid_clusters = {k: v for k, v in clusters.items() if len(v) >= min_synonym_group_size}
    num_clusters = len(valid_clusters)
    print(f"Number of synonym clusters discovered: {num_clusters}")

    # Build activity_to_cluster
    activity_to_cluster = {}
    for i, act in enumerate(unique_activities):
        root = uf.find(i)
        activity_to_cluster[act] = root if root in valid_clusters else -1

    # #5. Select Canonical Form (Majority/Mode)
    clean_to_originals = defaultdict(list)
    for orig in df['original_activity'].unique():
        cl = normalize_activity(orig)
        clean_to_originals[cl].append(orig)

    freq_original = df['original_activity'].value_counts()

    canonical_per_cluster = {}
    for root, cluster_cleans in valid_clusters.items():
        cluster_originals = set()
        for cl in cluster_cleans:
            cluster_originals.update(clean_to_originals.get(cl, []))
        if cluster_originals:
            freqs = {orig: freq_original.get(orig, 0) for orig in cluster_originals}
            canonical = max(freqs, key=freqs.get)
            canonical_per_cluster[root] = canonical

    clean_to_canonical = {}
    for root, canonical in canonical_per_cluster.items():
        for cl in valid_clusters[root]:
            clean_to_canonical[cl] = canonical

    df['canonical_activity'] = df.apply(
        lambda row: clean_to_canonical.get(row['Activity_clean'], row['original_activity']), axis=1
    )

    # Assign cluster ids
    cluster_id_map = {}
    cid = 0
    for root in valid_clusters:
        for clean in valid_clusters[root]:
            cluster_id_map[clean] = cid
        cid += 1

    df['SynonymGroup'] = df['Activity_clean'].map(cluster_id_map).fillna(-1).astype(int)

    df['is_synonymous_event'] = (
        (df['SynonymGroup'] != -1) & (df['original_activity'] != df['canonical_activity'])
    ).astype(int)

# #6. Calculate Detection Metrics (BEFORE FIXING)
print("=== Detection Performance Metrics ===")
if label_column in df.columns:
    y_true = (
        (df[label_column].notna()) & (df[label_column].astype(str) != '')
    ).astype(int)
    y_pred = df['is_synonymous_event']
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
else:
    print("Precision: 0.0000")
    print("Recall: 0.0000")
    print("F1-Score: 0.0000")
    print("No ground-truth labels available for evaluation.")

# #7. Integrity Check
print(f"Total synonym clusters found: {num_clusters}")
print(f"Total events flagged as synonyms: {df['is_synonymous_event'].sum()}")
print(f"Total canonical/clean events: {len(df) - df['is_synonymous_event'].sum()}")

# #8. Fix Activities
df['Activity'] = df['canonical_activity']

# #10. Create Final Fixed Dataset
helper_columns = ['original_activity', 'Activity_clean', 'canonical_activity', 'SynonymGroup', 'is_synonymous_event']
final_df = df.drop(columns=[col for col in helper_columns if col in df.columns], errors='ignore')

# #11. Save Output and Summary
final_df.to_csv(output_file, index=False)

before_unique = df['original_activity'].nunique()
after_unique = final_df['Activity'].nunique()
reduction = before_unique - after_unique
reduction_pct = (reduction / before_unique * 100) if before_unique > 0 else 0
replaced = df['is_synonymous_event'].sum()
rate = (replaced / len(df) * 100) if len(df) > 0 else 0

print(f"Total rows: {len(final_df)}")
print(f"Synonym clusters found: {num_clusters}")
print(f"Synonymous events replaced: {replaced}")
print(f"Replacement rate: {rate:.2f}%")
print(f"Unique activities before: {before_unique} → after: {after_unique}")
print(f"Activity reduction: {reduction} ({reduction_pct:.2f}%)")
print(f"Output file path: {output_file}")

# Sample transformations
changes = df[df['original_activity'] != df['canonical_activity']][['original_activity', 'canonical_activity']].drop_duplicates()
if not changes.empty:
    print("Sample transformations (up to 10):")
    for _, row in changes.head(10).iterrows():
        print(f"'{row['original_activity']}' → '{row['canonical_activity']}'")
else:
    print("No transformations applied.")

print(f"Run {run_number}: Processed dataset saved to: {output_file}")
print(f"Run {run_number}: Final dataset shape: {final_df.shape}")
print(f"Run {run_number}: Dataset: {dataset_name}")
print(f"Run {run_number}: Task type: synonymous")