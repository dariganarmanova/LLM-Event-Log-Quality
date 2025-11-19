# Generated script for Pub-Synonymous - Run 2
# Generated on: 2025-11-18T18:53:23.547259
# Model: grok-4-fast

import pandas as pd
import re
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
            return False
        if self.rank[rootP] < self.rank[rootQ]:
            self.parent[rootP] = rootQ
        elif self.rank[rootP] > self.rank[rootQ]:
            self.parent[rootQ] = rootP
        else:
            self.parent[rootQ] = rootP
            self.rank[rootP] += 1
        return True

def normalize_activity(activity):
    activity = str(activity).lower()
    activity = re.sub(r'[^a-z0-9\s]', ' ', activity)
    activity = ' '.join(activity.split()).strip()
    return activity

# Configuration parameters
similarity_threshold = 0.8
min_synonym_group_size = 2
ngram_range = (1, 3)
label_column = 'label'

# Run identifier
run_number = 2
dataset_name = 'pub'
input_file = 'data/pub/Pub-Synonymous.csv'
output_file = 'data/pub/pub_synonymous_cleaned_run2.csv'

# Load the data
df = pd.read_csv(input_file)
print(f"Run {run_number}: Original dataset shape: {df.shape}")

# Normalize column naming
if 'CaseID' in df.columns and 'Case' not in df.columns:
    df = df.rename(columns={'CaseID': 'Case'})

if 'Activity' not in df.columns:
    raise ValueError("Missing 'Activity' column.")

# Store original values
df['original_activity'] = df['Activity'].astype(str).fillna('')

# Ensure Activity is string-typed
df['Activity'] = df['original_activity']

# Parse timestamp if exists
if 'Timestamp' in df.columns:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

# Sort if both Case and Timestamp exist
if 'Case' in df.columns and 'Timestamp' in df.columns:
    df = df.sort_values(['Case', 'Timestamp']).reset_index(drop=True)

print(f"Dataset shape: {df.shape}")
print(df.head())
print(f"Number of unique Activity values: {df['original_activity'].nunique()}")

# Normalize Activity labels
df['Activity_clean'] = df['original_activity'].apply(normalize_activity)
df['Activity_clean'] = df['Activity_clean'].replace('', 'empty_activity')

# TF-IDF Embedding and Similarity
unique_activities = list(df['Activity_clean'].unique())
if len(unique_activities) < 2:
    print("Warning: Less than 2 unique activities, skipping clustering.")
    df['SynonymGroup'] = -1
    df['canonical_activity'] = df['original_activity']
    df['is_synonymous_event'] = 0
    num_clusters = 0
else:
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=ngram_range, lowercase=True, min_df=1)
    tfidf_matrix = vectorizer.fit_transform(unique_activities)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Number of unique activities: {len(unique_activities)}")

    # Cluster Using Union-Find
    uf = UnionFind(len(unique_activities))
    for i in range(len(unique_activities)):
        for j in range(i + 1, len(unique_activities)):
            if similarity_matrix[i, j] >= similarity_threshold:
                uf.union(i, j)

    clusters = {}
    for i, act in enumerate(unique_activities):
        root = uf.find(i)
        clusters.setdefault(root, []).append(act)

    valid_clusters = {k: v for k, v in clusters.items() if len(v) >= min_synonym_group_size}

    activity_to_cluster = {}
    cluster_canonical = {}
    cluster_id = 0
    original_freq = df['original_activity'].value_counts()

    for root, members in valid_clusters.items():
        for mem in members:
            activity_to_cluster[mem] = cluster_id

        # Select canonical form (most frequent original in cluster)
        all_originals = set()
        for mem in members:
            mask = df['Activity_clean'] == mem
            all_originals.update(df.loc[mask, 'original_activity'].dropna().unique())

        if all_originals:
            freqs = {o: original_freq.get(o, 0) for o in all_originals}
            canonical = max(freqs, key=freqs.get)
            cluster_canonical[cluster_id] = canonical

        cluster_id += 1

    df['SynonymGroup'] = df['Activity_clean'].map(activity_to_cluster).fillna(-1).astype(int)
    df['canonical_activity'] = df.apply(
        lambda row: cluster_canonical.get(row['SynonymGroup'], row['original_activity'])
        if row['SynonymGroup'] != -1 else row['original_activity'],
        axis=1
    )
    df['is_synonymous_event'] = ((df['SynonymGroup'] != -1) & (df['original_activity'] != df['canonical_activity'])).astype(int)
    num_clusters = len(valid_clusters)

if len(unique_activities) >= 2 or True:  # Always print after setting
    print(f"Number of synonym clusters discovered: {num_clusters}")

# Calculate Detection Metrics
print("=== Detection Performance Metrics ===")
if label_column in df.columns:
    y_true = ((df[label_column].notna()) & (df[label_column].astype(str).str.strip() != '')).astype(int)
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
print(f"Total synonym clusters found: {num_clusters}")
print(f"Total events flagged as synonyms: {(df['is_synonymous_event'] == 1).sum()}")
print(f"Total canonical/clean events: {(df['is_synonymous_event'] == 0).sum()}")

# Fix Activities
df['Activity'] = df['canonical_activity']

# Create Final Fixed Dataset
helper_cols = ['original_activity', 'Activity_clean', 'canonical_activity', 'SynonymGroup', 'is_synonymous_event']
final_df = df.drop(columns=[col for col in helper_cols if col in df.columns], errors='ignore')

# Summary
total_rows = len(final_df)
syn_replaced = (df['is_synonymous_event'] == 1).sum()
replacement_rate = (syn_replaced / total_rows * 100) if total_rows > 0 else 0
unique_before = df['original_activity'].nunique()
unique_after = final_df['Activity'].nunique()
reduction = unique_before - unique_after
reduction_rate = (reduction / unique_before * 100) if unique_before > 0 else 0

print(f"Total rows: {total_rows}")
print(f"Synonym clusters found: {num_clusters}")
print(f"Synonymous events replaced: {syn_replaced}")
print(f"Replacement rate: {replacement_rate:.2f}%")
print(f"Unique activities before: {unique_before} → after: {unique_after}")
print(f"Activity reduction: {reduction} ({reduction_rate:.2f}%)")
print(f"Output file path: {output_file}")

# Sample transformations
changed = df[df['original_activity'] != df['canonical_activity']][['original_activity', 'canonical_activity']].drop_duplicates()
if not changed.empty:
    print("Sample transformations (up to 10):")
    for _, row in changed.head(10).iterrows():
        print(f"  '{row['original_activity']}' → '{row['canonical_activity']}'")
else:
    print("No transformations applied.")

# Save Output
final_df.to_csv(output_file, index=False)

# REQUIRED: Print summary
print(f"Run {run_number}: Processed dataset saved to: {output_file}")
print(f"Run {run_number}: Final dataset shape: {final_df.shape}")
print(f"Run {run_number}: Dataset: {dataset_name}")
print(f"Run {run_number}: Task type: synonymous")