# Generated script for BPIC11-Synonymous - Run 1
# Generated on: 2025-11-19T13:07:25.706628
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
input_file = 'data/bpic11/BPIC11-Synonymous.csv'
output_file = 'data/bpic11/bpic11_synonymous_cleaned_run1.csv'
dataset_name = 'bpic11'
run_number = 1

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
    if pd.isna(activity):
        return 'empty_activity'
    activity = str(activity).lower()
    activity = re.sub(activity_suffix_pattern, '', activity)
    activity = re.sub(r'[^a-z0-9\s]', '', activity)
    activity = re.sub(r'\s+', ' ', activity).strip()
    if not activity:
        return 'empty_activity'
    return activity

# Load and Validate
df = pd.read_csv(input_file)
print(f"Run {run_number}: Original dataset shape: {df.shape}")

if 'CaseID' in df.columns and 'Case' not in df.columns:
    df = df.rename(columns={'CaseID': 'Case'})

if 'Activity' not in df.columns:
    raise ValueError("Missing 'Activity' column.")

df['original_activity'] = df['Activity'].astype(str)
df['Activity'] = df['original_activity']  # Ensure string type

if 'Timestamp' in df.columns:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

if 'Case' in df.columns and 'Timestamp' in df.columns:
    df = df.sort_values(['Case', 'Timestamp']).reset_index(drop=True)

df['Activity_clean'] = df['original_activity'].apply(normalize_activity)

print(df.head())
print(f"Number of unique Activity values: {df['original_activity'].nunique()}")

# TF-IDF Embedding and Similarity
unique_activities = list(df['original_activity'].dropna().unique())
num_unique = len(unique_activities)
if num_unique < 2:
    print("Warning: Less than 2 unique activities, skipping clustering.")
    canonical_mapping = {act: act for act in unique_activities}
    activity_to_cluster = {act: -1 for act in unique_activities}
    df['SynonymGroup'] = df['original_activity'].map(activity_to_cluster).fillna(-1)
    df['canonical_activity'] = df['original_activity'].map(canonical_mapping)
    df['is_synonymous_event'] = 0
    valid_clusters = {}
else:
    norms_for_tfidf = [normalize_activity(act) for act in unique_activities]
    vectorizer = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=ngram_range,
        lowercase=True,
        min_df=1
    )
    tfidf_matrix = vectorizer.fit_transform(norms_for_tfidf)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Unique activity count: {num_unique}")

    # Cluster Using Union-Find
    uf = UnionFind(num_unique)
    for i in range(num_unique):
        for j in range(i + 1, num_unique):
            if similarity_matrix[i, j] >= similarity_threshold:
                uf.union(i, j)

    # Build clusters
    clusters = defaultdict(list)
    for i in range(num_unique):
        root = uf.find(i)
        clusters[root].append(i)

    # Valid clusters
    valid_clusters_dict = {}
    cluster_counter = 0
    activity_to_cluster = {}
    for root, idx_list in clusters.items():
        cluster_acts = [unique_activities[idx] for idx in idx_list]
        if len(idx_list) >= min_synonym_group_size:
            cluster_counter += 1
            for idx in idx_list:
                act = unique_activities[idx]
                activity_to_cluster[act] = cluster_counter
            valid_clusters_dict[cluster_counter] = cluster_acts
        else:
            for idx in idx_list:
                act = unique_activities[idx]
                activity_to_cluster[act] = -1

    valid_clusters = valid_clusters_dict
    print(f"Number of synonym clusters discovered: {len(valid_clusters)}")

    # Select Canonical Form
    original_freq = df['original_activity'].value_counts()
    canonical_mapping = {}
    for cluster_id, cluster_acts in valid_clusters.items():
        cluster_freq = {act: original_freq.get(act, 0) for act in cluster_acts}
        canonical = max(cluster_freq, key=cluster_freq.get)
        for act in cluster_acts:
            canonical_mapping[act] = canonical

    # Unclustered
    for act in unique_activities:
        if act not in canonical_mapping:
            canonical_mapping[act] = act

    df['SynonymGroup'] = df['original_activity'].map(activity_to_cluster).fillna(-1)
    df['canonical_activity'] = df['original_activity'].map(canonical_mapping)
    df['is_synonymous_event'] = ((df['SynonymGroup'] != -1) & (df['original_activity'] != df['canonical_activity'])).astype(int)

# Calculate Detection Metrics (BEFORE FIXING)
if label_column in df.columns:
    y_true = ((df[label_column].notna()) & (df[label_column].astype(str).str.strip() != '')).astype(int)
    y_pred = df['is_synonymous_event']
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    print("=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
else:
    print("=== Detection Performance Metrics ===")
    print("Precision: 0.0000")
    print("Recall: 0.0000")
    print("F1-Score: 0.0000")
    print("No ground-truth labels available for evaluation.")

# Integrity Check
total_synonym_clusters = len(valid_clusters)
total_flagged = df['is_synonymous_event'].sum()
print(f"Total synonym clusters found: {total_synonym_clusters}")
print(f"Total events flagged as synonyms: {total_flagged}")
print(f"Total canonical/clean events: {len(df) - total_flagged}")

# Fix Activities
df['Activity'] = df['canonical_activity']

# Create Final Fixed Dataset
helper_columns = ['original_activity', 'Activity_clean', 'canonical_activity', 'SynonymGroup', 'is_synonymous_event']
final_df = df.drop(columns=[col for col in helper_columns if col in df.columns], errors='ignore')

# Save Output and Summary
final_df.to_csv(output_file, index=False)

print(f"Total rows: {len(final_df)}")
print(f"Synonym clusters found: {total_synonym_clusters}")
syn_replaced = total_flagged
print(f"Symonymous events replaced: {syn_replaced}")
replacement_rate = (syn_replaced / len(df) * 100) if len(df) > 0 else 0
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
changed = df[df['original_activity'] != df['canonical_activity']]
if not changed.empty:
    samples = changed[['original_activity', 'canonical_activity']].drop_duplicates().head(10)
    print("Sample transformations:")
    for _, row in samples.iterrows():
        print(f"'{row['original_activity']}' → '{row['canonical_activity']}'")
else:
    print("No transformations applied.")

print(f"Run {run_number}: Processed dataset saved to: {output_file}")
print(f"Run {run_number}: Final dataset shape: {final_df.shape}")
print(f"Run {run_number}: Dataset: {dataset_name}")
print(f"Run {run_number}: Task type: synonymous")