# Generated script for BPIC15-Synonymous - Run 3
# Generated on: 2025-11-18T21:59:06.413015
# Model: grok-4-fast

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import defaultdict

# Configuration parameters
similarity_threshold = 0.8
min_synonym_group_size = 2
ngram_range = (1, 3)
label_column = 'label'
activity_suffix_pattern = r"(_signed\d*|_\d+)$"
input_file = 'data/bpic15/BPIC15-Synonymous.csv'
output_file = 'data/bpic15/bpic15_synonymous_cleaned_run3.csv'
dataset_name = 'bpic15'
run_number = 3

def normalize_activity(activity):
    if pd.isna(activity):
        return ''
    activity = str(activity)
    # Remove suffix if matches pattern
    activity = re.sub(activity_suffix_pattern, '', activity)
    activity = activity.lower()
    # Remove non-alphanumeric except spaces
    activity = re.sub(r'[^a-z0-9\s]', ' ', activity)
    # Collapse multiple whitespace and trim
    activity = ' '.join(activity.split()).strip()
    return activity

# Load and validate
df = pd.read_csv(input_file)
print(f"Run {run_number}: Original dataset shape: {df.shape}")

# Normalize column naming
if 'CaseID' in df.columns:
    df.rename(columns={'CaseID': 'Case'}, inplace=True)

if 'Activity' not in df.columns:
    raise ValueError("Activity column is missing.")

original_activity_col = 'original_activity'
df[original_activity_col] = df['Activity'].fillna('').astype(str)

# Apply normalization
df['Activity_clean'] = df[original_activity_col].apply(normalize_activity)
df['Activity_clean'] = df['Activity_clean'].replace('', 'empty_activity')

# Handle timestamp if exists
if 'Timestamp' in df.columns:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

# Sort if Case and Timestamp exist
if 'Case' in df.columns and 'Timestamp' in df.columns:
    df = df.sort_values(by=['Case', 'Timestamp']).reset_index(drop=True)

print(df.shape)
print(df.head())
print(f"Number of unique Activity values: {df[original_activity_col].nunique()}")

# Compute clean_to_rep
clean_to_rep = {}
for clean, group in df.groupby('Activity_clean'):
    if len(group) > 0:
        orig_counts = group[original_activity_col].value_counts()
        clean_to_rep[clean] = orig_counts.index[0] if len(orig_counts) > 0 else ''

unique_activities = list(df['Activity_clean'].unique())

# TF-IDF and clustering
valid_clusters = {}
activity_to_cluster = {act: -1 for act in unique_activities}
activity_to_canonical = {clean: clean_to_rep.get(clean, '') for clean in unique_activities}

if len(unique_activities) >= min_synonym_group_size:
    vectorizer = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=ngram_range,
        lowercase=True,
        min_df=1
    )
    tfidf_matrix = vectorizer.fit_transform(unique_activities)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Number of unique activities: {len(unique_activities)}")

    # Union-Find
    class UnionFind:
        def __init__(self, n):
            self.parent = list(range(n))
            self.rank = [0] * n

        def find(self, x):
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]

        def union(self, x, y):
            px, py = self.find(x), self.find(y)
            if px != py:
                if self.rank[px] < self.rank[py]:
                    self.parent[px] = py
                elif self.rank[px] > self.rank[py]:
                    self.parent[py] = px
                else:
                    self.parent[py] = px
                    self.rank[px] += 1

    uf = UnionFind(len(unique_activities))
    for i in range(len(unique_activities)):
        for j in range(i + 1, len(unique_activities)):
            if similarity_matrix[i, j] >= similarity_threshold:
                uf.union(i, j)

    # Build clusters
    clusters = defaultdict(list)
    for i in range(len(unique_activities)):
        root = uf.find(i)
        clusters[root].append(i)

    valid_clusters = {k: v for k, v in clusters.items() if len(v) >= min_synonym_group_size}

    # activity_to_cluster
    activity_to_cluster = {}
    for act in unique_activities:
        # Find index
        idx = unique_activities.index(act)
        root = uf.find(idx)
        if root in valid_clusters:
            activity_to_cluster[act] = root
        else:
            activity_to_cluster[act] = -1

    # Canonical for clusters
    activity_to_canonical = {}
    for cluster_id, indices in valid_clusters.items():
        members = [unique_activities[idx] for idx in indices]
        freqs = {m: df['Activity_clean'].value_counts().get(m, 0) for m in members}
        if freqs:
            best_clean = max(freqs, key=freqs.get)
            canonical_original = clean_to_rep.get(best_clean, best_clean)
            for m in members:
                activity_to_canonical[m] = canonical_original
        else:
            for m in members:
                activity_to_canonical[m] = clean_to_rep.get(m, m)

    # Unclustered
    for clean in unique_activities:
        if clean not in activity_to_canonical:
            activity_to_canonical[clean] = clean_to_rep.get(clean, clean)

    print(f"Number of synonym clusters discovered: {len(valid_clusters)}")
else:
    print("Warning: Fewer than minimum activities for clustering.")

# Assign to DataFrame
df['SynonymGroup'] = df['Activity_clean'].map(activity_to_cluster).fillna(-1).astype(int)
df['canonical_activity'] = df['Activity_clean'].map(activity_to_canonical)
df['is_synonymous_event'] = ((df['SynonymGroup'] != -1) & (df[original_activity_col] != df['canonical_activity'])).astype(int)

# Detection metrics
has_labels = label_column in df.columns
if has_labels:
    y_true = (df[label_column].notna() & (df[label_column].astype(str).str.strip() != '')).astype(int)
    y_pred = df['is_synonymous_event']
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
else:
    prec = rec = f1 = 0.0
print("=== Detection Performance Metrics ===")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-Score: {f1:.4f}")
if not has_labels:
    print("No ground-truth labels available for evaluation")

# Integrity check
print(f"Total synonym clusters found: {len(valid_clusters)}")
print(f"Total events flagged as synonyms: {(df['is_synonymous_event'] == 1).sum()}")
print(f"Total canonical/clean events: {len(df) - (df['is_synonymous_event'] == 1).sum()}")

# Fix activities
df['Activity'] = df['canonical_activity']

# Summary
total_rows = len(df)
synonym_clusters = len(valid_clusters)
syn_events_replaced = (df['is_synonymous_event'] == 1).sum()
replacement_rate = (syn_events_replaced / total_rows * 100) if total_rows > 0 else 0
unique_before = df[original_activity_col].nunique()
unique_after = df['Activity'].nunique()
reduction = unique_before - unique_after
reduction_pct = (reduction / unique_before * 100) if unique_before > 0 else 0
print(f"Total rows: {total_rows}")
print(f"Synonym clusters found: {synonym_clusters}")
print(f"Synonymous events replaced: {syn_events_replaced}")
print(f"Replacement rate: {replacement_rate:.2f}%")
print(f"Unique activities before: {unique_before} → after: {unique_after}")
print(f"Activity reduction: {reduction} ({reduction_pct:.2f}%)")
print(f"Output file path: {output_file}")

# Sample transformations (only synonymous changes)
print("Sample transformations (up to 10):")
changed_mask = (df['is_synonymous_event'] == 1)
changed_df = df[changed_mask].head(10)
if len(changed_df) == 0:
    print("No synonymous transformations.")
else:
    for _, row in changed_df.iterrows():
        print(f"'{row[original_activity_col]}' → '{row['Activity']}'")

# Create final dataset
helper_cols = [original_activity_col, 'Activity_clean', 'canonical_activity', 'SynonymGroup', 'is_synonymous_event']
final_df = df.drop(columns=[col for col in helper_cols if col in df.columns], errors='ignore')

# Save
final_df.to_csv(output_file, index=False)
print(f"Run {run_number}: Processed dataset saved to: {output_file}")
print(f"Run {run_number}: Final dataset shape: {final_df.shape}")
print(f"Run {run_number}: Dataset: {dataset_name}")
print(f"Run {run_number}: Task type: synonymous")