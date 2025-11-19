# Generated script for Credit-Distorted - Run 2
# Generated on: 2025-11-18T19:18:05.703592
# Model: grok-4-fast

import pandas as pd
import re
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support

# Configuration
input_file = 'data/credit/Credit-Distorted.csv'
output_file = 'data/credit/credit_distorted_cleaned_run2.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
distorted_suffix = ':distorted'
ngram_size = 3
similarity_threshold = 0.56
min_length = 4

# Load CSV
df = pd.read_csv(input_file)
print(f"Run 2: Original dataset shape: {df.shape}")

# Normalize column names if needed
if 'Case ID' in df.columns:
    df.rename(columns={'Case ID': 'Case'}, inplace=True)
if 'Activity ID' in df.columns:
    df.rename(columns={'Activity ID': 'Activity'}, inplace=True)
# Assume required columns exist: Case, Activity, Timestamp

# Store original
df['original_activity'] = df[activity_column].copy()

# #2 Identify Distorted Activities
df['isdistorted'] = (df[activity_column].str.contains(distorted_suffix + '$', regex=True, na=False)).astype(int)
df['BaseActivity'] = df[activity_column].str.replace(distorted_suffix + '$', '', regex=True)

# #3 Preprocess Activity Names
def preprocess_activity(text):
    if pd.isna(text):
        return text
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = ' '.join(text.split())
    return text

df['ProcessedActivity'] = df['BaseActivity'].apply(preprocess_activity)

# Helper functions
def generate_ngrams(text, n):
    if pd.isna(text) or len(text) < n:
        return set()
    return set(text[i:i+n] for i in range(len(text) - n + 1))

def jaccard_similarity(text1, text2, n):
    set1 = generate_ngrams(text1, n)
    set2 = generate_ngrams(text2, n)
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

# #4 Calculate Jaccard N-gram Similarity
unique_processed = df['ProcessedActivity'].dropna().unique()
unique_activities = [act for act in unique_processed if len(act) >= min_length]

similar_pairs = []
num_comparisons = 0
for i in range(len(unique_activities)):
    for j in range(i + 1, len(unique_activities)):
        num_comparisons += 1
        if num_comparisons % 100 == 0:
            print(f"Processed {num_comparisons} comparisons...")
        sim = jaccard_similarity(unique_activities[i], unique_activities[j], ngram_size)
        if sim >= similarity_threshold:
            similar_pairs.append((unique_activities[i], unique_activities[j]))

# #5 Cluster Similar Activities (Union-Find)
parent = {act: act for act in unique_activities}
rank = {act: 0 for act in unique_activities}

def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(x, y):
    px = find(x)
    py = find(y)
    if px != py:
        if rank[px] > rank[py]:
            parent[py] = px
        elif rank[px] < rank[py]:
            parent[px] = py
        else:
            parent[py] = px
            rank[px] += 1

for pair in similar_pairs:
    union(pair[0], pair[1])

clusters = {}
for act in unique_activities:
    root = find(act)
    if root not in clusters:
        clusters[root] = []
    clusters[root].append(act)

# Handle short activities
processed_to_canonical = {}
short_acts = set(df[df['ProcessedActivity'].str.len() < min_length]['ProcessedActivity'].dropna().unique())
for short in short_acts:
    if pd.isna(short):
        continue
    mask = df['ProcessedActivity'] == short
    if mask.sum() > 0:
        counts = Counter(df.loc[mask, 'BaseActivity'].dropna())
        if counts:
            canonical = counts.most_common(1)[0][0]
            processed_to_canonical[short] = canonical

# #6 Majority Voting Within Clusters
for root, cluster_acts in clusters.items():
    mask = df['ProcessedActivity'].isin(cluster_acts)
    if mask.sum() == 0:
        continue
    cluster_df = df.loc[mask]
    counts = Counter(cluster_df['BaseActivity'].dropna())
    if counts:
        canonical = counts.most_common(1)[0][0]
        for act in cluster_acts:
            processed_to_canonical[act] = canonical

# Assign canonical
df['canonical_activity'] = df['ProcessedActivity'].map(processed_to_canonical)
df['canonical_activity'] = df['canonical_activity'].fillna(df['BaseActivity'])
df['is_distorted'] = (df['BaseActivity'] != df['canonical_activity']).astype(int)

# #7 Calculate Detection Metrics (BEFORE FIXING)
if label_column in df.columns:
    def normalize_label(l):
        if pd.isna(l):
            return 0
        return 1 if 'distorted' in str(l).lower() else 0
    y_true = df[label_column].apply(normalize_label)
    y_pred = df['is_distorted']
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, zero_division=0, average='binary')
    print("=== Detection Performance Metrics ===")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    prec_ok = "✓" if prec >= 0.6 else "✗"
    print(f"{prec_ok} Precision threshold (>= 0.6) met")
else:
    print("No labels available for metric calculation")
    prec, rec, f1 = 0.0000, 0.0000, 0.0000
    print("=== Detection Performance Metrics ===")
    print(f"Precision: 0.0000")
    print(f"Recall: 0.0000")
    print(f"F1-Score: 0.0000")
    print("✗ Precision threshold (>= 0.6) met")

# #8 Integrity Check
num_dist_clusters = len([v for v in clusters.values() if len(v) > 1])
total_distorted = df['is_distorted'].sum()
clean_unchanged = len(df) - total_distorted
print(f"Total distortion clusters detected: {num_dist_clusters}")
print(f"Total activities marked as distorted: {total_distorted}")
print(f"Activities to be fixed: {total_distorted}")
print(f"Clean activities that were NOT modified: {clean_unchanged}")

# #9 Fix Activities
df['Activity_fixed'] = df['canonical_activity']

# Format timestamp
df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')

# #10 Save Output
save_columns = [case_column, timestamp_column]
if 'Variant' in df.columns:
    save_columns.append('Variant')
save_columns += [activity_column, 'Activity_fixed']
if label_column in df.columns:
    save_columns.append(label_column)
if 'Resource' in df.columns:
    save_columns.append('Resource')
out_df = df[save_columns]
out_df.to_csv(output_file, index=False)

# #11 Summary Statistics
total_events = len(df)
unique_before = df[activity_column].nunique()
unique_after = df['Activity_fixed'].nunique()
reduction = unique_before - unique_after
percentage = (reduction / unique_before * 100) if unique_before > 0 else 0
print(f"Total number of events: {total_events}")
print(f"Number of distorted events detected: {total_distorted}")
print(f"Unique activities before fixing: {unique_before}")
print(f"Unique activities after fixing: {unique_after}")
print(f"Activity reduction count: {reduction} ({percentage:.2f}%)")
print(f"Output file path: {output_file}")

print("Sample transformations (up to 10):")
changed_samples = df[df['is_distorted'] == 1][[activity_column, 'Activity_fixed']].drop_duplicates().head(10)
if len(changed_samples) > 0:
    for _, row in changed_samples.iterrows():
        print(f"{row[activity_column]} → {row['Activity_fixed']}")
else:
    print("No transformations needed.")

print(f"Run 2: Processed dataset saved to: {output_file}")
print(f"Run 2: Final dataset shape: {out_df.shape}")
print(f"Run 2: Dataset: credit")
print(f"Run 2: Task type: distorted")