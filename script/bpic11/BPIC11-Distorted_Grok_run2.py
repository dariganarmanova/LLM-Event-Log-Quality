# Generated script for BPIC11-Distorted - Run 2
# Generated on: 2025-11-18T22:21:10.803799
# Model: grok-4-fast

import pandas as pd
import re
from itertools import combinations
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

# Algorithm parameters
ngram_size = 3
min_length = 4
jaccard_threshold = 0.56  # Use Jaccard threshold as per task parameters since use_fuzzy_matching=False

# Input and output
input_file = 'data/bpic11/BPIC11-Distorted.csv'
output_file = 'data/bpic11/bpic11_distorted_cleaned_run2.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
distorted_suffix = ':distorted'

print(f"Run 2: Original dataset shape: {pd.read_csv(input_file).shape}")

# #1. Load CSV
df = pd.read_csv(input_file)

# Normalize column names if needed
column_mapping = {'Case ID': 'Case', 'caseid': 'Case', 'CaseID': 'Case'}
for old, new in column_mapping.items():
    if old in df.columns:
        df.rename(columns={old: new}, inplace=True)

# Ensure required columns exist (assume they do, but check)
required_cols = [case_column, activity_column, timestamp_column]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Required column {col} not found")

# Optional columns
has_variant = 'Variant' in df.columns
has_resource = 'Resource' in df.columns
has_label = label_column in df.columns

# Store original Activity
df['original_activity'] = df[activity_column].copy()

# #2. Identify Distorted Activities
df['isdistorted'] = 0
mask = df[activity_column].str.endswith(distorted_suffix, na=False)
df.loc[mask, 'isdistorted'] = 1
df['BaseActivity'] = df[activity_column].str.replace(distorted_suffix + r'$', '', regex=True)

# Remove activity suffix pattern
df['BaseActivity'] = df['BaseActivity'].str.replace(activity_suffix_pattern, '', regex=True)

# #3. Preprocess Activity Names
def preprocess(text):
    if pd.isna(text):
        return ''
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = ' '.join(text.split())
    return text

df['ProcessedActivity'] = df['BaseActivity'].apply(preprocess)

# #4. Calculate Jaccard N-gram Similarity
def generate_ngrams(text, n=ngram_size):
    if pd.isna(text) or len(text) < n:
        return set()
    return set(text[i:i+n] for i in range(len(text) - n + 1))

def jaccard_similarity(text1, text2):
    set1 = generate_ngrams(text1)
    set2 = generate_ngrams(text2)
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

unique_processed = df['ProcessedActivity'].dropna().unique()
unique_long = [p for p in unique_processed if len(p) >= min_length]

similar_pairs = []
for i, act1 in enumerate(unique_long):
    for act2 in unique_long[i+1:]:
        sim = jaccard_similarity(act1, act2)
        if sim >= jaccard_threshold:
            similar_pairs.append((act1, act2))
    if (i + 1) % 100 == 0:
        print(f"Progress: {i+1}/{len(unique_long)} comparisons completed")

# #5. Cluster Similar Activities (Union-Find)
def find(parent, i):
    if parent[i] != i:
        parent[i] = find(parent, parent[i])
    return parent[i]

def union(parent, rank, x, y):
    xroot = find(parent, x)
    yroot = find(parent, y)
    if xroot == yroot:
        return
    if rank[xroot] < rank[yroot]:
        parent[xroot] = yroot
    elif rank[xroot] > rank[yroot]:
        parent[yroot] = xroot
    else:
        parent[yroot] = xroot
        rank[xroot] += 1

parent = {act: act for act in unique_long}
rank = {act: 0 for act in unique_long}

for pair in similar_pairs:
    union(parent, rank, pair[0], pair[1])

clusters = {}
for act in unique_long:
    root = find(parent, act)
    clusters.setdefault(root, []).append(act)

multi_clusters = {k: v for k, v in clusters.items() if len(v) >= 2}

# #6. Majority Voting Within Clusters
mapping = {}
for root, cluster_procs in multi_clusters.items():
    cluster_mask = df['ProcessedActivity'].isin(cluster_procs)
    cluster_df = df[cluster_mask]
    if len(cluster_df) == 0:
        continue
    variant_counts = cluster_df['BaseActivity'].value_counts()
    canonical = variant_counts.index[0]
    for proc in cluster_procs:
        mapping[proc] = canonical

# Set canonical_activity
df['canonical_activity'] = df['ProcessedActivity'].map(mapping)
df['canonical_activity'] = df['canonical_activity'].fillna(df['BaseActivity'])

# Mark distorted
df['is_distorted'] = (df[activity_column] != df['canonical_activity']).astype(int)

# #7. Calculate Detection Metrics (BEFORE FIXING)
print("=== Detection Performance Metrics ===")
if has_label:
    def normalize_label(label):
        if pd.isna(label):
            return 0
        return 1 if 'distorted' in str(label).lower() else 0

    y_true = df[label_column].apply(normalize_label)
    y_pred = df['is_distorted']
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    prec_met = "✓" if prec >= 0.6 else "✗"
    print(f"{prec_met} Precision threshold (≥ 0.6) met/not met")
else:
    print("Precision: 0.0000")
    print("Recall: 0.0000")
    print("F1-Score: 0.0000")
    print("No labels available for metric calculation")

# #8. Integrity Check
num_clusters = len(multi_clusters)
num_distorted = df['is_distorted'].sum()
num_clean_unchanged = len(df) - num_distorted
print(f"Total distortion clusters detected: {num_clusters}")
print(f"Total activities marked as distorted: {num_distorted}")
print(f"Clean activities that were NOT modified: {num_clean_unchanged}")
print(f"Activities to be fixed: {num_distorted}")

# #9. Fix Activities
df['Activity_fixed'] = df['canonical_activity']

# #10. Save Output
# Format timestamp
df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')

output_cols = [case_column, timestamp_column, activity_column, 'Activity_fixed']
if has_variant:
    output_cols.insert(2, 'Variant')
if has_resource:
    output_cols.append('Resource')
if has_label:
    output_cols.append(label_column)

output_df = df[output_cols].copy()
output_df.to_csv(output_file, index=False)

# #11. Summary Statistics
total_events = len(df)
distorted_detected = num_distorted
unique_before = len(df[activity_column].unique())
unique_after = len(df['Activity_fixed'].unique())
reduction = unique_before - unique_after
reduction_pct = (reduction / unique_before * 100) if unique_before > 0 else 0

print(f"Total number of events: {total_events}")
print(f"Number of distorted events detected: {distorted_detected}")
print(f"Unique activities before fixing: {unique_before}")
print(f"Unique activities after fixing: {unique_after}")
print(f"Activity reduction count: {reduction} ({reduction_pct:.2f}%)")
print(f"Output file path: {output_file}")

# Sample transformations
changes = df[df[activity_column] != df['Activity_fixed']][[activity_column, 'Activity_fixed']].drop_duplicates()
print("Sample transformations (up to 10):")
for i in range(min(10, len(changes))):
    orig = changes.iloc[i][activity_column]
    fixed = changes.iloc[i]['Activity_fixed']
    print(f"{orig} → {fixed}")

print(f"Run 2: Processed dataset saved to: data/bpic11/bpic11_distorted_cleaned_run2.csv")
print(f"Run 2: Final dataset shape: {output_df.shape}")
print(f"Run 2: Dataset: bpic11")
print(f"Run 2: Task type: distorted")