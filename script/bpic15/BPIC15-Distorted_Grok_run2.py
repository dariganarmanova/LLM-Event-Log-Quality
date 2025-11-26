# Generated script for BPIC15-Distorted - Run 2
# Generated on: 2025-11-18T21:26:18.961877
# Model: grok-4-fast

import pandas as pd
import re
from collections import defaultdict, Counter
from itertools import combinations
from sklearn.metrics import precision_recall_fscore_support

# Configuration
input_file = 'data/bpic15/BPIC15-Distorted.csv'
output_file = 'data/bpic15/bpic15_distorted_cleaned_run2.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
distorted_suffix = ':distorted'

# Algorithm Configuration Parameters
time_threshold = 2.0
require_same_resource = False
min_matching_events = 2
max_mismatches = 1
activity_suffix_pattern = r"(_signed\d*|_\d+)$"
similarity_threshold = 0.8
case_sensitive = False
use_fuzzy_matching = False

# Jaccard parameters
ngram_size = 3
min_length = 4

print(f"Run 2: Original dataset shape: {pd.read_csv(input_file).shape}")

# #1. Load CSV
df = pd.read_csv(input_file)

# Normalize column names if needed
column_mapping = {'Case ID': 'Case', 'case': 'Case', 'Activity ID': 'Activity', 'activity': 'Activity',
                  'Complete Timestamp': 'Timestamp', 'timestamp': 'Timestamp'}
df.rename(columns=column_mapping, inplace=True)

# Ensure required columns exist (assume they do, but check)
required_cols = [case_column, activity_column, timestamp_column]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Required column {col} not found")

# Store original Activity
df['original_activity'] = df[activity_column]

print(f"Run 2: Loaded dataset with shape {df.shape}")

# #2. Identify Distorted Activities
def remove_suffix(text):
    if pd.isna(text) or not isinstance(text, str):
        return text
    if text.endswith(distorted_suffix):
        return text[:-len(distorted_suffix)]
    return text

df['BaseActivity'] = df[activity_column].apply(remove_suffix)

def has_distorted_suffix(text):
    if pd.isna(text) or not isinstance(text, str):
        return 0
    return 1 if text.endswith(distorted_suffix) else 0

df['isdistorted'] = df[activity_column].apply(has_distorted_suffix)

# Incorporate activity_suffix_pattern
def clean_activity(text):
    if pd.isna(text) or not isinstance(text, str):
        return text
    text = re.sub(activity_suffix_pattern, '', text)
    return text

df['BaseActivity'] = df['BaseActivity'].apply(clean_activity)

# #3. Preprocess Activity Names
def preprocess_activity(text):
    if pd.isna(text) or not isinstance(text, str) or text == '':
        return ''
    text = text.lower() if not case_sensitive else text
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

df['ProcessedActivity'] = df['BaseActivity'].apply(preprocess_activity)

# #4. Calculate Jaccard N-gram Similarity
def generate_ngrams(text, n):
    if len(text) < n:
        return set()
    return set(text[i:i + n] for i in range(len(text) - n + 1))

def jaccard_similarity(text1, text2, n):
    set1 = generate_ngrams(text1, n)
    set2 = generate_ngrams(text2, n)
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0

# Build processed_to_originals for comparable activities
processed_to_originals = defaultdict(list)
activity_freq = Counter(df['original_activity'])
unique_originals = df['original_activity'].unique()

for orig in unique_originals:
    base = remove_suffix(orig)  # Already cleaned, but ensure
    base = clean_activity(base)
    proc = preprocess_activity(base)
    if len(proc) >= min_length and proc:
        processed_to_originals[proc].append(orig)

unique_processed = list(processed_to_originals.keys())

# Find similar pairs
similar_pairs = []
num_pairs = 0
for p1, p2 in combinations(unique_processed, 2):
    num_pairs += 1
    sim = jaccard_similarity(p1, p2, ngram_size)
    if sim >= similarity_threshold:
        similar_pairs.append((p1, p2))
    if num_pairs % 100 == 0:
        print(f"Progress: Processed {num_pairs} comparisons")

print(f"Found {len(similar_pairs)} similar pairs above threshold {similarity_threshold}")

# #5. Cluster Similar Activities (Union-Find)
parent = {p: p for p in unique_processed}
rank = {p: 0 for p in unique_processed}

def find(p):
    if parent[p] != p:
        parent[p] = find(parent[p])
    return parent[p]

def union(p1, p2):
    r1 = find(p1)
    r2 = find(p2)
    if r1 == r2:
        return
    if rank[r1] < rank[r2]:
        parent[r1] = r2
    elif rank[r1] > rank[r2]:
        parent[r2] = r1
    else:
        parent[r2] = r1
        rank[r1] += 1

for pair in similar_pairs:
    union(pair[0], pair[1])

# Build clusters
clusters = defaultdict(list)
for p in unique_processed:
    root = find(p)
    clusters[root].append(p)

valid_clusters = {k: v for k, v in clusters.items() if len(v) >= min_matching_events}

print(f"Detected {len(valid_clusters)} clusters with >= {min_matching_events} members")

# #6. Majority Voting Within Clusters
overall_mapping = {}
for root, cluster_ps in valid_clusters.items():
    all_origs = []
    for p in cluster_ps:
        all_origs.extend(processed_to_originals[p])
    all_origs = list(set(all_origs))
    if all_origs:
        cluster_freq = Counter({orig: activity_freq[orig] for orig in all_origs})
        canonical = cluster_freq.most_common(1)[0][0]
        for orig in all_origs:
            overall_mapping[orig] = canonical

# Map singletons to themselves
for orig in unique_originals:
    if orig not in overall_mapping:
        overall_mapping[orig] = orig

# Apply to dataframe
df['canonical_activity'] = df['original_activity'].map(overall_mapping)
df['is_distorted'] = (df['original_activity'] != df['canonical_activity']).astype(int)

# #7. Calculate Detection Metrics (BEFORE FIXING)
print("=== Detection Performance Metrics ===")
if label_column in df.columns:
    def normalize_label(label):
        if pd.isna(label):
            return 0
        label_str = str(label).lower()
        return 1 if 'distorted' in label_str else 0

    y_true = df[label_column].apply(normalize_label)
    y_pred = df['is_distorted']
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    prec_threshold_met = "✓" if precision >= 0.6 else "✗"
    print(f"{prec_threshold_met} Precision threshold (≥ 0.6) met/not met")
else:
    print("Precision: 0.0000")
    print("Recall: 0.0000")
    print("F1-Score: 0.0000")
    print("No labels available for metric calculation")

# #8. Integrity Check
total_distortion_clusters = len(valid_clusters)
total_distorted = df['is_distorted'].sum()
total_clean = len(df) - total_distorted
activities_fixed = total_distorted

print(f"Total distortion clusters detected: {total_distortion_clusters}")
print(f"Total activities marked as distorted: {total_distorted}")
print(f"Clean activities not modified: {total_clean}")
print(f"Activities to be fixed: {activities_fixed}")

# #9. Fix Activities
df['Activity_fixed'] = df['canonical_activity']

# #10. Save Output
# Format timestamp
df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')

# Prepare output dataframe
df_output = df[[case_column, timestamp_column]].copy()
if 'Variant' in df.columns:
    df_output['Variant'] = df['Variant']
if 'Resource' in df.columns:
    df_output['Resource'] = df['Resource']
df_output[activity_column] = df[activity_column]
df_output['Activity_fixed'] = df['Activity_fixed']
if label_column in df.columns:
    df_output[label_column] = df[label_column]

df_output.to_csv(output_file, index=False)

# #11. Summary Statistics
total_events = len(df)
distorted_events = total_distorted
unique_before = len(df[activity_column].unique())
unique_after = len(df['Activity_fixed'].unique())
reduction = unique_before - unique_after
percentage = (reduction / unique_before * 100) if unique_before > 0 else 0.0

print(f"Total number of events: {total_events}")
print(f"Number of distorted events detected: {distorted_events}")
print(f"Unique activities before fixing: {unique_before}")
print(f"Unique activities after fixing: {unique_after}")
print(f"Activity reduction count and percentage: {reduction} ({percentage:.2f}%)")
print(f"Output file path: {output_file}")

# Sample transformations
changed = df[df['is_distorted'] == 1][['original_activity', 'canonical_activity']].drop_duplicates().head(10)
print("Sample of up to 10 transformations showing: original → canonical")
for _, row in changed.iterrows():
    print(f"- {row['original_activity']} → {row['canonical_activity']}")

print(f"Run 2: Processed dataset saved to: data/bpic15/bpic15_distorted_cleaned_run2.csv")
print(f"Run 2: Final dataset shape: {df_output.shape}")
print(f"Run 2: Dataset: bpic15")
print(f"Run 2: Task type: distorted")