# Generated script for Pub-Distorted - Run 2
# Generated on: 2025-11-18T18:30:58.607018
# Model: grok-4-fast

import pandas as pd
import re
from collections import defaultdict, Counter

def preprocess(text):
    if pd.isna(text):
        return ''
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def generate_ngrams(text, n):
    if not text or len(text) < n:
        return set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}

def jaccard_similarity(set1, set2):
    if not set1 or not set2:
        return 0.0
    inter = len(set1.intersection(set2))
    union_len = len(set1.union(set2))
    return inter / union_len if union_len > 0 else 0.0

def jaccard_ngram(text1, text2, n):
    s1 = generate_ngrams(text1, n)
    s2 = generate_ngrams(text2, n)
    return jaccard_similarity(s1, s2)

def find(parent, i):
    if parent[i] != i:
        parent[i] = find(parent, parent[i])
    return parent[i]

def union(parent, i, j):
    pi = find(parent, i)
    pj = find(parent, j)
    if pi != pj:
        parent[pi] = pj

# Configuration
input_file = 'data/pub/Pub-Distorted.csv'
output_file = 'data/pub/pub_distorted_cleaned_run2.csv'
distorted_suffix = ':distorted'
label_column = 'label'
ngram_size = 3
similarity_threshold = 0.8
min_length = 4

print(f"Run 2: Original dataset shape: {pd.read_csv(input_file).shape}")

# Load data
df = pd.read_csv(input_file)

# Normalize column names if needed
if 'Case ID' in df.columns:
    df.rename(columns={'Case ID': 'Case'}, inplace=True)

# Ensure required columns (assume present, but check)
required = ['Case', 'Activity', 'Timestamp']
for col in required:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# Store original activity
df['original_activity'] = df['Activity']

# Step 2: Identify distorted and base activity
df['isdistorted'] = df['Activity'].str.endswith(distorted_suffix, na=False).astype(int)
df['BaseActivity'] = df['Activity'].str.rstrip(distorted_suffix)

# Step 3: Preprocess to ProcessedActivity
df['ProcessedActivity'] = df['BaseActivity'].apply(preprocess)

# Step 4: Find similar pairs
unique_pro = [p for p in df['ProcessedActivity'].unique() if p and len(p) >= min_length]
n = len(unique_pro)
similar_pairs = []
total_comparisons = n * (n - 1) // 2
completed = 0
for i in range(n):
    for j in range(i + 1, n):
        sim = jaccard_ngram(unique_pro[i], unique_pro[j], ngram_size)
        if sim >= similarity_threshold:
            similar_pairs.append((unique_pro[i], unique_pro[j]))
        completed += 1
        if completed % 100 == 0 and completed > 0:
            print(f"Completed {completed} comparisons out of {total_comparisons}")

# Step 5: Cluster with union-find
parent = {proc: proc for proc in unique_pro}
for a, b in similar_pairs:
    union(parent, a, b)

clusters = defaultdict(list)
for proc in unique_pro:
    root = find(parent, proc)
    clusters[root].append(proc)

# Get counters per processed
proc_to_orig_counter = df[df['ProcessedActivity'].isin(unique_pro)].groupby('ProcessedActivity')['Activity'].apply(lambda g: Counter(g)).to_dict()

# Step 6: Majority voting
activity_to_canonical = {}
for root, cluster_procs in clusters.items():
    total_counter = Counter()
    for proc in cluster_procs:
        total_counter += proc_to_orig_counter.get(proc, Counter())
    if total_counter:
        canonical = total_counter.most_common(1)[0][0]
        for orig in total_counter.keys():
            activity_to_canonical[orig] = canonical

# Apply canonical
df['Activity_fixed'] = df['Activity'].map(activity_to_canonical).fillna(df['Activity'])
df['is_distorted'] = (df['Activity'] != df['Activity_fixed']).astype(int)

# Step 7: Detection metrics
print("=== Detection Performance Metrics ===")
if label_column not in df.columns:
    prec = rec = f1 = 0.0000
    print("Precision: 0.0000")
    print("Recall: 0.0000")
    print("F1-Score: 0.0000")
    print("No labels available for metric calculation")
else:
    def normalize_label(l):
        if pd.isna(l):
            return 0
        return 1 if 'distorted' in str(l).lower() else 0
    y_true = df[label_column].apply(normalize_label)
    y_pred = df['is_distorted']
    from sklearn.metrics import precision_score, recall_score, f1_score
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
print(f"{'✓' if prec >= 0.6 else '✗'} Precision threshold (≥ 0.6) met/not met")

# Step 8: Integrity check
num_distortion_clusters = len([v for v in clusters.values() if len(v) >= 2])
total_distorted = df['is_distorted'].sum()
activities_fixed = total_distorted
clean_unchanged = len(df) - total_distorted
print(f"Total distortion clusters detected: {num_distortion_clusters}")
print(f"Total activities marked as distorted: {total_distorted}")
print(f"Activities to be fixed: {activities_fixed}")
print(f"Clean activities unchanged: {clean_unchanged}")

# Step 10: Prepare output (Step 9 is integrated via Activity_fixed)
output_df = df[['Case']].copy()
if 'Variant' in df.columns:
    output_df['Variant'] = df['Variant']
if 'Resource' in df.columns:
    output_df['Resource'] = df['Resource']
output_df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
output_df['Activity'] = df['Activity']
output_df['Activity_fixed'] = df['Activity_fixed']
if label_column in df.columns:
    output_df[label_column] = df[label_column]

output_df.to_csv(output_file, index=False)

# Step 11: Summary statistics
total_events = len(df)
distorted_detected = total_distorted
before_unique = df['Activity'].nunique()
after_unique = df['Activity_fixed'].nunique()
reduction = before_unique - after_unique
percentage = (reduction / before_unique * 100) if before_unique > 0 else 0.0
print(f"Total number of events: {total_events}")
print(f"Number of distorted events detected: {distorted_detected}")
print(f"Unique activities before fixing: {before_unique}")
print(f"Unique activities after fixing: {after_unique}")
print(f"Activity reduction: {reduction} ({percentage:.2f}%)")

changes = df[df['Activity'] != df['Activity_fixed']][['Activity', 'Activity_fixed']].drop_duplicates().head(10)
print("Sample transformations:")
if len(changes) > 0:
    for _, row in changes.iterrows():
        print(f"  {row['Activity']} → {row['Activity_fixed']}")
else:
    print("  No transformations needed")

print(f"Output file path: {output_file}")

# Required prints
print(f"Run 2: Processed dataset saved to: data/pub/pub_distorted_cleaned_run2.csv")
print(f"Run 2: Final dataset shape: {output_df.shape}")
print(f"Run 2: Dataset: pub")
print(f"Run 2: Task type: distorted")