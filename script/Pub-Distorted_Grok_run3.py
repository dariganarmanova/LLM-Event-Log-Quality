# Generated script for Pub-Distorted - Run 3
# Generated on: 2025-11-18T18:31:56.620631
# Model: grok-4-fast

import pandas as pd
import re
from collections import defaultdict
import itertools
from itertools import chain
from sklearn.metrics import precision_score, recall_score, f1_score
import time

# Configuration parameters for this task
ngram_size = 3
similarity_threshold = 0.56
min_length = 4
distorted_suffix = ':distorted'
label_column = 'label'
input_file = 'data/pub/Pub-Distorted.csv'
output_file = 'data/pub/pub_distorted_cleaned_run3.csv'

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text.strip())
    return text

def generate_ngrams(text, n):
    if len(text) < n:
        return set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}

def jaccard_similarity(text1, text2, n):
    set1 = generate_ngrams(text1, n)
    set2 = generate_ngrams(text2, n)
    if not set1 or not set2:
        return 0.0
    inter = len(set1 & set2)
    union_len = len(set1 | set2)
    return inter / union_len if union_len > 0 else 0.0

# #1. Load CSV
df = pd.read_csv(input_file)
print(f"Run 3: Original dataset shape: {df.shape}")

required = ['Case', 'Activity', 'Timestamp']
for col in required:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# Normalize CaseID to Case if needed
if 'CaseID' in df.columns and 'Case' not in df.columns:
    df['Case'] = df['CaseID']

df['original_activity'] = df['Activity'].copy()

# #2. Identify Distorted Activities
df['isdistorted'] = 0
df['BaseActivity'] = df['Activity']
mask = df['Activity'].str.endswith(distorted_suffix, na=False)
df.loc[mask, 'isdistorted'] = 1
df.loc[mask, 'BaseActivity'] = df['Activity'].str[:-len(distorted_suffix)]

# #3. Preprocess Activity Names
df['ProcessedActivity'] = df['BaseActivity'].apply(preprocess)

# #4. Calculate Jaccard N-gram Similarity
unique_processed = [p for p in df['ProcessedActivity'].dropna().unique() if len(p) >= min_length]
print(f"Number of unique processed activities for clustering: {len(unique_processed)}")

parent = {p: p for p in unique_processed}
rank = {p: 0 for p in unique_processed}

def find(p):
    if parent[p] != p:
        parent[p] = find(parent[p])
    return parent[p]

def union(p1, p2):
    r1 = find(p1)
    r2 = find(p2)
    if r1 != r2:
        if rank[r1] > rank[r2]:
            parent[r2] = r1
        elif rank[r1] < rank[r2]:
            parent[r1] = r2
        else:
            parent[r2] = r1
            rank[r1] += 1

start_time = time.time()
num_comparisons = 0
for i, j in itertools.combinations(range(len(unique_processed)), 2):
    num_comparisons += 1
    if num_comparisons % 100 == 0:
        print(f"Progress: Processed {num_comparisons} comparisons")
    p1 = unique_processed[i]
    p2 = unique_processed[j]
    sim = jaccard_similarity(p1, p2, ngram_size)
    if sim >= similarity_threshold:
        union(p1, p2)
print(f"Similarity computation completed in {time.time() - start_time:.2f} seconds")

# #5. Cluster Similar Activities (Union-Find)
clusters = defaultdict(list)
for p in unique_processed:
    root = find(p)
    clusters[root].append(p)

# Build processed_to_originals
processed_to_originals = defaultdict(list)
for orig in df['Activity'].unique():
    if orig.endswith(distorted_suffix):
        base = orig[:-len(distorted_suffix)]
    else:
        base = orig
    proc = preprocess(base)
    processed_to_originals[proc].append(orig)

# Frequency of original activities
freq_original = df['Activity'].value_counts().to_dict()

# #6. Majority Voting Within Clusters
mapping = {}
for root, proc_list in clusters.items():
    candidate_originals = set()
    for proc in proc_list:
        candidate_originals.update(processed_to_originals[proc])
    if len(candidate_originals) >= 2:
        freqs = {orig: freq_original.get(orig, 0) for orig in candidate_originals}
        if freqs:
            canonical = max(freqs, key=freqs.get)
            for orig in candidate_originals:
                mapping[orig] = canonical

# Assign canonical and is_distorted
df['canonical_activity'] = df['Activity']
for orig, can in mapping.items():
    df.loc[df['Activity'] == orig, 'canonical_activity'] = can
df['is_distorted'] = (df['Activity'] != df['canonical_activity']).astype(int)

# #7. Calculate Detection Metrics (BEFORE FIXING)
has_label = 'label' in df.columns
if has_label:
    df['label'] = df['label'].astype(str)
    def normalize_label(label):
        if pd.isna(label) or label == 'nan' or label == 'None':
            return 0
        label_lower = str(label).lower()
        if 'distorted' in label_lower:
            return 1
        return 0
    y_true = df['label'].apply(normalize_label)
    y_pred = df['is_distorted']
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print("=== Detection Performance Metrics ===")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    if prec >= 0.6:
        print("✓ Precision threshold (≥ 0.6) met")
    else:
        print("✗ Precision threshold (≥ 0.6) not met")
else:
    print("=== Detection Performance Metrics ===")
    print("Precision: 0.0000")
    print("Recall: 0.0000")
    print("F1-Score: 0.0000")
    print("No labels available for metric calculation")

# #8. Integrity Check
num_distortion_clusters = 0
for root, proc_list in clusters.items():
    candidate_originals = set(chain.from_iterable(processed_to_originals[proc] for proc in proc_list))
    if len(candidate_originals) >= 2:
        num_distortion_clusters += 1
total_distorted = (df['is_distorted'] == 1).sum()
activities_fixed = total_distorted
clean_unchanged = len(df) - total_distorted
print(f"Total distortion clusters detected: {num_distortion_clusters}")
print(f"Total activities marked as distorted: {total_distorted}")
print(f"Activities to be fixed: {activities_fixed}")

# #9. Fix Activities
df['Activity_fixed'] = df['canonical_activity']

# #10. Save Output
# Standardize timestamp
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')

# Prepare output columns
output_cols = ['Case', 'Timestamp', 'Activity', 'Activity_fixed']
output_df = df[output_cols].copy()

if 'Variant' in df.columns:
    output_df.insert(2, 'Variant', df['Variant'])
if 'Resource' in df.columns:
    insert_pos = 3 if 'Variant' in df.columns else 2
    output_df.insert(insert_pos, 'Resource', df['Resource'])
if has_label:
    output_df['label'] = df['label']

output_df.to_csv(output_file, index=False)

# #11. Summary Statistics
total_events = len(df)
num_distorted_detected = total_distorted
unique_before = len(df['Activity'].unique())
unique_after = len(output_df['Activity_fixed'].unique())
reduction = unique_before - unique_after
percentage = (reduction / unique_before * 100) if unique_before > 0 else 0.0

print(f"Total number of events: {total_events}")
print(f"Number of distorted events detected: {num_distorted_detected}")
print(f"Unique activities before fixing: {unique_before}")
print(f"Unique activities after fixing: {unique_after}")
print(f"Activity reduction count and percentage: {reduction} ({percentage:.2f}%)")
print(f"Output file path: {output_file}")

# Sample transformations
transform_samples = {orig: can for orig, can in mapping.items() if orig != can}
sample_list = list(transform_samples.items())[:10]
print("Sample of up to 10 transformations showing: original → canonical")
for orig, can in sample_list:
    print(f"- {orig} → {can}")

print(f"Run 3: Processed dataset saved to: {output_file}")
print(f"Run 3: Final dataset shape: {output_df.shape}")
print(f"Run 3: Dataset: pub")
print(f"Run 3: Task type: distorted")