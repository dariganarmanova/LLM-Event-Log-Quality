# Generated script for BPIC11-Distorted - Run 3
# Generated on: 2025-11-18T22:22:11.815311
# Model: grok-4-fast

import pandas as pd
import re
from collections import defaultdict, Counter
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration
input_file = 'data/bpic11/BPIC11-Distorted.csv'
output_file = 'data/bpic11/bpic11_distorted_cleaned_run3.csv'
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
print(f"Run 3: Original dataset shape: {df.shape}")

# Normalize column names if needed (e.g., CaseID to Case)
if 'Case ID' in df.columns:
    df['Case'] = df.pop('Case ID')

# Store original Activity
df['original_activity'] = df[activity_column].copy()

# #2. Identify Distorted Activities
df['isdistorted'] = df[activity_column].str.endswith(distorted_suffix, na=False).astype(int)
df['BaseActivity'] = df.apply(lambda row: row[activity_column][:-len(distorted_suffix)] if row['isdistorted'] == 1 else row[activity_column], axis=1)

# #3. Preprocess Activity Names
def preprocess(text):
    if pd.isna(text):
        return ''
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df['ProcessedActivity'] = df['BaseActivity'].apply(preprocess)

# #4. Calculate Jaccard N-gram Similarity
def generate_ngrams(text, n=ngram_size):
    if len(text) < n:
        return set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}

def jaccard_sim(text1, text2):
    set1 = generate_ngrams(text1)
    set2 = generate_ngrams(text2)
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0

# Build mappings
activity_to_base = {}
for act in df[activity_column].unique():
    if act in df[activity_column].values:
        is_d = df.loc[df[activity_column] == act, 'isdistorted'].iloc[0]
        base = act[:-len(distorted_suffix)] if is_d == 1 else act
        activity_to_base[act] = base

processed_dict = {act: preprocess(base) for act, base in activity_to_base.items()}
proc_to_acts = defaultdict(list)
for act, proc in processed_dict.items():
    if len(proc) >= min_length:
        proc_to_acts[proc].append(act)

unique_procs = list(proc_to_acts.keys())

# Find similar pairs
similar_pairs = []
for i in range(len(unique_procs)):
    for j in range(i + 1, len(unique_procs)):
        proc1 = unique_procs[i]
        proc2 = unique_procs[j]
        sim = jaccard_sim(proc1, proc2)
        if sim >= similarity_threshold:
            similar_pairs.append((proc1, proc2))
        # Progress every 100 comparisons
        if (i * (i - 1) // 2 + j - i) % 100 == 0:
            print(f"Progress: {i * (i - 1) // 2 + j - i} comparisons processed")

# #5. Cluster Similar Activities (Union-Find)
parent = {p: p for p in unique_procs}
rank = {p: 0 for p in unique_procs}

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

for p1, p2 in similar_pairs:
    union(p1, p2)

clusters = defaultdict(list)
for p in unique_procs:
    clusters[find(p)].append(p)

valid_clusters = {k: v for k, v in clusters.items() if len(v) >= 2}

# #6. Majority Voting Within Clusters
canonical_mapping = {act: act for act in df[activity_column].unique()}
for root, cluster_procs in valid_clusters.items():
    cluster_acts = []
    for p in cluster_procs:
        cluster_acts.extend(proc_to_acts[p])
    if not cluster_acts:
        continue
    act_counts = Counter({act: len(df[df[activity_column] == act]) for act in cluster_acts})
    if act_counts:
        canonical = max(act_counts, key=act_counts.get)
        for act in cluster_acts:
            canonical_mapping[act] = canonical

df['canonical_activity'] = df[activity_column].map(canonical_mapping)
df['is_distorted'] = (df[activity_column] != df['canonical_activity']).astype(int)

# #7. Calculate Detection Metrics (BEFORE FIXING)
print("=== Detection Performance Metrics ===")
if label_column not in df.columns:
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
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    status = "✓" if prec >= 0.6 else "✗"
    print(f"{status} Precision threshold (≥ 0.6) met/not met")

# #8. Integrity Check
num_clusters = len(valid_clusters)
num_distorted = int(df['is_distorted'].sum())
num_clean = len(df) - num_distorted
print(f"Total distortion clusters detected: {num_clusters}")
print(f"Total activities marked as distorted: {num_distorted}")
print(f"Activities to be fixed: {num_distorted}")
print(f"Clean activities that were NOT modified: {num_clean}")
clean_changed = ((df['is_distorted'] == 0) & (df[activity_column] != df['canonical_activity'])).sum()
if clean_changed > 0:
    print(f"WARNING: {clean_changed} clean activities were modified!")
else:
    print("✓ All clean activities unchanged")

# #9. Fix Activities
df['Activity_fixed'] = df['canonical_activity']

# #10. Save Output
# Format timestamp
df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')

# Select columns
columns_to_save = [case_column, timestamp_column, activity_column, 'Activity_fixed']
if 'Variant' in df.columns:
    columns_to_save.insert(2, 'Variant')
if 'Resource' in df.columns:
    insert_pos = 3 if 'Variant' in df.columns else 2
    columns_to_save.insert(insert_pos, 'Resource')
if label_column in df.columns:
    columns_to_save.append(label_column)

df_out = df[columns_to_save].copy()

# #11. Summary Statistics
total_events = len(df)
distorted_detected = num_distorted
unique_before = len(df[activity_column].unique())
unique_after = len(df['Activity_fixed'].unique())
reduction = unique_before - unique_after
percentage = (reduction / unique_before * 100) if unique_before > 0 else 0.0
print(f"Total number of events: {total_events}")
print(f"Number of distorted events detected: {distorted_detected}")
print(f"Unique activities before fixing: {unique_before}")
print(f"Unique activities after fixing: {unique_after}")
print(f"Activity reduction count and percentage: {reduction} ({percentage:.2f}%)")
print(f"Output file path: {output_file}")
print("Sample of up to 10 transformations showing: original → canonical")
changes = [(k, v) for k, v in canonical_mapping.items() if k != v]
for i in range(min(10, len(changes))):
    orig, can = changes[i]
    print(f"- {orig} → {can}")
if len(changes) > 10:
    print(f"... and {len(changes) - 10} more")

# Save
df_out.to_csv(output_file, index=False)
print(f"Run 3: Processed dataset saved to: {output_file}")
print(f"Run 3: Final dataset shape: {df_out.shape}")
print(f"Run 3: Dataset: bpic11")
print(f"Run 3: Task type: distorted")