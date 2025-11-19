# Generated script for BPIC15-Distorted - Run 2
# Generated on: 2025-11-13T14:38:04.921423
# Model: deepseek-ai/DeepSeek-V3-0324

import pandas as pd
import re
from collections import defaultdict
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
ngram_size = 3
min_length = 4
distorted_suffix = ':distorted'

# Load the data
df = pd.read_csv('data/bpic15/BPIC15-Distorted.csv')
print(f"Run 2: Original dataset shape: {df.shape}")

# Step 1: Load CSV and prepare columns
df['original_activity'] = df['Activity']
required_columns = ['Case', 'Activity', 'Timestamp']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Required column {col} not found in the input file")

# Step 2: Identify Distorted Activities
df['isdistorted'] = df['Activity'].str.endswith(distorted_suffix).astype(int)
df['BaseActivity'] = df['Activity'].str.replace(distorted_suffix, '', regex=False)

# Step 3: Preprocess Activity Names
def preprocess_activity(activity):
    activity = str(activity)
    if not case_sensitive:
        activity = activity.lower()
    activity = re.sub(r'[^a-zA-Z0-9\s]', '', activity)
    activity = re.sub(r'\s+', ' ', activity).strip()
    return activity

df['ProcessedActivity'] = df['BaseActivity'].apply(preprocess_activity)
df = df[df['ProcessedActivity'].str.len() >= min_length].copy()

# Step 4: Calculate Jaccard N-gram Similarity
def get_ngrams(text, n):
    return set([text[i:i+n] for i in range(len(text)-n+1)])

def jaccard_similarity(a, b, n):
    ngrams_a = get_ngrams(a, n)
    ngrams_b = get_ngrams(b, n)
    if not ngrams_a and not ngrams_b:
        return 1.0
    intersection = len(ngrams_a & ngrams_b)
    union = len(ngrams_a | ngrams_b)
    return intersection / union

unique_activities = df['ProcessedActivity'].unique()
similar_pairs = []

for i in range(len(unique_activities)):
    for j in range(i+1, len(unique_activities)):
        a = unique_activities[i]
        b = unique_activities[j]
        sim = jaccard_similarity(a, b, ngram_size)
        if sim >= similarity_threshold:
            similar_pairs.append((a, b))

# Step 5: Cluster Similar Activities (Union-Find)
parent = {}

def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(x, y):
    x_root = find(x)
    y_root = find(y)
    if x_root != y_root:
        parent[y_root] = x_root

for activity in unique_activities:
    parent[activity] = activity

for a, b in similar_pairs:
    union(a, b)

clusters = defaultdict(list)
for activity in unique_activities:
    clusters[find(activity)].append(activity)

clusters = {k: v for k, v in clusters.items() if len(v) >= 2}

# Step 6: Majority Voting Within Clusters
activity_to_canonical = {}
for cluster_root, cluster_activities in clusters.items():
    original_activities = []
    for processed_act in cluster_activities:
        original_acts = df[df['ProcessedActivity'] == processed_act]['BaseActivity'].unique()
        original_activities.extend(original_acts)
    
    freq = pd.Series(original_activities).value_counts()
    canonical = freq.idxmax()
    
    for processed_act in cluster_activities:
        variants = df[df['ProcessedActivity'] == processed_act]['BaseActivity'].unique()
        for variant in variants:
            activity_to_canonical[variant] = canonical

df['canonical_activity'] = df['BaseActivity'].map(activity_to_canonical).fillna(df['BaseActivity'])
df['is_distorted'] = (df['BaseActivity'] != df['canonical_activity']).astype(int)

# Step 7: Calculate Detection Metrics (BEFORE FIXING)
if 'label' in df.columns:
    def normalize_label(label):
        if pd.isna(label):
            return 0
        label = str(label).lower()
        return 1 if 'distorted' in label else 0
    
    y_true = df['label'].apply(normalize_label)
    y_pred = df['is_distorted']
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print("=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("✓ Precision threshold (≥ 0.6) met" if precision >= 0.6 else "✗ Precision threshold (≥ 0.6) not met")
else:
    print("=== Detection Performance Metrics ===")
    print("Precision: 0.0000")
    print("Recall: 0.0000")
    print("F1-Score: 0.0000")
    print("No labels available for metric calculation")

# Step 8: Integrity Check
total_clusters = len(clusters)
total_distorted = df['is_distorted'].sum()
total_clean = len(df) - total_distorted

print("\n=== Integrity Check ===")
print(f"Total distortion clusters detected: {total_clusters}")
print(f"Total activities marked as distorted: {total_distorted}")
print(f"Activities to be fixed: {total_distorted}")

# Step 9: Fix Activities
df['Activity_fixed'] = df['canonical_activity']

# Step 10: Save Output
output_columns = ['Case', 'Timestamp', 'Activity', 'Activity_fixed']
if 'label' in df.columns:
    output_columns.append('label')
    
df[output_columns].to_csv('data/bpic15/bpic15_distorted_cleaned_run2.csv', index=False)

# Step 11: Summary Statistics
unique_before = df['Activity'].nunique()
unique_after = df['Activity_fixed'].nunique()
reduction = unique_before - unique_after
reduction_pct = (reduction / unique_before) * 100

print("\n=== Summary Statistics ===")
print(f"Total number of events: {len(df)}")
print(f"Number of distorted events detected: {total_distorted}")
print(f"Unique activities before fixing: {unique_before}")
print(f"Unique activities after fixing: {unique_after}")
print(f"Activity reduction count: {reduction} ({reduction_pct:.2f}%)")
print(f"Output file path: data/bpic15/bpic15_distorted_cleaned_run2.csv")

sample_transforms = df[df['is_distorted'] == 1][['Activity', 'Activity_fixed']].drop_duplicates().head(10)
if not sample_transforms.empty:
    print("\nSample transformations (original → canonical):")
    for _, row in sample_transforms.iterrows():
        print(f"{row['Activity']} → {row['Activity_fixed']}")

print(f"\nRun 2: Processed dataset saved to: data/bpic15/bpic15_distorted_cleaned_run2.csv")
print(f"Run 2: Final dataset shape: {df.shape}")
print(f"Run 2: Dataset: bpic15")
print(f"Run 2: Task type: distorted")