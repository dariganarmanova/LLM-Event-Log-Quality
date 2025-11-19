# Generated script for BPIC11-Distorted - Run 3
# Generated on: 2025-11-13T11:24:33.525024
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
input_file = 'data/bpic11/BPIC11-Distorted.csv'
output_file = 'data/bpic11/bpic11_distorted_cleaned_run3.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'

df = pd.read_csv(input_file)
print(f"Run 3: Original dataset shape: {df.shape}")

# Step 1: Store original activities
df['original_activity'] = df[activity_column]

# Step 2: Identify distorted activities
df['isdistorted'] = df[activity_column].str.endswith(distorted_suffix).astype(int)
df['BaseActivity'] = df[activity_column].replace(f'{distorted_suffix}$', '', regex=True)

# Step 3: Preprocess activity names
def preprocess_activity(activity):
    if not isinstance(activity, str):
        return ''
    activity = activity.lower() if not case_sensitive else activity
    activity = re.sub(r'[^a-zA-Z0-9\s]', '', activity)
    activity = re.sub(r'\s+', ' ', activity).strip()
    return activity

df['ProcessedActivity'] = df['BaseActivity'].apply(preprocess_activity)
valid_activities = df[df['ProcessedActivity'].str.len() >= min_length]

# Step 4: Jaccard similarity functions
def get_ngrams(text, n):
    return set([text[i:i+n] for i in range(len(text)-n+1)])

def jaccard_similarity(a, b, n):
    ngrams_a = get_ngrams(a, n)
    ngrams_b = get_ngrams(b, n)
    if not ngrams_a or not ngrams_b:
        return 0.0
    intersection = len(ngrams_a & ngrams_b)
    union = len(ngrams_a | ngrams_b)
    return intersection / union

unique_activities = valid_activities['ProcessedActivity'].unique()
similar_pairs = []

for i in range(len(unique_activities)):
    for j in range(i+1, len(unique_activities)):
        sim = jaccard_similarity(unique_activities[i], unique_activities[j], ngram_size)
        if sim >= similarity_threshold:
            similar_pairs.append((unique_activities[i], unique_activities[j]))

# Step 5: Union-Find clustering
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

for a1, a2 in similar_pairs:
    union(a1, a2)

clusters = defaultdict(list)
for activity in unique_activities:
    clusters[find(activity)].append(activity)

clusters = {k: v for k, v in clusters.items() if len(v) >= 2}

# Step 6: Majority voting for canonical forms
activity_to_canonical = {}
original_activity_counts = valid_activities.groupby('ProcessedActivity')[activity_column].value_counts()

for cluster_root, cluster_members in clusters.items():
    cluster_activities = []
    for member in cluster_members:
        original_forms = valid_activities[valid_activities['ProcessedActivity'] == member][activity_column].unique()
        cluster_activities.extend(original_forms)
    
    if not cluster_activities:
        continue
    
    most_common = max(cluster_activities, key=lambda x: original_activity_counts[cluster_root][x])
    for member in cluster_members:
        member_original_forms = valid_activities[valid_activities['ProcessedActivity'] == member][activity_column].unique()
        for form in member_original_forms:
            activity_to_canonical[form] = most_common

# Apply canonical forms
df['canonical_activity'] = df[activity_column]
df['is_distorted'] = 0

for variant, canonical in activity_to_canonical.items():
    mask = df[activity_column] == variant
    df.loc[mask, 'canonical_activity'] = canonical
    df.loc[mask, 'is_distorted'] = 1

# Step 7: Detection metrics
if label_column in df.columns:
    def normalize_label(label):
        if pd.isna(label):
            return 0
        return 1 if 'distorted' in str(label).lower() else 0
    
    y_true = df[label_column].apply(normalize_label)
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
    print("No labels available for metric calculation")
    print("Precision: 0.0000")
    print("Recall: 0.0000")
    print("F1-Score: 0.0000")

# Step 8: Integrity check
total_clusters = len(clusters)
total_distorted = df['is_distorted'].sum()
total_clean = len(df) - total_distorted

print("\n=== Integrity Check ===")
print(f"Total distortion clusters detected: {total_clusters}")
print(f"Total activities marked as distorted: {total_distorted}")
print(f"Activities to be fixed: {total_distorted}")

# Step 9: Fix activities
df['Activity_fixed'] = df['canonical_activity']

# Step 10: Save output
output_columns = [case_column, timestamp_column, activity_column, 'Activity_fixed']
if label_column in df.columns:
    output_columns.append(label_column)

df[output_columns].to_csv(output_file, index=False)

# Step 11: Summary statistics
unique_before = df[activity_column].nunique()
unique_after = df['Activity_fixed'].nunique()
reduction = unique_before - unique_after
reduction_pct = (reduction / unique_before) * 100

print("\n=== Summary Statistics ===")
print(f"Total number of events: {len(df)}")
print(f"Number of distorted events detected: {total_distorted}")
print(f"Unique activities before fixing: {unique_before}")
print(f"Unique activities after fixing: {unique_after}")
print(f"Activity reduction: {reduction} ({reduction_pct:.2f}%)")
print(f"Output file path: {output_file}")

sample_transforms = df[df['is_distorted'] == 1].head(10)[[activity_column, 'Activity_fixed']].drop_duplicates()
print("\nSample transformations (original → canonical):")
for _, row in sample_transforms.iterrows():
    print(f"{row[activity_column]} → {row['Activity_fixed']}")

print(f"\nRun 3: Processed dataset saved to: {output_file}")
print(f"Run 3: Final dataset shape: {df.shape}")
print(f"Run 3: Dataset: bpic11")
print(f"Run 3: Task type: distorted")