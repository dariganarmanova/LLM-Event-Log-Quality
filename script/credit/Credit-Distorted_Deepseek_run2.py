# Generated script for Credit-Distorted - Run 2
# Generated on: 2025-11-13T15:14:04.666239
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

# Load data
input_file = 'data/credit/Credit-Distorted.csv'
df = pd.read_csv(input_file)

# Rename columns if needed
column_mapping = {'CaseID': 'Case', 'ActivityName': 'Activity', 'Timestamp': 'Timestamp'}
df = df.rename(columns=column_mapping)

# Store original activity
df['original_activity'] = df['Activity']

# Step 2: Identify distorted activities
df['isdistorted'] = df['Activity'].str.endswith(distorted_suffix).astype(int)
df['BaseActivity'] = df['Activity'].str.replace(distorted_suffix, '', regex=False)

# Step 3: Preprocess activity names
def preprocess_activity(activity):
    activity = str(activity).lower() if not case_sensitive else str(activity)
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

# Find similar pairs
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

# Initialize parent
for activity in unique_activities:
    parent[activity] = activity

# Union similar pairs
for a1, a2 in similar_pairs:
    union(a1, a2)

# Get clusters
clusters = defaultdict(list)
for activity in unique_activities:
    clusters[find(activity)].append(activity)

# Filter clusters with at least 2 members
clusters = {k: v for k, v in clusters.items() if len(v) >= 2}

# Step 6: Majority voting
activity_mapping = {}
for cluster in clusters.values():
    original_activities = []
    for processed_act in cluster:
        original_acts = valid_activities[valid_activities['ProcessedActivity'] == processed_act]['BaseActivity'].unique()
        original_activities.extend(original_acts)
    
    # Count frequencies
    freq = pd.Series(original_activities).value_counts()
    canonical = freq.idxmax()
    
    for processed_act in cluster:
        variants = valid_activities[valid_activities['ProcessedActivity'] == processed_act]['BaseActivity'].unique()
        for variant in variants:
            activity_mapping[variant] = canonical

# Apply mapping
df['canonical_activity'] = df['BaseActivity'].map(activity_mapping).fillna(df['BaseActivity'])
df['is_distorted'] = (df['BaseActivity'] != df['canonical_activity']).astype(int)

# Step 7: Detection metrics
if 'label' in df.columns:
    def normalize_label(label):
        if pd.isna(label):
            return 0
        return 1 if 'distorted' in str(label).lower() else 0
    
    y_true = df['label'].apply(normalize_label)
    y_pred = df['is_distorted']
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print('=== Detection Performance Metrics ===')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-Score: {f1:.4f}')
    print('✓ Precision threshold (≥ 0.6) met' if precision >= 0.6 else '✗ Precision threshold (≥ 0.6) not met')
else:
    print('=== Detection Performance Metrics ===')
    print('Precision: 0.0000')
    print('Recall: 0.0000')
    print('F1-Score: 0.0000')
    print('No labels available for metric calculation')

# Step 8: Integrity check
total_clusters = len(clusters)
total_distorted = df['is_distorted'].sum()
total_clean = len(df) - total_distorted

print('\n=== Integrity Check ===')
print(f'Total distortion clusters detected: {total_clusters}')
print(f'Total activities marked as distorted: {total_distorted}')
print(f'Clean activities (not modified): {total_clean}')

# Step 9: Fix activities
df['Activity_fixed'] = df['canonical_activity']

# Step 10: Save output
output_cols = ['Case', 'Timestamp', 'Activity', 'Activity_fixed']
if 'label' in df.columns:
    output_cols.append('label')
if 'Variant' in df.columns:
    output_cols.append('Variant')

output_df = df[output_cols]
output_file = 'data/credit/credit_distorted_cleaned_run2.csv'
output_df.to_csv(output_file, index=False)

# Step 11: Summary statistics
unique_before = df['Activity'].nunique()
unique_after = df['Activity_fixed'].nunique()
reduction = unique_before - unique_after
reduction_pct = (reduction / unique_before) * 100

print('\n=== Summary Statistics ===')
print(f'Run 2: Original dataset shape: {df.shape}')
print(f'Total number of events: {len(df)}')
print(f'Number of distorted events detected: {total_distorted}')
print(f'Unique activities before fixing: {unique_before}')
print(f'Unique activities after fixing: {unique_after}')
print(f'Activity reduction: {reduction} ({reduction_pct:.2f}%)')
print(f'Run 2: Processed dataset saved to: {output_file}')
print(f'Run 2: Final dataset shape: {output_df.shape}')
print(f'Run 2: Dataset: credit')
print(f'Run 2: Task type: distorted')

# Print sample transformations
sample_size = min(10, len(activity_mapping))
sample = pd.Series(activity_mapping).head(sample_size)
print('\nSample transformations (original → canonical):')
for orig, canon in sample.items():
    print(f'{orig} → {canon}')