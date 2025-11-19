# Generated script for Credit-Distorted - Run 1
# Generated on: 2025-11-13T15:13:07.730416
# Model: deepseek-ai/DeepSeek-V3-0324

import pandas as pd
import re
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration parameters
input_file = 'data/credit/Credit-Distorted.csv'
output_file = 'data/credit/credit_distorted_cleaned_run1.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
distorted_suffix = ':distorted'
ngram_size = 3
similarity_threshold = 0.56
min_length = 4

# Step 1: Load CSV
df = pd.read_csv(input_file)
df['original_activity'] = df[activity_column].copy()

# Step 2: Identify Distorted Activities
df['isdistorted'] = df[activity_column].str.endswith(distorted_suffix).astype(int)
df['BaseActivity'] = df[activity_column].str.replace(distorted_suffix, '', regex=False)

# Step 3: Preprocess Activity Names
def preprocess_activity(activity):
    activity = str(activity).lower()
    activity = re.sub(r'[^a-z0-9\s]', '', activity)
    activity = re.sub(r'\s+', ' ', activity).strip()
    return activity

df['ProcessedActivity'] = df['BaseActivity'].apply(preprocess_activity)
valid_activities = df[df['ProcessedActivity'].str.len() >= min_length]

# Step 4: Calculate Jaccard N-gram Similarity
def get_ngrams(text, n):
    return set([text[i:i+n] for i in range(len(text)-n+1)])

def jaccard_similarity(a, b):
    a_ngrams = get_ngrams(a, ngram_size)
    b_ngrams = get_ngrams(b, ngram_size)
    if not a_ngrams or not b_ngrams:
        return 0.0
    intersection = len(a_ngrams & b_ngrams)
    union = len(a_ngrams | b_ngrams)
    return intersection / union

unique_activities = valid_activities['ProcessedActivity'].unique()
similar_pairs = []

for i in range(len(unique_activities)):
    for j in range(i+1, len(unique_activities)):
        sim = jaccard_similarity(unique_activities[i], unique_activities[j])
        if sim >= similarity_threshold:
            similar_pairs.append((unique_activities[i], unique_activities[j]))

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

for a1, a2 in similar_pairs:
    union(a1, a2)

clusters = defaultdict(list)
for activity in unique_activities:
    clusters[find(activity)].append(activity)

clusters = {k: v for k, v in clusters.items() if len(v) >= 2}

# Step 6: Majority Voting Within Clusters
activity_mapping = {}

for cluster in clusters.values():
    original_activities = []
    for processed_activity in cluster:
        mask = df['ProcessedActivity'] == processed_activity
        original_activities.extend(df.loc[mask, 'original_activity'].tolist())
    
    if not original_activities:
        continue
    
    freq = pd.Series(original_activities).value_counts()
    canonical = freq.idxmax()
    
    for processed_activity in cluster:
        mask = df['ProcessedActivity'] == processed_activity
        df.loc[mask, 'canonical_activity'] = canonical
        df.loc[mask, 'isdistorted'] = (df.loc[mask, 'original_activity'] != canonical).astype(int)

df['canonical_activity'] = df['canonical_activity'].fillna(df['original_activity'])

# Step 7: Calculate Detection Metrics (BEFORE FIXING)
if label_column in df.columns:
    def normalize_label(label):
        if pd.isna(label):
            return 0
        return 1 if 'distorted' in str(label).lower() else 0
    
    y_true = df[label_column].apply(normalize_label)
    y_pred = df['isdistorted']
    
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
total_distorted = df['isdistorted'].sum()
total_clean = len(df) - total_distorted

print("\n=== Integrity Check ===")
print(f"Total distortion clusters detected: {total_clusters}")
print(f"Total activities marked as distorted: {total_distorted}")
print(f"Activities to be fixed: {total_distorted}")

# Step 9: Fix Activities
df['Activity_fixed'] = df['canonical_activity']

# Step 10: Save Output
output_columns = [case_column, timestamp_column, activity_column, 'Activity_fixed']
if label_column in df.columns:
    output_columns.append(label_column)

df[output_columns].to_csv(output_file, index=False)

# Step 11: Summary Statistics
unique_before = df[activity_column].nunique()
unique_after = df['Activity_fixed'].nunique()
reduction = unique_before - unique_after
reduction_pct = (reduction / unique_before) * 100

print("\n=== Summary Statistics ===")
print(f"Run 1: Original dataset shape: {df.shape}")
print(f"Total number of events: {len(df)}")
print(f"Number of distorted events detected: {total_distorted}")
print(f"Unique activities before fixing: {unique_before}")
print(f"Unique activities after fixing: {unique_after}")
print(f"Activity reduction count: {reduction} ({reduction_pct:.2f}%)")
print(f"Run 1: Processed dataset saved to: {output_file}")
print(f"Run 1: Final dataset shape: {df.shape}")
print(f"Run 1: Dataset: credit")
print(f"Run 1: Task type: distorted")

sample_transforms = df[df['isdistorted'] == 1].head(10)[[activity_column, 'Activity_fixed']].drop_duplicates()
if not sample_transforms.empty:
    print("\nSample transformations (original → canonical):")
    for _, row in sample_transforms.iterrows():
        print(f"{row[activity_column]} → {row['Activity_fixed']}")