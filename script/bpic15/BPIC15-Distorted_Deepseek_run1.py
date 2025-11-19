# Generated script for BPIC15-Distorted - Run 1
# Generated on: 2025-11-13T14:37:14.866724
# Model: deepseek-ai/DeepSeek-V3-0324

import pandas as pd
import re
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration parameters
input_file = 'data/bpic15/BPIC15-Distorted.csv'
output_file = 'data/bpic15/bpic15_distorted_cleaned_run1.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
distorted_suffix = ':distorted'
ngram_size = 3
similarity_threshold = 0.56
min_length = 4

# Load the data
df = pd.read_csv(input_file)
print(f"Run 1: Original dataset shape: {df.shape}")

# Step 1: Store original activities
df['original_activity'] = df[activity_column]

# Step 2: Identify distorted activities
df['isdistorted'] = df[activity_column].str.endswith(distorted_suffix).astype(int)
df['BaseActivity'] = df[activity_column].replace(f'{distorted_suffix}$', '', regex=True)

# Step 3: Preprocess activity names
def preprocess_activity(activity):
    activity = str(activity).lower()
    activity = re.sub(r'[^a-z0-9 ]', '', activity)
    activity = re.sub(r'\s+', ' ', activity).strip()
    return activity

df['ProcessedActivity'] = df[activity_column].apply(preprocess_activity)

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

unique_activities = df[df['ProcessedActivity'].str.len() >= min_length]['ProcessedActivity'].unique()
similar_pairs = []

for i in range(len(unique_activities)):
    for j in range(i+1, len(unique_activities)):
        sim = jaccard_similarity(unique_activities[i], unique_activities[j], ngram_size)
        if sim >= similarity_threshold:
            similar_pairs.append((unique_activities[i], unique_activities[j]))

# Step 5: Cluster Similar Activities (Union-Find)
parent = {}

def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(x, y):
    parent[find(x)] = find(y)

for activity in unique_activities:
    parent[activity] = activity

for a1, a2 in similar_pairs:
    union(a1, a2)

clusters = defaultdict(list)
for activity in unique_activities:
    clusters[find(activity)].append(activity)

clusters = {k: v for k, v in clusters.items() if len(v) >= 2}

# Step 6: Majority Voting Within Clusters
activity_to_canonical = {}
for cluster in clusters.values():
    original_activities_in_cluster = df[df['ProcessedActivity'].isin(cluster)][activity_column]
    most_common = original_activities_in_cluster.value_counts().index[0]
    for activity in cluster:
        activity_to_canonical[activity] = most_common

df['canonical_activity'] = df[activity_column]
df.loc[df['ProcessedActivity'].isin(activity_to_canonical), 'canonical_activity'] = df['ProcessedActivity'].map(activity_to_canonical)
df['is_distorted'] = (df[activity_column] != df['canonical_activity']).astype(int)

# Step 7: Calculate Detection Metrics
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

# Step 8: Integrity Check
print("\n=== Integrity Check ===")
print(f"Total distortion clusters detected: {len(clusters)}")
print(f"Total activities marked as distorted: {df['is_distorted'].sum()}")
print(f"Activities to be fixed: {df['is_distorted'].sum()}")

# Step 9: Fix Activities
df['Activity_fixed'] = df['canonical_activity']

# Step 10: Save Output
output_columns = [case_column, timestamp_column, activity_column, 'Activity_fixed']
if label_column in df.columns:
    output_columns.append(label_column)
df[output_columns].to_csv(output_file, index=False)

# Step 11: Summary Statistics
print("\n=== Summary Statistics ===")
print(f"Total number of events: {len(df)}")
print(f"Number of distorted events detected: {df['is_distorted'].sum()}")
print(f"Unique activities before fixing: {df[activity_column].nunique()}")
print(f"Unique activities after fixing: {df['Activity_fixed'].nunique()}")
reduction = df[activity_column].nunique() - df['Activity_fixed'].nunique()
reduction_pct = (reduction / df[activity_column].nunique()) * 100
print(f"Activity reduction count: {reduction} ({reduction_pct:.2f}%)")
print(f"Output file path: {output_file}")

# Print sample transformations
sample_transformations = df[df['is_distorted'] == 1][[activity_column, 'Activity_fixed']].drop_duplicates().head(10)
print("\nSample transformations (original → canonical):")
for _, row in sample_transformations.iterrows():
    print(f"{row[activity_column]} → {row['Activity_fixed']}")

print(f"\nRun 1: Processed dataset saved to: {output_file}")
print(f"Run 1: Final dataset shape: {df.shape}")
print(f"Run 1: Dataset: bpic15")
print(f"Run 1: Task type: distorted")