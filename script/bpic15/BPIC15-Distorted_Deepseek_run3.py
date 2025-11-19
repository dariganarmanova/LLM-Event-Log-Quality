# Generated script for BPIC15-Distorted - Run 3
# Generated on: 2025-11-13T14:38:58.369132
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
input_file = 'data/bpic15/BPIC15-Distorted.csv'
output_file = 'data/bpic15/bpic15_distorted_cleaned_run3.csv'
df = pd.read_csv(input_file)
print(f"Run 3: Original dataset shape: {df.shape}")

# Ensure required columns exist
required_columns = ['Case', 'Activity', 'Timestamp']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' not found in the input file.")

# Store original activity
df['original_activity'] = df['Activity']

# Step 2: Identify Distorted Activities
df['isdistorted'] = df['Activity'].str.endswith(distorted_suffix).astype(int)
df['BaseActivity'] = df['Activity'].str.replace(re.escape(distorted_suffix), '', regex=True)

# Step 3: Preprocess Activity Names
def preprocess_activity(activity):
    if not case_sensitive:
        activity = activity.lower()
    activity = re.sub(r'[^a-zA-Z0-9\s]', '', activity)
    activity = re.sub(r'\s+', ' ', activity).strip()
    return activity

df['ProcessedActivity'] = df['BaseActivity'].apply(preprocess_activity)
df = df[df['ProcessedActivity'].str.len() >= min_length]

# Step 4: Calculate Jaccard N-gram Similarity
def generate_ngrams(text, n):
    return set([text[i:i+n] for i in range(len(text)-n+1)])

def jaccard_similarity(a, b):
    set_a = generate_ngrams(a, ngram_size)
    set_b = generate_ngrams(b, ngram_size)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union != 0 else 0.0

unique_activities = df['ProcessedActivity'].unique()
similar_pairs = []

for i in range(len(unique_activities)):
    for j in range(i+1, len(unique_activities)):
        sim = jaccard_similarity(unique_activities[i], unique_activities[j])
        if sim >= similarity_threshold:
            similar_pairs.append((unique_activities[i], unique_activities[j]))

# Step 5: Cluster Similar Activities (Union-Find)
parent = {}

def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(x, y):
    root_x = find(x)
    root_y = find(y)
    if root_x != root_y:
        parent[root_y] = root_x

for activity in unique_activities:
    parent[activity] = activity

for a1, a2 in similar_pairs:
    union(a1, a2)

clusters = defaultdict(list)
for activity in unique_activities:
    root = find(activity)
    clusters[root].append(activity)

clusters = {k: v for k, v in clusters.items() if len(v) >= 2}

# Step 6: Majority Voting Within Clusters
canonical_mapping = {}

for cluster in clusters.values():
    original_activities = []
    for processed_activity in cluster:
        mask = df['ProcessedActivity'] == processed_activity
        original_activities.extend(df.loc[mask, 'BaseActivity'].tolist())
    
    freq = pd.Series(original_activities).value_counts()
    canonical = freq.idxmax()
    
    for processed_activity in cluster:
        mask = df['ProcessedActivity'] == processed_activity
        df.loc[mask, 'canonical_activity'] = canonical
        df.loc[mask, 'isdistorted'] = (df.loc[mask, 'BaseActivity'] != canonical).astype(int)

# Fill canonical_activity for non-clustered activities
mask = df['canonical_activity'].isna()
df.loc[mask, 'canonical_activity'] = df.loc[mask, 'BaseActivity']
df.loc[mask, 'isdistorted'] = 0

# Step 7: Calculate Detection Metrics (BEFORE FIXING)
if 'label' in df.columns:
    def normalize_label(label):
        if pd.isna(label):
            return 0
        return 1 if 'distorted' in str(label).lower() else 0
    
    y_true = df['label'].apply(normalize_label)
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
    print("No labels available for metric calculation")
    print("Precision: 0.0000")
    print("Recall: 0.0000")
    print("F1-Score: 0.0000")

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
output_columns = ['Case', 'Timestamp', 'Activity', 'Activity_fixed']
if 'label' in df.columns:
    output_columns.append('label')
df_output = df[output_columns]
df_output.to_csv(output_file, index=False)

# Step 11: Summary Statistics
unique_before = df['Activity'].nunique()
unique_after = df['Activity_fixed'].nunique()
reduction = unique_before - unique_after
reduction_pct = (reduction / unique_before) * 100 if unique_before != 0 else 0

print("\n=== Summary Statistics ===")
print(f"Total number of events: {len(df)}")
print(f"Number of distorted events detected: {total_distorted}")
print(f"Unique activities before fixing: {unique_before}")
print(f"Unique activities after fixing: {unique_after}")
print(f"Activity reduction count: {reduction} ({reduction_pct:.2f}%)")
print(f"Output file path: {output_file}")

# Print sample transformations
sample_size = min(10, len(df))
sample = df.sample(sample_size)[['Activity', 'Activity_fixed']].drop_duplicates()
print("\nSample transformations (original → canonical):")
for _, row in sample.iterrows():
    print(f"{row['Activity']} → {row['Activity_fixed']}")

print(f"\nRun 3: Processed dataset saved to: {output_file}")
print(f"Run 3: Final dataset shape: {df.shape}")
print(f"Run 3: Dataset: bpic15")
print(f"Run 3: Task type: distorted")