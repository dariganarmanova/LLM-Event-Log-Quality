# Generated script for Credit-Distorted - Run 3
# Generated on: 2025-11-13T15:15:00.327006
# Model: deepseek-ai/DeepSeek-V3-0324

import pandas as pd
import re
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration parameters
input_file = 'data/credit/Credit-Distorted.csv'
output_file = 'data/credit/credit_distorted_cleaned_run3.csv'
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

# Store original activities
df['original_activity'] = df[activity_column]

# Identify distorted activities
df['isdistorted'] = df[activity_column].str.endswith(distorted_suffix).astype(int)
df['BaseActivity'] = df[activity_column].replace(f'{distorted_suffix}$', '', regex=True)

# Preprocess activity names
def preprocess_activity(activity):
    activity = str(activity).lower()
    activity = re.sub(r'[^a-z0-9\s]', '', activity)
    activity = re.sub(r'\s+', ' ', activity).strip()
    return activity

df['ProcessedActivity'] = df['BaseActivity'].apply(preprocess_activity)

# Generate n-grams
def get_ngrams(text, n):
    return set([text[i:i+n] for i in range(len(text)-n+1)])

# Calculate Jaccard similarity
def jaccard_similarity(a, b, n):
    ngrams_a = get_ngrams(a, n)
    ngrams_b = get_ngrams(b, n)
    if not ngrams_a and not ngrams_b:
        return 1.0
    intersection = len(ngrams_a & ngrams_b)
    union = len(ngrams_a | ngrams_b)
    return intersection / union

# Find similar pairs
unique_activities = df[df['ProcessedActivity'].str.len() >= min_length]['ProcessedActivity'].unique()
similar_pairs = []

for i in range(len(unique_activities)):
    for j in range(i+1, len(unique_activities)):
        sim = jaccard_similarity(unique_activities[i], unique_activities[j], ngram_size)
        if sim >= similarity_threshold:
            similar_pairs.append((unique_activities[i], unique_activities[j]))

# Union-Find implementation
parent = {}

def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(x, y):
    parent[find(x)] = find(y)

# Initialize parent
for activity in unique_activities:
    parent[activity] = activity

# Build clusters
for a1, a2 in similar_pairs:
    union(a1, a2)

# Group activities by cluster
clusters = defaultdict(list)
for activity in unique_activities:
    clusters[find(activity)].append(activity)

# Filter clusters with at least 2 members
clusters = {k: v for k, v in clusters.items() if len(v) >= 2}

# Majority voting for canonical forms
activity_to_canonical = {}
for cluster in clusters.values():
    original_activities = []
    for processed_activity in cluster:
        mask = df['ProcessedActivity'] == processed_activity
        original_activities.extend(df.loc[mask, 'original_activity'].tolist())
    
    if not original_activities:
        continue
    
    # Get most frequent original activity
    canonical = max(set(original_activities), key=original_activities.count)
    
    for processed_activity in cluster:
        mask = df['ProcessedActivity'] == processed_activity
        activity_to_canonical.update({orig: canonical for orig in df.loc[mask, 'original_activity']})

# Apply canonical forms
df['canonical_activity'] = df['original_activity']
df.loc[df['original_activity'].isin(activity_to_canonical), 'canonical_activity'] = df['original_activity'].map(activity_to_canonical)
df['Activity_fixed'] = df['canonical_activity']

# Update isdistorted based on canonical form
df['isdistorted'] = (df['original_activity'] != df['canonical_activity']).astype(int)

# Calculate metrics if label column exists
if label_column in df.columns:
    y_true = df[label_column].fillna(0).astype(str).str.contains('distorted', case=False).astype(int)
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

# Integrity check
total_clusters = len(clusters)
total_distorted = df['isdistorted'].sum()
total_clean = len(df) - total_distorted

print("\n=== Integrity Check ===")
print(f"Total distortion clusters detected: {total_clusters}")
print(f"Total activities marked as distorted: {total_distorted}")
print(f"Activities to be fixed: {total_distorted}")

# Fix activities
df[activity_column] = df['Activity_fixed']

# Save output
output_columns = [case_column, timestamp_column, activity_column, 'original_activity', 'Activity_fixed']
if label_column in df.columns:
    output_columns.append(label_column)

df[output_columns].to_csv(output_file, index=False)

# Summary statistics
unique_before = df['original_activity'].nunique()
unique_after = df['Activity_fixed'].nunique()
reduction = unique_before - unique_after
reduction_pct = (reduction / unique_before) * 100

print("\n=== Summary Statistics ===")
print(f"Total number of events: {len(df)}")
print(f"Number of distorted events detected: {total_distorted}")
print(f"Unique activities before fixing: {unique_before}")
print(f"Unique activities after fixing: {unique_after}")
print(f"Activity reduction count: {reduction} ({reduction_pct:.2f}%)")
print(f"Output file path: {output_file}")

# Print sample transformations
sample_transforms = df[df['isdistorted'] == 1][['original_activity', 'Activity_fixed']].drop_duplicates().head(10)
if not sample_transforms.empty:
    print("\nSample transformations (original → canonical):")
    for _, row in sample_transforms.iterrows():
        print(f"{row['original_activity']} → {row['Activity_fixed']}")

print(f"\nRun 3: Processed dataset saved to: {output_file}")
print(f"Run 3: Final dataset shape: {df.shape}")
print(f"Run 3: Dataset: credit")
print(f"Run 3: Task type: distorted")