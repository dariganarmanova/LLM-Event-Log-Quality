# Generated script for Pub-Distorted - Run 3
# Generated on: 2025-11-13T17:46:24.288228
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
df = pd.read_csv('data/pub/Pub-Distorted.csv')
print(f"Run 3: Original dataset shape: {df.shape}")

# Normalize column names
df.columns = df.columns.str.strip()
column_mapping = {
    'CaseID': 'Case',
    'case': 'Case',
    'Activity': 'Activity',
    'Timestamp': 'Timestamp',
    'Variant': 'Variant',
    'Resource': 'Resource',
    'label': 'label'
}
df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

# Store original activity
df['original_activity'] = df['Activity']

# Identify distorted activities
df['isdistorted'] = df['Activity'].str.endswith(distorted_suffix).astype(int)
df['BaseActivity'] = df['Activity'].str.replace(distorted_suffix, '', regex=False)

# Preprocess activity names
def preprocess_activity(activity):
    if pd.isna(activity):
        return ''
    activity = str(activity)
    if not case_sensitive:
        activity = activity.lower()
    activity = re.sub(r'[^a-zA-Z0-9\s]', '', activity)
    activity = re.sub(r'\s+', ' ', activity).strip()
    return activity

df['ProcessedActivity'] = df['BaseActivity'].apply(preprocess_activity)

# Generate n-grams
def get_ngrams(text, n):
    return set([text[i:i+n] for i in range(len(text)-n+1)]) if len(text) >= n else set()

# Calculate Jaccard similarity
def jaccard_similarity(a, b, n):
    a_ngrams = get_ngrams(a, n)
    b_ngrams = get_ngrams(b, n)
    if not a_ngrams and not b_ngrams:
        return 0.0
    intersection = len(a_ngrams & b_ngrams)
    union = len(a_ngrams | b_ngrams)
    return intersection / union

# Union-Find data structure
class UnionFind:
    def __init__(self):
        self.parent = {}
    
    def find(self, item):
        if item not in self.parent:
            self.parent[item] = item
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item
    
    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a

# Find similar pairs
unique_activities = df[df['ProcessedActivity'].str.len() >= min_length]['ProcessedActivity'].unique()
uf = UnionFind()

for i in range(len(unique_activities)):
    for j in range(i+1, len(unique_activities)):
        a = unique_activities[i]
        b = unique_activities[j]
        similarity = jaccard_similarity(a, b, ngram_size)
        if similarity >= similarity_threshold:
            uf.union(a, b)

# Build clusters
clusters = defaultdict(list)
for activity in unique_activities:
    root = uf.find(activity)
    clusters[root].append(activity)

# Filter clusters with at least min_matching_events
clusters = {k: v for k, v in clusters.items() if len(v) >= min_matching_events}

# Create canonical mapping
canonical_mapping = {}
for cluster in clusters.values():
    # Get original activities for this cluster
    cluster_activities = []
    for processed_activity in cluster:
        mask = df['ProcessedActivity'] == processed_activity
        cluster_activities.extend(df.loc[mask, 'BaseActivity'].tolist())
    
    # Find most frequent original activity
    if cluster_activities:
        canonical = max(set(cluster_activities), key=cluster_activities.count)
        for processed_activity in cluster:
            mask = df['ProcessedActivity'] == processed_activity
            df.loc[mask, 'canonical_activity'] = canonical

# Mark distorted activities
df['is_distorted'] = 0
mask = (df['canonical_activity'].notna()) & (df['BaseActivity'] != df['canonical_activity'])
df.loc[mask, 'is_distorted'] = 1

# Fill canonical_activity with BaseActivity where missing
df['canonical_activity'] = df['canonical_activity'].fillna(df['BaseActivity'])

# Calculate metrics if label column exists
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
    print("No labels available for metric calculation")
    print("Precision: 0.0000")
    print("Recall: 0.0000")
    print("F1-Score: 0.0000")

# Integrity check
total_clusters = len(clusters)
total_distorted = df['is_distorted'].sum()
total_clean = len(df) - total_distorted

print("\n=== Integrity Check ===")
print(f"Total distortion clusters detected: {total_clusters}")
print(f"Total activities marked as distorted: {total_distorted}")
print(f"Activities to be fixed: {total_distorted}")

# Fix activities
df['Activity_fixed'] = df['canonical_activity']

# Save output
output_columns = ['Case', 'Timestamp', 'Activity', 'Activity_fixed']
if 'Variant' in df.columns:
    output_columns.append('Variant')
if 'Resource' in df.columns:
    output_columns.append('Resource')
if 'label' in df.columns:
    output_columns.append('label')

df_output = df[output_columns]
df_output.to_csv('data/pub/pub_distorted_cleaned_run3.csv', index=False)

# Summary statistics
unique_before = df['Activity'].nunique()
unique_after = df['Activity_fixed'].nunique()
reduction = unique_before - unique_after
reduction_pct = (reduction / unique_before) * 100 if unique_before > 0 else 0

print("\n=== Summary Statistics ===")
print(f"Total number of events: {len(df)}")
print(f"Number of distorted events detected: {total_distorted}")
print(f"Unique activities before fixing: {unique_before}")
print(f"Unique activities after fixing: {unique_after}")
print(f"Activity reduction: {reduction} ({reduction_pct:.2f}%)")
print(f"Output file path: data/pub/pub_distorted_cleaned_run3.csv")

# Print sample transformations
sample_size = min(10, len(df))
sample = df[['Activity', 'Activity_fixed']].drop_duplicates().head(sample_size)
print("\nSample transformations (original → canonical):")
for _, row in sample.iterrows():
    print(f"{row['Activity']} → {row['Activity_fixed']}")

print(f"\nRun 3: Processed dataset saved to: data/pub/pub_distorted_cleaned_run3.csv")
print(f"Run 3: Final dataset shape: {df.shape}")
print(f"Run 3: Dataset: pub")
print(f"Run 3: Task type: distorted")