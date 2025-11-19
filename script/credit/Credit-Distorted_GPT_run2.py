# Generated script for Credit-Distorted - Run 2
# Generated on: 2025-11-13T15:11:20.665556
# Model: gpt-4o-2024-11-20

import pandas as pd
import re
from itertools import combinations
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration parameters
input_file = 'data/credit/Credit-Distorted.csv'
output_file = 'data/credit/credit_distorted_cleaned_run2.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
distorted_suffix = ':distorted'
ngram_size = 3
similarity_threshold = 0.56
min_length = 4

# Helper functions
def normalize_text(text, case_sensitive=False):
    if not case_sensitive:
        text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove non-alphanumeric characters
    text = re.sub(r'\s+', ' ', text).strip()  # Collapse multiple spaces
    return text

def generate_ngrams(text, n):
    return set([text[i:i+n] for i in range(len(text) - n + 1)])

def jaccard_similarity(set1, set2):
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union

def union_find_find(parent, x):
    if parent[x] != x:
        parent[x] = union_find_find(parent, parent[x])
    return parent[x]

def union_find_union(parent, rank, x, y):
    root_x = union_find_find(parent, x)
    root_y = union_find_find(parent, y)
    if root_x != root_y:
        if rank[root_x] > rank[root_y]:
            parent[root_y] = root_x
        elif rank[root_x] < rank[root_y]:
            parent[root_x] = root_y
        else:
            parent[root_y] = root_x
            rank[root_x] += 1

# Step 1: Load CSV
try:
    df = pd.read_csv(input_file)
    if not all(col in df.columns for col in [case_column, activity_column, timestamp_column]):
        raise ValueError("Missing required columns in the dataset.")
    df['original_activity'] = df[activity_column]
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# Step 2: Identify Distorted Activities
df['isdistorted'] = df[activity_column].str.endswith(distorted_suffix).astype(int)
df['BaseActivity'] = df[activity_column].str.replace(distorted_suffix, '', regex=False)

# Step 3: Preprocess Activity Names
df['ProcessedActivity'] = df['BaseActivity'].apply(lambda x: normalize_text(x))
df = df[df['ProcessedActivity'].str.len() >= min_length]

# Step 4: Calculate Jaccard N-gram Similarity
unique_activities = df['ProcessedActivity'].unique()
similar_pairs = []

for act1, act2 in combinations(unique_activities, 2):
    set1 = generate_ngrams(act1, ngram_size)
    set2 = generate_ngrams(act2, ngram_size)
    similarity = jaccard_similarity(set1, set2)
    if similarity >= similarity_threshold:
        similar_pairs.append((act1, act2))

# Step 5: Cluster Similar Activities (Union-Find)
parent = {act: act for act in unique_activities}
rank = {act: 0 for act in unique_activities}

for act1, act2 in similar_pairs:
    union_find_union(parent, rank, act1, act2)

clusters = defaultdict(list)
for act in unique_activities:
    root = union_find_find(parent, act)
    clusters[root].append(act)

# Step 6: Majority Voting Within Clusters
canonical_mapping = {}
for cluster in clusters.values():
    if len(cluster) > 1:
        original_forms = df[df['ProcessedActivity'].isin(cluster)]['BaseActivity']
        canonical_form = original_forms.value_counts().idxmax()
        for variant in cluster:
            canonical_mapping[variant] = canonical_form

df['canonical_activity'] = df['ProcessedActivity'].map(canonical_mapping).fillna(df['BaseActivity'])
df['is_distorted'] = (df['BaseActivity'] != df['canonical_activity']).astype(int)

# Step 7: Calculate Detection Metrics
if label_column in df.columns:
    df['y_true'] = df[label_column].fillna('').str.contains('distorted', case=False).astype(int)
    y_true = df['y_true']
    y_pred = df['is_distorted']
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print("=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
else:
    print("No labels available for metric calculation.")

# Step 8: Integrity Check
total_clusters = len([c for c in clusters.values() if len(c) > 1])
total_distorted = df['is_distorted'].sum()
print(f"Total distortion clusters detected: {total_clusters}")
print(f"Total activities marked as distorted: {total_distorted}")

# Step 9: Fix Activities
df['Activity_fixed'] = df['canonical_activity']

# Step 10: Save Output
try:
    df.to_csv(output_file, index=False)
    print(f"Run 2: Processed dataset saved to: {output_file}")
except Exception as e:
    print(f"Error saving file: {e}")
    exit()

# Step 11: Summary Statistics
print(f"Run 2: Final dataset shape: {df.shape}")
print(f"Unique activities before fixing: {len(unique_activities)}")
print(f"Unique activities after fixing: {df['Activity_fixed'].nunique()}")
print(f"Activity reduction: {len(unique_activities) - df['Activity_fixed'].nunique()} activities")