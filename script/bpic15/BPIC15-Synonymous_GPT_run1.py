# Generated script for BPIC15-Synonymous - Run 1
# Generated on: 2025-11-13T12:37:59.579472
# Model: gpt-4o-2024-11-20

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
import re

# Configuration parameters
time_threshold = 2.0
require_same_resource = False
min_matching_events = 2
max_mismatches = 1
activity_suffix_pattern = r"(_signed\d*|_\d+)$"
similarity_threshold = 0.8
case_sensitive = False
use_fuzzy_matching = False
ngram_range = (1, 3)
min_synonym_group_size = 2

# File paths
input_file = 'data/bpic15/BPIC15-Synonymous.csv'
output_file = 'data/bpic15/bpic15_synonymous_cleaned_run1.csv'

# Load the data
try:
    df = pd.read_csv(input_file)
    print(f"Run 1: Original dataset shape: {df.shape}")
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# Normalize column names
if 'CaseID' in df.columns and 'Case' not in df.columns:
    df.rename(columns={'CaseID': 'Case'}, inplace=True)

if 'Activity' not in df.columns:
    raise ValueError("The dataset must contain an 'Activity' column.")

# Store original activity column
df['original_activity'] = df['Activity'].astype(str).fillna('')

# Normalize activities
def normalize_activity(activity):
    activity = activity.lower() if not case_sensitive else activity
    activity = re.sub(r'[^a-zA-Z0-9\s]', '', activity)  # Remove non-alphanumeric except spaces
    activity = re.sub(r'\s+', ' ', activity).strip()  # Collapse multiple spaces
    return activity

df['Activity_clean'] = df['original_activity'].apply(normalize_activity)
df['Activity_clean'].replace('', 'empty_activity', inplace=True)

# Extract unique activities
unique_activities = df['Activity_clean'].unique()
if len(unique_activities) < 2:
    print("Warning: Less than 2 unique activities found. No clustering performed.")
    df['is_synonymous_event'] = 0
    df.to_csv(output_file, index=False)
    print(f"Run 1: Processed dataset saved to: {output_file}")
    print(f"Run 1: Final dataset shape: {df.shape}")
    exit()

# TF-IDF vectorization
vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=ngram_range, lowercase=True, min_df=1)
tfidf_matrix = vectorizer.fit_transform(unique_activities)
similarity_matrix = cosine_similarity(tfidf_matrix)

# Union-Find for clustering
parent = list(range(len(unique_activities)))

def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(x, y):
    root_x = find(x)
    root_y = find(y)
    if root_x != root_y:
        parent[root_y] = root_x

for i in range(len(unique_activities)):
    for j in range(i + 1, len(unique_activities)):
        if similarity_matrix[i, j] >= similarity_threshold:
            union(i, j)

clusters = defaultdict(list)
for i in range(len(unique_activities)):
    clusters[find(i)].append(unique_activities[i])

# Filter clusters by size
valid_clusters = {k: v for k, v in clusters.items() if len(v) >= min_synonym_group_size}

# Map activities to canonical forms
activity_to_canonical = {}
for cluster in valid_clusters.values():
    activity_counts = Counter(df[df['Activity_clean'].isin(cluster)]['Activity_clean'])
    canonical = activity_counts.most_common(1)[0][0]
    for activity in cluster:
        activity_to_canonical[activity] = canonical

df['SynonymGroup'] = df['Activity_clean'].map(lambda x: find(unique_activities.tolist().index(x)) if x in unique_activities else -1)
df['canonical_activity'] = df['Activity_clean'].map(activity_to_canonical).fillna(df['Activity_clean'])
df['is_synonymous_event'] = np.where(df['Activity_clean'] != df['canonical_activity'], 1, 0)

# Metrics calculation
if 'label' in df.columns:
    y_true = df['label'].notnull().astype(int)
    y_pred = df['is_synonymous_event']
    precision = np.sum((y_pred == 1) & (y_true == 1)) / max(np.sum(y_pred == 1), 1)
    recall = np.sum((y_pred == 1) & (y_true == 1)) / max(np.sum(y_true == 1), 1)
    f1_score = 2 * (precision * recall) / max((precision + recall), 1)
    print("=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
else:
    print("No ground-truth labels available for evaluation.")
    print("=== Detection Performance Metrics ===")
    print("Precision: 0.0000")
    print("Recall: 0.0000")
    print("F1-Score: 0.0000")

# Integrity check
print(f"Total synonym clusters found: {len(valid_clusters)}")
print(f"Total events flagged as synonyms: {df['is_synonymous_event'].sum()}")
print(f"Total canonical/clean events: {df['canonical_activity'].nunique()}")

# Save final dataset
final_df = df.drop(columns=['original_activity', 'Activity_clean', 'canonical_activity', 'SynonymGroup', 'is_synonymous_event'])
final_df.to_csv(output_file, index=False)

# Summary
print(f"Run 1: Processed dataset saved to: {output_file}")
print(f"Run 1: Final dataset shape: {final_df.shape}")
print(f"Run 1: Dataset: bpic15")
print(f"Run 1: Task type: synonymous")
print(f"Unique activities before: {len(unique_activities)}, after: {final_df['Activity'].nunique()}")