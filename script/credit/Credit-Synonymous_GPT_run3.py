# Generated script for Credit-Synonymous - Run 3
# Generated on: 2025-11-13T16:47:06.593388
# Model: gpt-4o-2024-11-20

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
import re

# Configuration parameters
input_file = 'data/credit/Credit-Synonymous.csv'
output_file = 'data/credit/credit_synonymous_cleaned_run3.csv'
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

# Load the data
try:
    df = pd.read_csv(input_file)
    print(f"Run 3: Original dataset shape: {df.shape}")
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# Normalize column names
if 'CaseID' in df.columns and 'Case' not in df.columns:
    df.rename(columns={'CaseID': 'Case'}, inplace=True)
if 'Activity' not in df.columns:
    raise ValueError("The dataset must contain an 'Activity' column.")

# Store original activity and ensure Activity column is string-typed
df['original_activity'] = df['Activity']
df['Activity'] = df['Activity'].fillna('').astype(str)

# Normalize timestamps if present
if 'Timestamp' in df.columns:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

# Sort by Case and Timestamp if both exist
if 'Case' in df.columns and 'Timestamp' in df.columns:
    df.sort_values(by=['Case', 'Timestamp'], inplace=True)

# Print dataset info
print(f"Run 3: Dataset shape after loading: {df.shape}")
print(f"Run 3: Number of unique activities: {df['Activity'].nunique()}")

# Normalize activity labels
def normalize_activity(activity):
    activity = activity.lower() if not case_sensitive else activity
    activity = re.sub(r'[^a-zA-Z0-9\s]', '', activity)
    activity = re.sub(r'\s+', ' ', activity).strip()
    return activity

df['Activity_clean'] = df['Activity'].apply(normalize_activity)
df['Activity_clean'] = df['Activity_clean'].replace('', 'empty_activity')

# Extract unique activities
unique_activities = df['Activity_clean'].unique()
if len(unique_activities) < 2:
    print("Run 3: Not enough unique activities for clustering.")
    df['is_synonymous_event'] = 0
    df.to_csv(output_file, index=False)
    print(f"Run 3: Processed dataset saved to: {output_file}")
    print(f"Run 3: Final dataset shape: {df.shape}")
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

# Group activities by clusters
clusters = defaultdict(list)
for i, activity in enumerate(unique_activities):
    clusters[find(i)].append(activity)

# Filter clusters by minimum size
valid_clusters = {k: v for k, v in clusters.items() if len(v) >= min_synonym_group_size}

# Map activities to canonical forms
activity_to_canonical = {}
for cluster in valid_clusters.values():
    activity_counts = Counter(df.loc[df['Activity_clean'].isin(cluster), 'Activity_clean'])
    canonical = activity_counts.most_common(1)[0][0]
    for activity in cluster:
        activity_to_canonical[activity] = canonical

# Assign canonical forms and synonym flags
df['SynonymGroup'] = df['Activity_clean'].map(lambda x: find(unique_activities.tolist().index(x)) if x in unique_activities else -1)
df['canonical_activity'] = df['Activity_clean'].map(activity_to_canonical).fillna(df['Activity_clean'])
df['is_synonymous_event'] = np.where(df['Activity_clean'] != df['canonical_activity'], 1, 0)

# Print detection metrics if label column exists
if 'label' in df.columns:
    y_true = df['label'].notna().astype(int)
    y_pred = df['is_synonymous_event']
    precision = np.sum((y_true & y_pred)) / np.sum(y_pred) if np.sum(y_pred) > 0 else 0
    recall = np.sum((y_true & y_pred)) / np.sum(y_true) if np.sum(y_true) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print("=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}")
else:
    print("No ground-truth labels available for evaluation.")

# Replace Activity with canonical_activity
df['Activity'] = df['canonical_activity']

# Drop helper columns for final output
final_df = df.drop(columns=['original_activity', 'Activity_clean', 'canonical_activity', 'SynonymGroup', 'is_synonymous_event'], errors='ignore')

# Save final processed dataset
final_df.to_csv(output_file, index=False)

# Print summary
print(f"Run 3: Processed dataset saved to: {output_file}")
print(f"Run 3: Final dataset shape: {final_df.shape}")
print(f"Run 3: Total synonym clusters found: {len(valid_clusters)}")
print(f"Run 3: Total synonymous events replaced: {df['is_synonymous_event'].sum()}")
print(f"Run 3: Replacement rate: {df['is_synonymous_event'].mean() * 100:.2f}%")
print(f"Run 3: Unique activities before: {len(unique_activities)}, after: {final_df['Activity'].nunique()}")
print(f"Run 3: Activity reduction: {len(unique_activities) - final_df['Activity'].nunique()} ({(len(unique_activities) - final_df['Activity'].nunique()) / len(unique_activities) * 100:.2f}%)")

# Print sample transformations
sample_changes = df[df['is_synonymous_event'] == 1][['original_activity', 'Activity']].drop_duplicates().head(10)
print("Sample transformations:")
for _, row in sample_changes.iterrows():
    print(f"'{row['original_activity']}' → '{row['Activity']}'")