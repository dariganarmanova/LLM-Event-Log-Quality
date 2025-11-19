# Generated script for BPIC15-Synonymous - Run 1
# Generated on: 2025-11-13T12:37:31.126873
# Model: meta-llama/Llama-3.1-8B-Instruct

```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# Configuration parameters
time_threshold = 2.0
require_same_resource = False
min_matching_events = 2
max_mismatches = 1
activity_suffix_pattern = r"(_signed\d*|_\d+)$"
similarity_threshold = 0.8
case_sensitive = False
use_fuzzy_matching = False

# Load the data
input_file = 'data/bpic15/BPIC15-Synonymous.csv'
input_directory = 'data/bpic15'
dataset_name = 'bpic15'
output_suffix = '_synonymous_cleaned_run1.csv'
detection_output_suffix = '_synonymous_detection_run1.csv'

df = pd.read_csv(input_file)

# Normalize column naming
if 'CaseID' in df.columns and 'Case' not in df.columns:
    df.rename(columns={'CaseID': 'Case'}, inplace=True)

# Ensure 'Activity' column exists; raise a clear error if missing
if 'Activity' not in df.columns:
    raise ValueError("Missing 'Activity' column in the dataset.")

# Store original values: create 'original_activity' from 'Activity'
df['original_activity'] = df['Activity']

# Ensure 'Activity' is string-typed; fill missing with empty string
df['Activity'] = df['Activity'].astype(str).fillna('')

# If 'Timestamp' exists, parse to datetime (coerce errors)
if 'Timestamp' in df.columns:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

# If both 'Case' and 'Timestamp' exist, sort by 'Case', then 'Timestamp'
if 'Case' in df.columns and 'Timestamp' in df.columns:
    df.sort_values(by=['Case', 'Timestamp'], inplace=True)

# Print: dataset shape, first few rows, and number of unique 'Activity' values
print(f"Run 1: Original dataset shape: {df.shape}")
print(df.head())
print(f"Run 1: Number of unique 'Activity' values: {df['Activity'].nunique()}")

# Define normalize_activity function
def normalize_activity(activity):
    if pd.isnull(activity):
        return ''
    activity = activity.lower()
    activity = ''.join(e for e in activity if e.isalnum() or e.isspace())
    activity = ' '.join(activity.split())
    return activity

# Apply normalize_activity to build 'Activity_clean' from 'original_activity'
df['Activity_clean'] = df['original_activity'].apply(normalize_activity)

# Replace empty cleans with a sentinel like 'empty_activity'
df['Activity_clean'] = df['Activity_clean'].apply(lambda x: 'empty_activity' if x == '' else x)

# Extract unique activities from 'Activity_clean'
unique_activities = df['Activity_clean'].unique()

# If count < 2: set 'is_synonymous_event' = 0 for all rows, print warning, and skip clustering
if len(unique_activities) < 2:
    df['is_synonymous_event'] = 0
    print("Warning: Less than 2 unique activities found. Skipping clustering.")
else:
    # Build TF-IDF vectorizer
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 3), lowercase=True, min_df=1)
    tfidf_matrix = vectorizer.fit_transform(unique_activities)

    # Compute cosine similarity matrix between all unique activities
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Print: TF-IDF matrix shape and unique activity count
    print(f"Run 1: TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Run 1: Number of unique activities: {len(unique_activities)}")

    # Initialize union-find over indices of unique activities
    n = len(unique_activities)
    parent = list(range(n))

    # For each pair (i, j) with i < j:
    # If similarity_matrix[i, j] ≥ similarity_threshold, union the sets
    for i in range(n):
        for j in range(i + 1, n):
            if similarity_matrix[i, j] >= similarity_threshold:
                union(parent, i, j)

    # Build clusters by root parent; map indices to cluster lists
    clusters = {}
    for i in range(n):
        root = find(parent, i)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(i)

    # Keep only clusters with size ≥ min_synonym_group_size
    min_synonym_group_size = 2
    valid_clusters = {k: v for k, v in clusters.items() if len(v) >= min_synonym_group_size}

    # Create 'activity_to_cluster' mapping:
    # Each 'Activity_clean' → cluster id (-1 for unclustered)
    activity_to_cluster = {}
    for i, activity in enumerate(unique_activities):
        if i in valid_clusters:
            activity_to_cluster[activity] = find(parent, i)
        else:
            activity_to_cluster[activity] = -1

    # Print: number of synonym clusters discovered
    print(f"Run 1: Number of synonym clusters discovered: {len(valid_clusters)}")

    # For each valid cluster:
    # Gather member 'Activity_clean' strings.
    # Count each member’s total occurrences in the DataFrame.
    # **Canonical** = member with the highest frequency in the original data.
    for cluster_id, cluster in valid_clusters.items():
        cluster_activities = [unique_activities[i] for i in cluster]
        cluster_counts = Counter(df['original_activity'].apply(normalize_activity).values)
        canonical_activity = max(cluster_activities, key=lambda x: cluster_counts[x])
        activity_to_cluster[canonical_activity] = cluster_id

    # Assign to DataFrame:
    # Create 'SynonymGroup' (cluster id or -1).
    # Create 'canonical_activity': if clustered, set to cluster canonical; else keep 'Activity_clean'.
    # Create 'is_synonymous_event':
    # 1 if in a cluster **and** not equal to its canonical.
    # 0 otherwise (canonical or unclustered).
    df['SynonymGroup'] = df['Activity_clean'].map(activity_to_cluster)
    df['canonical_activity'] = df['Activity_clean']
    df['is_synonymous_event'] = (df['SynonymGroup'] != -1) & (df['Activity_clean'] != df['canonical_activity'])
    df['canonical_activity'] = df['canonical_activity'].apply(lambda x: df.loc[df['Activity_clean'] == x, 'original_activity'].iloc[0] if x != -1 else x)

# Compute detection metrics (BEFORE FIXING)
label_column = 'label'
if label_column in df.columns:
    y_true = (df[label_column].notnull() & df[label_column].apply(lambda x: not pd.isnull(x) or len(x) > 0))
    y_pred = df['is_synonymous_event']
    precision = np.mean(y_true[y_pred])
    recall = np.mean(y_true[y_pred])
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    print(f"=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
else:
    print("No ground-truth labels available for evaluation.")

# Integrity check
print(f"Run 1: Total synonym clusters found: {len(valid_clusters)}")
print(f"Run 1: Total events flagged as synonyms: {df['is_synonymous_event'].sum()}")
print(f"Run 1: Total canonical/clean events: {df['is_synonymous_event'].sum() + df['SynonymGroup'].eq(-1).sum()}")

# Confirm only clustered non-canonical items are marked for change
assert (df['is_synonymous_event'] & (df['SynonymGroup'] != -1)).sum() == (df['SynonymGroup'] != -1).sum()

# Fix activities
df['Activity'] = df['canonical_activity']

# Create final fixed dataset
fixed_df = df.drop(['original_activity', 'Activity_clean', 'canonical_activity', 'SynonymGroup', 'is_synonymous_event'], axis=1)

# Save fixed dataset to:
# input_directory + '/' + dataset_name + output_suffix + '.csv' (no index)
fixed_df.to_csv(input_directory + '/' + dataset_name + output_suffix + '.csv', index=False)

# Print summary:
print(f"Run 1: Processed dataset saved to: {input_directory + '/' + dataset_name + output_suffix + '.csv'}")
print(f"Run 1: Final dataset shape: {fixed_df.shape}")
print(f"Run 1: Dataset: {dataset_name}")
print(f"Run 1: Task type: synonymous")

# Print up to 10 sample transformations:
transformations = []
for index, row in df.iterrows():
    if row['Activity'] != row['canonical_activity']:
        transformations.append(f"'{row['original_activity']}' → '{row['canonical_activity']}'")
    if len(transformations) >= 10:
        break
print(f"Run 1: Sample transformations:")
for transformation in transformations:
    print(transformation)

def find(parent