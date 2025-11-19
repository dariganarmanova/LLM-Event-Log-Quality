# Generated script for BPIC11-Synonymous - Run 3
# Generated on: 2025-11-13T11:52:38.261819
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
input_file = 'data/bpic11/BPIC11-Synonymous.csv'
input_directory = 'data/bpic11'
dataset_name = 'bpic11'
output_suffix = '_synonymous_cleaned_run3.csv'
detection_output_suffix = '_synonymous_detection_run3.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'

df = pd.read_csv(input_file)

# Normalize column naming
if 'CaseID' in df.columns and 'Case' not in df.columns:
    df.rename(columns={'CaseID': 'Case'}, inplace=True)

# Ensure Activity column exists; raise a clear error if missing
if activity_column not in df.columns:
    raise ValueError(f"Missing required column: {activity_column}")

# Store original values
df['original_activity'] = df[activity_column]

# Ensure Activity is string-typed; fill missing with empty string
df[activity_column] = df[activity_column].astype(str).fillna('')

# If Timestamp exists, parse to datetime (coerce errors)
if timestamp_column in df.columns:
    df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce')

# If both Case and Timestamp exist, sort by Case, then Timestamp
if case_column in df.columns and timestamp_column in df.columns:
    df.sort_values(by=[case_column, timestamp_column], inplace=True)

# Print: dataset shape, first few rows, and number of unique Activity values
print(f"Run 3: Original dataset shape: {df.shape}")
print(df.head())
print(f"Run 3: Number of unique Activity values: {df[activity_column].nunique()}")

# Define normalize_activity function
def normalize_activity(activity):
    if pd.isnull(activity):
        return ''
    activity = activity.lower()
    activity = ''.join(char if char.isalnum() or char.isspace() else '' for char in activity)
    activity = ' '.join(activity.split())
    return activity.strip()

# Apply normalize_activity to build Activity_clean from original_activity
df['Activity_clean'] = df['original_activity'].apply(normalize_activity)

# Replace empty cleans with a sentinel like empty_activity
empty_activity = 'empty_activity'
df['Activity_clean'] = df['Activity_clean'].replace('', empty_activity)

# Extract unique activities from Activity_clean (unique values)
unique_activities = df['Activity_clean'].unique()

# If count < 2: set is_synonymous_event = 0 for all rows, print warning, and skip clustering
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
    print(f"Run 3: TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Run 3: Number of unique activities: {len(unique_activities)}")

    # Initialize union-find over indices of unique_activities
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        root_x = find(x)
        root_y = find(y)
        if root_x != root_y:
            parent[root_x] = root_y

    parent = list(range(len(unique_activities)))
    cluster_labels = np.zeros(len(unique_activities), dtype=int)

    # For each pair (i, j) with i < j:
    for i in range(len(unique_activities)):
        for j in range(i + 1, len(unique_activities)):
            if similarity_matrix[i, j] >= similarity_threshold:
                union(i, j)

    # Build clusters by root parent; map indices to cluster lists
    clusters = {}
    for i in range(len(unique_activities)):
        root = find(i)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(i)

    # Keep only clusters with size ≥ min_synonym_group_size
    min_synonym_group_size = 2
    valid_clusters = {k: v for k, v in clusters.items() if len(v) >= min_synonym_group_size}

    # Create activity_to_cluster mapping:
    # Each Activity_clean → cluster id (−1 for unclustered)
    activity_to_cluster = {}
    for i, activity in enumerate(unique_activities):
        if i in valid_clusters:
            activity_to_cluster[activity] = valid_clusters[i]
        else:
            activity_to_cluster[activity] = -1

    # Print: number of synonym clusters discovered
    print(f"Run 3: Number of synonym clusters discovered: {len(valid_clusters)}")

    # For each valid cluster:
    canonical_mapping = {}
    for cluster_id, activities in valid_clusters.items():
        # Gather member Activity_clean strings
        activity_list = [unique_activities[i] for i in activities]

        # Count each member’s total occurrences in the DataFrame
        counts = df[activity_column].value_counts()

        # Canonical = member with the highest frequency in the original data
        canonical = activity_list[Counter(counts[activity_list].values).most_common(1)[0][0]]

        # Build canonical_mapping:
        # Each member Activity_clean in a cluster → its cluster canonical
        for activity in activity_list:
            canonical_mapping[activity] = canonical

    # Assign to DataFrame:
    # Create SynonymGroup (cluster id or −1)
    # Create canonical_activity: if clustered, set to cluster canonical; else keep Activity_clean
    # Create is_synonymous_event:
    # 1 if in a cluster **and** not equal to its canonical.
    # 0 otherwise (canonical or unclustered)
    df['SynonymGroup'] = df['Activity_clean'].map(activity_to_cluster)
    df['canonical_activity'] = df['Activity_clean'].map(canonical_mapping)
    df['is_synonymous_event'] = (df['SynonymGroup'] != -1) & (df['Activity'] != df['canonical_activity'])

    # If label_column exists:
    # Ground truth y_true = 1 if label non-null and non-empty; else 0.
    # Predictions y_pred = is_synonymous_event.
    # Compute precision, recall, F1-score (handle zero divisions with 0).
    if label_column in df.columns:
        y_true = (df[label_column].notnull()) & (df[label_column] != '')
        y_pred = df['is_synonymous_event']
        precision = np.mean(y_true & y_pred)
        recall = np.mean(y_true & y_pred)
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        print(f"=== Detection Performance Metrics ===")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1_score:.4f}")
    else:
        print("No ground-truth labels available for evaluation.")

    # Print: Total synonym clusters found
    # Total events flagged as synonyms
    # Total canonical/clean events
    print(f"Run 3: Total synonym clusters found: {len(valid_clusters)}")
    print(f"Run 3: Total events flagged as synonyms: {df['is_synonymous_event'].sum()}")
    print(f"Run 3: Total canonical/clean events: {df['is_synonymous_event'].sum() + df['is_synonymous_event'].mean()}")

    # Confirm only clustered non-canonical items are marked for change.
    print(f"Run 3: Only clustered non-canonical items are marked for change: {df['is_synonymous_event'].sum()}")

    # Fix Activities
    # Replace Activity with canonical_activity for all rows (clustered rows map to canonical; unclustered remain unchanged).
    df['Activity'] = df['canonical_activity']

    # Create Final Fixed Dataset
    # Drop helper columns from a copy to produce the final output (e.g., original_activity, Activity_clean, canonical_activity, SynonymGroup, is_synonymous_event).
    # Preserve all original non-helper columns (e.g., Case, Timestamp, label).
    final_df = df.drop(columns=['original_activity', 'Activity_clean', 'canonical_activity', 'SynonymGroup', 'is_synonymous_event'])

    # Save Output and Summary
    # Save fixed dataset to:
    # input_directory + / + dataset_name + output_suffix + .csv (no index)
    final_df.to_csv(f'{input_directory}/{dataset_name}{output_suffix