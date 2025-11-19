# Generated script for BPIC15-Synonymous - Run 2
# Generated on: 2025-11-13T12:37:32.589125
# Model: meta-llama/Llama-3.1-8B-Instruct

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import LabelEncoder

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
output_suffix = '_synonymous_cleaned_run2.csv'
detection_output_suffix = '_synonymous_detection_run2.csv'

df = pd.read_csv(input_file)
print(f"Run 2: Original dataset shape: {df.shape}")

# Normalize column naming
if 'CaseID' in df.columns and 'Case' not in df.columns:
    df.rename(columns={'CaseID': 'Case'}, inplace=True)

# Ensure 'Activity' column exists; raise a clear error if missing
if 'Activity' not in df.columns:
    raise ValueError("Missing required column: 'Activity'")

# Store original values
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
print(df.head())
print(f"Run 2: Number of unique 'Activity' values: {df['Activity'].nunique()}")

# Define normalize_activity function
def normalize_activity(activity):
    if pd.isnull(activity):
        return ''
    activity = activity.lower()
    activity = ''.join(char for char in activity if char.isalnum() or char.isspace())
    activity = ' '.join(activity.split())
    return activity

# Apply normalize_activity to build Activity_clean from original_activity
df['Activity_clean'] = df['original_activity'].apply(normalize_activity)

# Replace empty cleans with a sentinel like empty_activity
empty_activity = 'empty_activity'
df['Activity_clean'] = df['Activity_clean'].apply(lambda x: empty_activity if x == '' else x)

# Extract unique activities from Activity_clean (unique values)
unique_activities = df['Activity_clean'].unique()

# If count < 2: set is_synonymous_event = 0 for all rows, print warning, and skip clustering
if len(unique_activities) < 2:
    df['is_synonymous_event'] = 0
    print("Warning: Less than 2 unique activities; skipping clustering.")
else:
    # Build TF-IDF vectorizer
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 3), lowercase=True, min_df=1)
    tfidf_matrix = vectorizer.fit_transform(unique_activities)

    # Compute cosine similarity matrix between all unique activities
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Print: TF-IDF matrix shape and unique activity count
    print(f"Run 2: TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Run 2: Unique activity count: {len(unique_activities)}")

    # Initialize union-find over indices of unique_activities
    from unionfind import UnionFind
    uf = UnionFind(range(len(unique_activities)))

    # For each pair (i, j) with i < j:
    # If similarity_matrix[i, j] ≥ similarity_threshold, union the sets
    for i in range(len(unique_activities)):
        for j in range(i + 1, len(unique_activities)):
            if similarity_matrix[i, j] >= similarity_threshold:
                uf.union(i, j)

    # Build clusters by root parent; map indices to cluster lists
    clusters = {}
    for i in range(len(unique_activities)):
        root = uf.find(i)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(i)

    # Keep only clusters with size ≥ min_synonym_group_size
    min_synonym_group_size = 2
    valid_clusters = {k: v for k, v in clusters.items() if len(v) >= min_synonym_group_size}

    # Create activity_to_cluster mapping:
    # Each Activity_clean → cluster id (−1 for unclustered)
    activity_to_cluster = {}
    for i, activity in enumerate(df['Activity_clean']):
        if i in valid_clusters:
            activity_to_cluster[i] = valid_clusters[i]
        else:
            activity_to_cluster[i] = -1

    # Print: number of synonym clusters discovered
    print(f"Run 2: Number of synonym clusters discovered: {len(valid_clusters)}")

    # For each valid cluster:
    # Gather member Activity_clean strings.
    # Count each member’s total occurrences in the DataFrame.
    # **Canonical** = member with the highest frequency in the original data.
    canonical_mapping = {}
    for cluster_id, indices in valid_clusters.items():
        member_activities = [df.loc[i, 'Activity_clean'] for i in indices]
        frequencies = df['original_activity'].value_counts()[member_activities].values
        canonical = member_activities[np.argmax(frequencies)]
        canonical_mapping.update({i: canonical for i in indices})

    # Assign to DataFrame:
    # Create SynonymGroup (cluster id or −1).
    # Create canonical_activity: if clustered, set to cluster canonical; else keep Activity_clean.
    # Create is_synonymous_event:
    # 1 if in a cluster **and** not equal to its canonical.
    # 0 otherwise (canonical or unclustered).
    df['SynonymGroup'] = df['Activity_clean'].map(activity_to_cluster)
    df['canonical_activity'] = df['Activity_clean'].map(canonical_mapping)
    df['is_synonymous_event'] = (df['SynonymGroup'] != -1) & (df['Activity'] != df['canonical_activity'])

# If label_column exists:
# Ground truth y_true = 1 if label non-null and non-empty; else 0.
# Predictions y_pred = is_synonymous_event.
# Compute precision, recall, F1-score (handle zero divisions with 0).
if 'label' in df.columns:
    y_true = (df['label'].notnull()) & (df['label'] != '')
    y_pred = df['is_synonymous_event']
    from sklearn.metrics import precision_score, recall_score, f1_score
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
else:
    print("No ground-truth labels available for evaluation.")

# Print: total synonym clusters found
print(f"Run 2: Total synonym clusters found: {len(valid_clusters)}")

# Print: total events flagged as synonyms
print(f"Run 2: Total events flagged as synonyms: {df['is_synonymous_event'].sum()}")

# Print: total canonical/clean events
print(f"Run 2: Total canonical/clean events: {df['is_synonymous_event'].eq(0).sum()}")

# Confirm only clustered non-canonical items are marked for change.
assert (df['is_synonymous_event'] & (df['SynonymGroup'] != -1)).all()

# Fix Activities
df['Activity'] = df['canonical_activity']

# Drop helper columns from a copy to produce the final output
final_df = df.drop(columns=['original_activity', 'Activity_clean', 'canonical_activity', 'SynonymGroup', 'is_synonymous_event'])

# Preserve all original non-helper columns (e.g., Case, Timestamp, label)
final_df = final_df[['Case', 'Timestamp', 'Activity', 'label']]

# Save fixed dataset to:
final_df.to_csv(f'{input_directory}/{dataset_name}{output_suffix}', index=False)

# Print summary:
print(f"Run 2: Processed dataset saved to: {input_directory}/{dataset_name}{output_suffix}")
print(f"Run 2: Final dataset shape: {final_df.shape}")
print(f"Run 2: Dataset: {dataset_name}")
print(f"Run 2: Task type: synonymous")

# Print up to 10 sample transformations:
print("=== Sample Transformations ===")
for i in range(min(10, df.shape[0])):
    if df.loc[i, 'Activity'] != df.loc[i, 'canonical_activity']:
        print(f"{df.loc[i, 'original_activity']} → {df.loc[i, 'canonical_activity']}")

print(f"Run 2: Output file path: {input_directory}/{dataset_name}{output_suffix}")