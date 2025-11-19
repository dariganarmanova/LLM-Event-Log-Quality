# Generated script for Credit-Synonymous - Run 2
# Generated on: 2025-11-13T16:40:12.615480
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
input_file = 'data/credit/Credit-Synonymous.csv'
input_directory = 'data/credit'
dataset_name = 'credit'
output_suffix = '_synonymous_cleaned_run2.csv'
detection_output_suffix = '_synonymous_detection_run2.csv'

df = pd.read_csv(input_file)
print(f"Run 2: Original dataset shape: {df.shape}")

# Normalize column naming
if 'CaseID' in df.columns and 'Case' not in df.columns:
    df.rename(columns={'CaseID': 'Case'}, inplace=True)

# Ensure 'Activity' column exists; raise a clear error if missing
if 'Activity' not in df.columns:
    raise ValueError("Missing 'Activity' column")

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
    activity = ''.join(e for e in activity if e.isalnum() or e.isspace())
    activity = ' '.join(activity.split())
    return activity

# Apply normalize_activity function
df['Activity_clean'] = df['original_activity'].apply(normalize_activity)
df['Activity_clean'] = df['Activity_clean'].fillna('empty_activity')

# Replace empty cleans with a sentinel like 'empty_activity'
df['Activity_clean'] = df['Activity_clean'].apply(lambda x: 'empty_activity' if x == '' else x)

# Extract unique activities
unique_activities = df['Activity_clean'].unique()

# If count < 2: set 'is_synonymous_event' = 0 for all rows, print warning, and skip clustering
if len(unique_activities) < 2:
    df['is_synonymous_event'] = 0
    print("Warning: Less than 2 unique activities; skipping clustering.")
else:
    # Build TF-IDF vectorizer
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 3), lowercase=True, min_df=1)
    tfidf_matrix = vectorizer.fit_transform(unique_activities)

    # Compute cosine similarity matrix between all unique activities
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Print: TF-IDF matrix shape and unique activity count
    print(f"Run 2: TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Run 2: Unique activity count: {len(unique_activities)}")

    # Initialize union-find over indices of unique activities
    from scipy.cluster.hierarchy import linkage
    from scipy.cluster.hierarchy import fcluster
    from scipy.cluster.hierarchy import dendrogram
    from scipy.cluster.hierarchy import leaves_list
    from scipy.cluster.hierarchy import fcluster

    # Compute linkage matrix
    linkage_matrix = linkage(similarity_matrix, method='ward')

    # Perform hierarchical clustering
    clusters = fcluster(linkage_matrix, t=0.8, criterion='distance')

    # Build clusters by root parent; map indices to cluster lists
    cluster_dict = {}
    for i, cluster in enumerate(clusters):
        if cluster not in cluster_dict:
            cluster_dict[cluster] = []
        cluster_dict[cluster].append(i)

    # Keep only clusters with size ≥ min_synonym_group_size
    min_synonym_group_size = 2
    valid_clusters = {cluster: members for cluster, members in cluster_dict.items() if len(members) >= min_synonym_group_size}

    # Create 'activity_to_cluster' mapping
    activity_to_cluster = {}
    for cluster, members in valid_clusters.items():
        for member in members:
            activity_to_cluster[unique_activities[member]] = cluster

    # Print: number of synonym clusters discovered
    print(f"Run 2: Number of synonym clusters discovered: {len(valid_clusters)}")

    # Select canonical form (majority/mode)
    canonical_mapping = {}
    for cluster, members in valid_clusters.items():
        cluster_activities = [unique_activities[member] for member in members]
        cluster_counts = df['original_activity'].value_counts()[cluster_activities]
        canonical = cluster_counts.idxmax()
        canonical_mapping[canonical] = canonical

    # Assign to DataFrame
    df['SynonymGroup'] = df['Activity_clean'].apply(lambda x: activity_to_cluster[x] if x in activity_to_cluster else -1)
    df['canonical_activity'] = df['Activity_clean'].apply(lambda x: canonical_mapping[x] if x in canonical_mapping else x)
    df['is_synonymous_event'] = df.apply(lambda row: 1 if row['SynonymGroup'] != -1 and row['Activity_clean'] != row['canonical_activity'] else 0, axis=1)

# Calculate detection metrics (before fixing)
if 'label' in df.columns:
    y_true = df['label'].notnull() & (df['label'] != '')
    y_pred = df['is_synonymous_event']
    precision = np.mean(y_true & y_pred)
    recall = np.mean(y_true & y_pred)
    f1_score = 2 * (precision * recall) / (precision + recall)
    print(f"=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
else:
    print("No ground-truth labels available for evaluation.")

# Integrity check
print(f"Run 2: Total synonym clusters found: {len(valid_clusters)}")
print(f"Run 2: Total events flagged as synonyms: {df['is_synonymous_event'].sum()}")
print(f"Run 2: Total canonical/clean events: {df['Activity_clean'].value_counts()['empty_activity']}")

# Fix activities
df['Activity'] = df['canonical_activity']

# Create final fixed dataset
final_df = df.drop(['original_activity', 'Activity_clean', 'canonical_activity', 'SynonymGroup', 'is_synonymous_event'], axis=1)

# Save output and summary
output_file = f"{input_directory}/{dataset_name}{output_suffix}"
final_df.to_csv(output_file, index=False)
print(f"Run 2: Processed dataset saved to: {output_file}")
print(f"Run 2: Final dataset shape: {final_df.shape}")
print(f"Run 2: Dataset: {dataset_name}")
print(f"Run 2: Task type: synonymous")
print(f"Run 2: Output file path: {output_file}")

# Print summary
print(f"Run 2: Total rows: {final_df.shape[0]}")
print(f"Run 2: Synonym clusters found: {len(valid_clusters)}")
print(f"Run 2: Synonymous events replaced: {df['is_synonymous_event'].sum()}")
print(f"Run 2: Replacement rate: {(df['is_synonymous_event'].sum() / final_df.shape[0]) * 100:.2f}%")
print(f"Run 2: Unique activities before → after: {len(unique_activities)} → {final_df['Activity'].nunique()}")
print(f"Run 2: Activity reduction: {len(unique_activities) - final_df['Activity'].nunique()} ({(len(unique_activities) - final_df['Activity'].nunique()) / len(unique_activities) * 100:.2f}%)")

# Print up to 10 sample transformations
transformations = []
for index, row in df.iterrows():
    if row['Activity'] != row['canonical_activity']:
        transformations.append(f"'{row['original_activity']}' → '{row['canonical_activity']}'")
print(f"Run 2: Sample transformations: {transformations[:10]}")