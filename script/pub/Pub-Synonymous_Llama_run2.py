# Generated script for Pub-Synonymous - Run 2
# Generated on: 2025-11-13T17:05:41.159222
# Model: meta-llama/Llama-3.1-8B-Instruct

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict

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
input_file = 'data/pub/Pub-Synonymous.csv'
input_directory = 'data/pub'
dataset_name = 'pub'
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
    activity = ''.join(char if char.isalnum() or char.isspace() else '' for char in activity)
    activity = ' '.join(activity.split())
    return activity

# Apply normalize_activity to build 'Activity_clean' from 'original_activity'
df['Activity_clean'] = df['original_activity'].apply(normalize_activity)

# Replace empty cleans with a sentinel like 'empty_activity'
df['Activity_clean'] = df['Activity_clean'].apply(lambda x: 'empty_activity' if x == '' else x)

# Extract unique activities from 'Activity_clean'
unique_activities = df['Activity_clean'].unique()

# If count < 2: set 'is_synonymous_event' to 0 for all rows, print warning, and skip clustering
if len(unique_activities) < 2:
    df['is_synonymous_event'] = 0
    print("Warning: Less than two unique activities found. Skipping clustering.")
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
    union_find = {}
    for i in range(len(unique_activities)):
        union_find[i] = i

    # For each pair (i, j) with i < j:
    for i in range(len(unique_activities)):
        for j in range(i + 1, len(unique_activities)):
            if similarity_matrix[i, j] >= similarity_threshold:
                union_find[j] = i

    # Build clusters by root parent; map indices to cluster lists
    clusters = defaultdict(list)
    for i in range(len(unique_activities)):
        clusters[find(union_find, i)].append(i)

    # Keep only clusters with size ≥ min_synonym_group_size
    min_synonym_group_size = 2
    clusters = {cluster: members for cluster, members in clusters.items() if len(members) >= min_synonym_group_size}

    # Create 'activity_to_cluster' mapping:
    # Each 'Activity_clean' → cluster id (-1 for unclustered)
    activity_to_cluster = {}
    for i, activity in enumerate(unique_activities):
        activity_to_cluster[activity] = find(union_find, i) if i in clusters else -1

    # Print: number of synonym clusters discovered
    print(f"Run 2: Number of synonym clusters discovered: {len(clusters)}")

    # For each valid cluster:
    canonical_mapping = {}
    for cluster, members in clusters.items():
        # Gather member 'Activity_clean' strings
        activities = [unique_activities[i] for i in members]

        # Count each member's total occurrences in the DataFrame
        counts = df['Activity_clean'].value_counts()

        # Canonical = member with the highest frequency in the original data
        canonical = activities[counts[activities].idxmax()]

        # Map cluster members to canonical
        for activity in activities:
            canonical_mapping[activity] = canonical

    # Assign to DataFrame:
    # Create 'SynonymGroup' (cluster id or -1)
    # Create 'canonical_activity': if clustered, set to cluster canonical; else keep 'Activity_clean'
    # Create 'is_synonymous_event':
    # 1 if in a cluster and not equal to its canonical
    # 0 otherwise (canonical or unclustered)
    df['SynonymGroup'] = df['Activity_clean'].map(lambda x: activity_to_cluster[x] if activity_to_cluster[x] != -1 else -1)
    df['canonical_activity'] = df['Activity_clean'].map(canonical_mapping)
    df['is_synonymous_event'] = (df['SynonymGroup'] != -1) & (df['Activity_clean'] != df['canonical_activity'])

# Calculate detection metrics (BEFORE FIXING)
label_column = 'label'
if label_column in df.columns:
    y_true = (df[label_column].notnull()) & (df[label_column] != '')
    y_pred = df['is_synonymous_event']
    precision = sum(y_true & y_pred) / sum(y_pred)
    recall = sum(y_true & y_pred) / sum(y_true)
    f1_score = 2 * precision * recall / (precision + recall)
    print(f"=== Detection Performance Metrics ===")
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
total_synonym_clusters = len(clusters)
total_events_flagged = sum(df['is_synonymous_event'])
total_canonical_events = sum(~df['is_synonymous_event'])
print(f"Run 2: Total synonym clusters found: {total_synonym_clusters}")
print(f"Run 2: Total events flagged as synonyms: {total_events_flagged}")
print(f"Run 2: Total canonical/clean events: {total_canonical_events}")
print(f"Run 2: Only clustered non-canonical items are marked for change.")

# Fix activities
df['Activity'] = df['canonical_activity']

# Create final fixed dataset
final_df = df.drop(['original_activity', 'Activity_clean', 'canonical_activity', 'SynonymGroup', 'is_synonymous_event'], axis=1)

# Save fixed dataset to:
output_file = input_directory + '/' + dataset_name + output_suffix
final_df.to_csv(output_file, index=False)

# Print summary:
print(f"Run 2: Processed dataset saved to: {output_file}")
print(f"Run 2: Final dataset shape: {final_df.shape}")
print(f"Run 2: Dataset: {dataset_name}")
print(f"Run 2: Task type: synonymous")

# Print up to 10 sample transformations:
transformations = []
for index, row in df.iterrows():
    if row['Activity'] != row['canonical_activity']:
        transformations.append(f"'{row['original_activity']}' → '{row['canonical_activity']}'")
print(f"Run 2: Sample transformations: {transformations[:10]}")

def find(union_find, i):
    if union_find[i] != i:
        union_find[i] = find(union_find, union_find[i])
    return union_find[i]