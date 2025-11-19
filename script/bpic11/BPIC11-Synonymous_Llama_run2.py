# Generated script for BPIC11-Synonymous - Run 2
# Generated on: 2025-11-13T11:52:36.719795
# Model: meta-llama/Llama-3.1-8B-Instruct

```python
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
input_file = 'data/bpic11/BPIC11-Synonymous.csv'
input_directory = 'data/bpic11'
dataset_name = 'bpic11'
output_suffix = '_synonymous_cleaned_run2.csv'
detection_output_suffix = '_synonymous_detection_run2.csv'

df = pd.read_csv(input_file)

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
print(f"Run 2: Original dataset shape: {df.shape}")
print(f"Run 2: First few rows:")
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
df['Activity_clean'] = df['Activity_clean'].fillna('empty_activity')

# Replace empty cleans with a sentinel like 'empty_activity'
df['Activity_clean'] = df['Activity_clean'].replace('', 'empty_activity')

# Extract unique activities from Activity_clean
unique_activities = df['Activity_clean'].unique()

# If count < 2: set is_synonymous_event = 0 for all rows, print warning, and skip clustering
if len(unique_activities) < 2:
    df['is_synonymous_event'] = 0
    print("Warning: Less than 2 unique activities found; skipping clustering")
else:
    # Build TF-IDF vectorizer
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 3), lowercase=True, min_df=1)
    tfidf_matrix = vectorizer.fit_transform(unique_activities)

    # Compute cosine similarity matrix between all unique activities
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Print: TF-IDF matrix shape and unique activity count
    print(f"Run 2: TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Run 2: Unique activity count: {len(unique_activities)}")

    # Initialize union-find over indices of unique_activities
    union_find = {}
    for i in range(len(unique_activities)):
        union_find[i] = i

    # For each pair (i, j) with i < j:
    # If similarity_matrix[i, j] ≥ similarity_threshold, union the sets
    for i in range(len(unique_activities)):
        for j in range(i + 1, len(unique_activities)):
            if similarity_matrix[i, j] >= similarity_threshold:
                union_find[find(union_find, j)] = find(union_find, i)

    # Build clusters by root parent; map indices to cluster lists
    clusters = defaultdict(list)
    for i in range(len(unique_activities)):
        clusters[find(union_find, i)].append(i)

    # Keep only clusters with size ≥ min_synonym_group_size
    min_synonym_group_size = 2
    valid_clusters = [cluster for cluster in clusters.values() if len(cluster) >= min_synonym_group_size]

    # Create activity_to_cluster mapping:
    # Each Activity_clean → cluster id (−1 for unclustered)
    activity_to_cluster = {}
    for i, activity in enumerate(df['Activity_clean']):
        activity_to_cluster[activity] = find(union_find, i) if i in clusters else -1

    # Print: number of synonym clusters discovered
    print(f"Run 2: Number of synonym clusters discovered: {len(valid_clusters)}")

    # For each valid cluster:
    # Gather member Activity_clean strings.
    # Count each member’s total occurrences in the DataFrame.
    # Canonical = member with the highest frequency in the original data.
    canonical_mapping = {}
    for cluster in valid_clusters:
        member_activities = [df['Activity_clean'].iloc[i] for i in cluster]
        counts = df['original_activity'].value_counts()[member_activities]
        canonical = member_activities[counts.idxmax()]
        canonical_mapping[canonical] = canonical

    # Assign to DataFrame:
    # Create SynonymGroup (cluster id or −1).
    # Create canonical_activity: if clustered, set to cluster canonical; else keep Activity_clean.
    # Create is_synonymous_event:
    # 1 if in a cluster and not equal to its canonical.
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
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1_score = f1_score(y_true, y_pred)
    print(f"=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
else:
    print("No ground-truth labels available for evaluation")

# Print: Total synonym clusters found
# Total events flagged as synonyms
# Total canonical/clean events
print(f"Run 2: Total synonym clusters found: {len(valid_clusters)}")
print(f"Run 2: Total events flagged as synonyms: {df['is_synonymous_event'].sum()}")
print(f"Run 2: Total canonical/clean events: {df['is_synonymous_event'].sum() + df['is_synonymous_event'].mean()}")

# Confirm only clustered non-canonical items are marked for change
print(f"Run 2: Only clustered non-canonical items are marked for change: {df['is_synonymous_event'] & (df['SynonymGroup'] != -1)}")

# Fix Activities
df['Activity'] = df['canonical_activity']

# Drop helper columns from a copy to produce the final output
final_df = df.drop(columns=['original_activity', 'Activity_clean', 'canonical_activity', 'SynonymGroup', 'is_synonymous_event'])

# Save fixed dataset to: input_directory + '/' + dataset_name + output_suffix + '.csv'
final_df.to_csv(input_directory + '/' + dataset_name + output_suffix + '.csv', index=False)

# Print summary:
# Total rows
# Synonym clusters found
# Synonymous events replaced (count)
# Replacement rate (% of rows)
# Unique activities **before** → **after**
# Activity reduction (count and %)
# Output file path
print(f"Run 2: Total rows: {final_df.shape[0]}")
print(f"Run 2: Synonym clusters found: {len(valid_clusters)}")
print(f"Run 2: Synonymous events replaced: {df['is_synonymous_event'].sum()}")
print(f"Run 2: Replacement rate: {(df['is_synonymous_event'].sum() / final_df.shape[0]) * 100:.2f}%")
print(f"Run 2: Unique activities before → after: {len(unique_activities)} → {final_df['Activity'].nunique()}")
print(f"Run 2: Activity reduction: {len(unique_activities) - final_df['Activity'].nunique()} ({(len(unique_activities) - final_df