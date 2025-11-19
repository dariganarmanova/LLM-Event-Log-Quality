# Generated script for BPIC15-Synonymous - Run 3
# Generated on: 2025-11-13T12:41:50.648571
# Model: deepseek-ai/DeepSeek-V3-0324

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score
import re
from collections import defaultdict

# Configuration parameters
input_file = 'data/bpic15/BPIC15-Synonymous.csv'
output_file = 'data/bpic15/bpic15_synonymous_cleaned_run3.csv'
dataset_name = 'bpic15'
task_type = 'synonymous'
run_number = 3

similarity_threshold = 0.8
min_synonym_group_size = 2
use_case_scope = False
ngram_range = (1, 3)
save_detection_file = False
label_column = 'label'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'

# Step 1: Load and Validate
df = pd.read_csv(input_file)
print(f"Run {run_number}: Original dataset shape: {df.shape}")

# Normalize column names
if 'CaseID' in df.columns and 'Case' not in df.columns:
    df.rename(columns={'CaseID': 'Case'}, inplace=True)

if activity_column not in df.columns:
    raise ValueError(f"Required column '{activity_column}' not found in the dataset.")

# Store original activity values
df['original_activity'] = df[activity_column].copy()

# Ensure Activity is string type
df[activity_column] = df[activity_column].astype(str)

# Handle missing values
df[activity_column] = df[activity_column].fillna('')

# Parse Timestamp if exists
if timestamp_column in df.columns:
    df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce')

# Sort by Case and Timestamp if both exist
if case_column in df.columns and timestamp_column in df.columns:
    df.sort_values(by=[case_column, timestamp_column], inplace=True)

print(f"Run {run_number}: First few rows:")
print(df.head())
print(f"Run {run_number}: Number of unique activities: {df[activity_column].nunique()}")

# Step 2: Normalize Activity Labels
def normalize_activity(activity):
    if pd.isna(activity):
        return ''
    activity = str(activity).lower()
    activity = re.sub(r'[^a-z0-9\s]', '', activity)
    activity = re.sub(r'\s+', ' ', activity).strip()
    return activity

df['Activity_clean'] = df['original_activity'].apply(normalize_activity)
df['Activity_clean'] = df['Activity_clean'].replace('', 'empty_activity')

# Step 3: TF-IDF Embedding and Similarity
unique_activities = df['Activity_clean'].unique()
if len(unique_activities) < 2:
    print(f"Run {run_number}: Warning: Not enough unique activities for clustering.")
    df['is_synonymous_event'] = 0
else:
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=ngram_range, lowercase=True, min_df=1)
    tfidf_matrix = vectorizer.fit_transform(unique_activities)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    print(f"Run {run_number}: TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Run {run_number}: Unique activities count: {len(unique_activities)}")

    # Step 4: Cluster Using Union-Find
    parent = list(range(len(unique_activities)))

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    def union(u, v):
        u_root = find(u)
        v_root = find(v)
        if u_root != v_root:
            parent[v_root] = u_root

    for i in range(len(unique_activities)):
        for j in range(i + 1, len(unique_activities)):
            if similarity_matrix[i, j] >= similarity_threshold:
                union(i, j)

    clusters = defaultdict(list)
    for i in range(len(unique_activities)):
        clusters[find(i)].append(i)

    valid_clusters = [cluster for cluster in clusters.values() if len(cluster) >= min_synonym_group_size]
    print(f"Run {run_number}: Number of synonym clusters discovered: {len(valid_clusters)}")

    # Step 5: Select Canonical Form (Majority/Mode)
    activity_to_index = {act: idx for idx, act in enumerate(unique_activities)}
    index_to_activity = {idx: act for idx, act in enumerate(unique_activities)}

    canonical_mapping = {}
    activity_to_cluster = {act: -1 for act in unique_activities}

    for cluster_id, cluster in enumerate(valid_clusters):
        cluster_activities = [index_to_activity[idx] for idx in cluster]
        activity_counts = df[df['Activity_clean'].isin(cluster_activities)]['Activity_clean'].value_counts()
        canonical = activity_counts.idxmax()

        for idx in cluster:
            activity = index_to_activity[idx]
            canonical_mapping[activity] = canonical
            activity_to_cluster[activity] = cluster_id

    df['SynonymGroup'] = df['Activity_clean'].map(activity_to_cluster).fillna(-1).astype(int)
    df['canonical_activity'] = df['Activity_clean'].map(canonical_mapping).fillna(df['Activity_clean'])
    df['is_synonymous_event'] = ((df['SynonymGroup'] != -1) & (df['Activity_clean'] != df['canonical_activity'])).astype(int)

# Step 6: Calculate Detection Metrics (BEFORE FIXING)
if label_column in df.columns:
    y_true = (~df[label_column].isna() & (df[label_column] != '')).astype(int)
    y_pred = df['is_synonymous_event']
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print("=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
else:
    print("=== Detection Performance Metrics ===")
    print("Precision: 0.0000")
    print("Recall: 0.0000")
    print("F1-Score: 0.0000")
    print("No ground-truth labels available for evaluation")

# Step 7: Integrity Check
synonym_clusters = df['SynonymGroup'].nunique() - (1 if -1 in df['SynonymGroup'].values else 0)
synonym_events = df['is_synonymous_event'].sum()
clean_events = len(df) - synonym_events
print(f"Run {run_number}: Total synonym clusters found: {synonym_clusters}")
print(f"Run {run_number}: Total events flagged as synonyms: {synonym_events}")
print(f"Run {run_number}: Total canonical/clean events: {clean_events}")

# Step 8: Fix Activities
df[activity_column] = df['canonical_activity']

# Step 10: Create Final Fixed Dataset
columns_to_drop = ['original_activity', 'Activity_clean', 'canonical_activity', 'SynonymGroup', 'is_synonymous_event']
final_columns = [col for col in df.columns if col not in columns_to_drop]
final_df = df[final_columns].copy()

# Step 11: Save Output and Summary
final_df.to_csv(output_file, index=False)

unique_before = df['original_activity'].nunique()
unique_after = final_df[activity_column].nunique()
replacement_rate = (synonym_events / len(df)) * 100
reduction_percent = ((unique_before - unique_after) / unique_before) * 100

print(f"Run {run_number}: Summary:")
print(f"Total rows: {len(df)}")
print(f"Synonym clusters found: {synonym_clusters}")
print(f"Synonymous events replaced: {synonym_events}")
print(f"Replacement rate: {replacement_rate:.2f}%")
print(f"Unique activities before: {unique_before} → after: {unique_after}")
print(f"Activity reduction: {unique_before - unique_after} ({reduction_percent:.2f}%)")
print(f"Output file path: {output_file}")

# Print sample transformations
changed = df[df['is_synonymous_event'] == 1].head(10)
if not changed.empty:
    print("Sample transformations:")
    for _, row in changed.iterrows():
        print(f"'{row['original_activity']}' → '{row[activity_column]}'")
else:
    print("No sample transformations to display