# Generated script for Credit-Synonymous - Run 2
# Generated on: 2025-11-13T16:42:45.888056
# Model: deepseek-ai/DeepSeek-V3-0324

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score
import re
from collections import defaultdict

# Configuration parameters
similarity_threshold = 0.8
min_synonym_group_size = 2
use_case_scope = False
ngram_range = (1, 3)
save_detection_file = False
label_column = 'label'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
output_file = 'data/credit/credit_synonymous_cleaned_run2.csv'

# Step 1: Load and Validate
df = pd.read_csv('data/credit/Credit-Synonymous.csv')
print(f"Run 2: Original dataset shape: {df.shape}")

# Normalize column names
if 'CaseID' in df.columns and 'Case' not in df.columns:
    df.rename(columns={'CaseID': 'Case'}, inplace=True)

# Ensure Activity column exists
if 'Activity' not in df.columns:
    raise ValueError("Activity column is missing in the dataset")

# Store original activity values
df['original_activity'] = df['Activity'].copy()

# Ensure Activity is string-typed
df['Activity'] = df['Activity'].astype(str)

# Parse Timestamp if exists
if 'Timestamp' in df.columns:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

# Sort by Case and Timestamp if both exist
if 'Case' in df.columns and 'Timestamp' in df.columns:
    df.sort_values(by=['Case', 'Timestamp'], inplace=True)

print(f"Run 2: First few rows:\n{df.head()}")
print(f"Run 2: Number of unique Activity values: {df['Activity'].nunique()}")

# Step 2: Normalize Activity Labels
def normalize_activity(activity):
    if pd.isna(activity):
        return ''
    activity = str(activity).lower()
    activity = re.sub(r'[^a-z0-9 ]', '', activity)
    activity = re.sub(r'\s+', ' ', activity).strip()
    return activity

df['Activity_clean'] = df['original_activity'].apply(normalize_activity)
df['Activity_clean'] = df['Activity_clean'].replace('', 'empty_activity')

# Step 3: TF-IDF Embedding and Similarity
unique_activities = df['Activity_clean'].unique()
if len(unique_activities) < 2:
    print("Run 2: Warning: Not enough unique activities for clustering")
    df['is_synonymous_event'] = 0
else:
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=ngram_range, lowercase=True, min_df=1)
    tfidf_matrix = vectorizer.fit_transform(unique_activities)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    print(f"Run 2: TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Run 2: Unique activity count: {len(unique_activities)}")

    # Step 4: Cluster Using Union-Find
    parent = list(range(len(unique_activities)))

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    for i in range(len(unique_activities)):
        for j in range(i + 1, len(unique_activities)):
            if similarity_matrix[i, j] >= similarity_threshold:
                root_i = find(i)
                root_j = find(j)
                if root_i != root_j:
                    parent[root_j] = root_i

    clusters = defaultdict(list)
    for idx, activity in enumerate(unique_activities):
        root = find(idx)
        clusters[root].append(activity)

    valid_clusters = {k: v for k, v in clusters.items() if len(v) >= min_synonym_group_size}
    print(f"Run 2: Number of synonym clusters discovered: {len(valid_clusters)}")

    # Step 5: Select Canonical Form (Majority/Mode)
    activity_to_cluster = {}
    canonical_mapping = {}

    for cluster_id, members in valid_clusters.items():
        member_counts = {member: df[df['Activity_clean'] == member].shape[0] for member in members}
        canonical = max(member_counts, key=member_counts.get)
        for member in members:
            activity_to_cluster[member] = cluster_id
            canonical_mapping[member] = canonical

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
    print(f"Run 2: Total synonym clusters found: {len(valid_clusters)}")
    print(f"Run 2: Total events flagged as synonyms: {df['is_synonymous_event'].sum()}")
    print(f"Run 2: Total canonical/clean events: {len(df) - df['is_synonymous_event'].sum()}")

    # Step 8: Fix Activities
    df['Activity'] = df['canonical_activity']

# Step 10: Create Final Fixed Dataset
output_columns = [col for col in df.columns if col not in ['original_activity', 'Activity_clean', 'canonical_activity', 'SynonymGroup', 'is_synonymous_event']]
final_df = df[output_columns].copy()

# Step 11: Save Output and Summary
final_df.to_csv(output_file, index=False)

unique_before = df['original_activity'].nunique()
unique_after = final_df['Activity'].nunique()
replaced_count = df['is_synonymous_event'].sum() if 'is_synonymous_event' in df.columns else 0

print(f"Run 2: Total rows: {len(final_df)}")
print(f"Run 2: Synonym clusters found: {len(valid_clusters) if 'valid_clusters' in locals() else 0}")
print(f"Run 2: Synonymous events replaced: {replaced_count}")
print(f"Run 2: Replacement rate: {replaced_count / len(final_df) * 100:.2f}%")
print(f"Run 2: Unique activities before → after: {unique_before} → {unique_after}")
print(f"Run 2: Activity reduction: {unique_before - unique_after} ({(unique_before - unique_after) / unique_before * 100:.2f}%)")
print(f"Run 2: Output file path: {output_file}")

if 'valid_clusters' in locals():
    sample_changes = df[df['is_synonymous_event'] == 1][['original_activity', 'canonical_activity']].drop_duplicates().head(10)
    print("Run 2: Sample transformations:")
    for _, row in sample_changes.iterrows():
        print(f"'{row['original_activity']}' → '{row['canonical_activity']}'")

print(f"Run 2: Processed dataset saved to: {output_file}")
print(f"Run 2: Final dataset shape: {final_df.shape}")
print(f"Run 2: Dataset: credit")
print(f"Run 2: Task type: synonymous")