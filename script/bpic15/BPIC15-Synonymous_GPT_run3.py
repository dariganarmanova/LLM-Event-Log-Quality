# Generated script for BPIC15-Synonymous - Run 3
# Generated on: 2025-11-13T12:38:25.689400
# Model: gpt-4o-2024-11-20

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter

# Configuration parameters
time_threshold = 2.0
require_same_resource = False
min_matching_events = 2
max_mismatches = 1
activity_suffix_pattern = r"(_signed\d*|_\d+)$"
similarity_threshold = 0.8
ngram_range = (1, 3)
min_synonym_group_size = 2
label_column = 'label'
use_case_scope = False

# File paths
input_file = 'data/bpic15/BPIC15-Synonymous.csv'
output_file = 'data/bpic15/bpic15_synonymous_cleaned_run3.csv'

# Step 1: Load and validate
try:
    df = pd.read_csv(input_file)
    print(f"Run 3: Original dataset shape: {df.shape}")

    # Normalize column names
    if 'CaseID' in df.columns and 'Case' not in df.columns:
        df.rename(columns={'CaseID': 'Case'}, inplace=True)
    if 'Activity' not in df.columns:
        raise ValueError("The dataset must contain an 'Activity' column.")

    # Store original activities
    df['original_activity'] = df['Activity']
    df['Activity'] = df['Activity'].astype(str).fillna('')
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    if 'Case' in df.columns and 'Timestamp' in df.columns:
        df.sort_values(by=['Case', 'Timestamp'], inplace=True)

    print(f"Run 3: Number of unique activities: {df['Activity'].nunique()}")
    print(df.head())

except Exception as e:
    print(f"Error loading or validating the dataset: {e}")
    exit()

# Step 2: Normalize activity labels
def normalize_activity(activity):
    activity = activity.lower()
    activity = ''.join(char if char.isalnum() or char.isspace() else ' ' for char in activity)
    activity = ' '.join(activity.split())
    return activity if activity else 'empty_activity'

df['Activity_clean'] = df['original_activity'].apply(normalize_activity)

# Step 3: TF-IDF embedding and similarity
unique_activities = df['Activity_clean'].unique()
if len(unique_activities) < 2:
    print("Warning: Less than two unique activities after normalization. Skipping clustering.")
    df['is_synonymous_event'] = 0
else:
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=ngram_range, lowercase=True, min_df=1)
    tfidf_matrix = vectorizer.fit_transform(unique_activities)
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Step 4: Cluster using union-find
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
    for i, activity in enumerate(unique_activities):
        clusters[find(i)].append(activity)

    valid_clusters = {k: v for k, v in clusters.items() if len(v) >= min_synonym_group_size}

    # Step 5: Select canonical form
    activity_to_canonical = {}
    for cluster, members in valid_clusters.items():
        member_counts = Counter(df[df['Activity_clean'].isin(members)]['Activity_clean'])
        canonical = member_counts.most_common(1)[0][0]
        for member in members:
            activity_to_canonical[member] = canonical

    df['SynonymGroup'] = df['Activity_clean'].map(lambda x: find(unique_activities.tolist().index(x)) if x in unique_activities else -1)
    df['canonical_activity'] = df['Activity_clean'].map(activity_to_canonical).fillna(df['Activity_clean'])
    df['is_synonymous_event'] = np.where(df['Activity_clean'] != df['canonical_activity'], 1, 0)

    print(f"Run 3: Number of synonym clusters discovered: {len(valid_clusters)}")

# Step 6: Calculate detection metrics
if label_column in df.columns:
    y_true = df[label_column].notna().astype(int)
    y_pred = df['is_synonymous_event']
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    print(f"=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}")
else:
    print("No ground-truth labels available for evaluation.")

# Step 7: Integrity check
synonym_clusters = len(valid_clusters)
synonymous_events = df['is_synonymous_event'].sum()
canonical_events = len(df) - synonymous_events
print(f"Run 3: Total synonym clusters: {synonym_clusters}")
print(f"Run 3: Total synonymous events: {synonymous_events}")
print(f"Run 3: Total canonical events: {canonical_events}")

# Step 8: Fix activities
df['Activity'] = df['canonical_activity']

# Step 10: Create final fixed dataset
final_df = df.drop(columns=['original_activity', 'Activity_clean', 'canonical_activity', 'SynonymGroup', 'is_synonymous_event'])

# Step 11: Save output and summary
try:
    final_df.to_csv(output_file, index=False)
    print(f"Run 3: Processed dataset saved to: {output_file}")
    print(f"Run 3: Final dataset shape: {final_df.shape}")
    print(f"Run 3: Dataset: bpic15")
    print(f"Run 3: Task type: synonymous")
except Exception as e:
    print(f"Error saving the processed dataset: {e}")