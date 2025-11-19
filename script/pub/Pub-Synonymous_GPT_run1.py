# Generated script for Pub-Synonymous - Run 1
# Generated on: 2025-11-13T17:01:21.544132
# Model: gpt-4o-2024-11-20

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
import re

# Configuration parameters
input_file = 'data/pub/Pub-Synonymous.csv'
output_file = 'data/pub/pub_synonymous_cleaned_run1.csv'
similarity_threshold = 0.8
min_synonym_group_size = 2
ngram_range = (1, 3)
label_column = 'label'

# Helper function to normalize activity labels
def normalize_activity(activity):
    if pd.isna(activity):
        return "empty_activity"
    activity = activity.lower()
    activity = re.sub(r'[^a-z0-9\s]', '', activity)  # Remove non-alphanumeric except spaces
    activity = re.sub(r'\s+', ' ', activity).strip()  # Collapse multiple spaces and trim
    return activity

# Load and validate the dataset
try:
    df = pd.read_csv(input_file)
    print(f"Run 1: Original dataset shape: {df.shape}")

    # Normalize column names
    if 'CaseID' in df.columns and 'Case' not in df.columns:
        df.rename(columns={'CaseID': 'Case'}, inplace=True)
    if 'Activity' not in df.columns:
        raise ValueError("The required column 'Activity' is missing from the dataset.")

    # Create original_activity column
    df['original_activity'] = df['Activity']
    df['Activity'] = df['Activity'].astype(str).fillna('')
    df['Activity_clean'] = df['Activity'].apply(normalize_activity)

    # Parse Timestamp if exists
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

    # Sort by Case and Timestamp if both exist
    if 'Case' in df.columns and 'Timestamp' in df.columns:
        df.sort_values(by=['Case', 'Timestamp'], inplace=True)

    print(f"Run 1: Dataset after preprocessing: {df.shape}")
    print(f"Run 1: Unique activities before normalization: {df['Activity'].nunique()}")

    # Extract unique activities
    unique_activities = df['Activity_clean'].unique()
    if len(unique_activities) < 2:
        print("Run 1: Not enough unique activities for clustering. Skipping synonym detection.")
        df['is_synonymous_event'] = 0
        df.to_csv(output_file, index=False)
        print(f"Run 1: Processed dataset saved to: {output_file}")
        print(f"Run 1: Final dataset shape: {df.shape}")
        exit()

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=ngram_range, lowercase=True, min_df=1)
    tfidf_matrix = vectorizer.fit_transform(unique_activities)
    similarity_matrix = cosine_similarity(tfidf_matrix)

    print(f"Run 1: TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Run 1: Unique activities count: {len(unique_activities)}")

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

    # Build clusters
    clusters = defaultdict(list)
    for idx in range(len(unique_activities)):
        clusters[find(idx)].append(unique_activities[idx])

    # Filter clusters by minimum size
    valid_clusters = {k: v for k, v in clusters.items() if len(v) >= min_synonym_group_size}
    print(f"Run 1: Synonym clusters discovered: {len(valid_clusters)}")

    # Map activities to canonical forms
    activity_to_canonical = {}
    for cluster in valid_clusters.values():
        activity_counts = Counter(df.loc[df['Activity_clean'].isin(cluster), 'Activity_clean'])
        canonical = activity_counts.most_common(1)[0][0]
        for activity in cluster:
            activity_to_canonical[activity] = canonical

    # Assign canonical forms and flags
    df['SynonymGroup'] = df['Activity_clean'].map(lambda x: find(unique_activities.tolist().index(x)) if x in unique_activities else -1)
    df['canonical_activity'] = df['Activity_clean'].map(activity_to_canonical).fillna(df['Activity_clean'])
    df['is_synonymous_event'] = np.where(df['Activity_clean'] != df['canonical_activity'], 1, 0)

    # Detection metrics
    if label_column in df.columns:
        y_true = df[label_column].notna().astype(int)
        y_pred = df['is_synonymous_event']
        precision = np.sum((y_true & y_pred)) / np.sum(y_pred) if np.sum(y_pred) > 0 else 0
        recall = np.sum((y_true & y_pred)) / np.sum(y_true) if np.sum(y_true) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        print("=== Detection Performance Metrics ===")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1_score:.4f}")
    else:
        print("No ground-truth labels available for evaluation.")

    # Replace Activity with canonical_activity
    df['Activity'] = df['canonical_activity']

    # Save final dataset
    df.drop(columns=['original_activity', 'Activity_clean', 'canonical_activity', 'SynonymGroup', 'is_synonymous_event'], inplace=True, errors='ignore')
    df.to_csv(output_file, index=False)

    # Summary
    print(f"Run 1: Processed dataset saved to: {output_file}")
    print(f"Run 1: Final dataset shape: {df.shape}")
    print(f"Run 1: Unique activities after normalization: {df['Activity'].nunique()}")
    print(f"Run 1: Activity reduction: {len(unique_activities)} → {df['Activity'].nunique()}")

except Exception as e:
    print(f"Run 1: Error occurred: {e}")