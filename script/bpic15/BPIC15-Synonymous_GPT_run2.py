# Generated script for BPIC15-Synonymous - Run 2
# Generated on: 2025-11-13T12:38:12.652786
# Model: gpt-4o-2024-11-20

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
import re

# Configuration parameters
input_file = 'data/bpic15/BPIC15-Synonymous.csv'
output_file = 'data/bpic15/bpic15_synonymous_cleaned_run2.csv'
similarity_threshold = 0.8
min_synonym_group_size = 2
ngram_range = (1, 3)
label_column = 'label'

# Load and validate the dataset
try:
    df = pd.read_csv(input_file)
    print(f"Run 2: Original dataset shape: {df.shape}")

    # Normalize column names
    if 'CaseID' in df.columns and 'Case' not in df.columns:
        df.rename(columns={'CaseID': 'Case'}, inplace=True)
    if 'Activity' not in df.columns:
        raise ValueError("The required column 'Activity' is missing from the dataset.")

    # Store original activity column
    df['original_activity'] = df['Activity']
    df['Activity'] = df['Activity'].astype(str).fillna('')

    # Parse timestamp if available
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        if 'Case' in df.columns:
            df.sort_values(by=['Case', 'Timestamp'], inplace=True)

    print(f"Run 2: Dataset loaded successfully. Total unique activities: {df['Activity'].nunique()}")

    # Normalize activity labels
    def normalize_activity(activity):
        activity = activity.lower()  # Convert to lowercase
        activity = re.sub(r'[^a-z0-9\s]', '', activity)  # Remove non-alphanumeric characters except spaces
        activity = re.sub(r'\s+', ' ', activity).strip()  # Collapse multiple spaces and trim
        return activity if activity else 'empty_activity'

    df['Activity_clean'] = df['Activity'].apply(normalize_activity)
    unique_activities = df['Activity_clean'].unique()

    if len(unique_activities) < 2:
        print("Run 2: Not enough unique activities for clustering. Skipping synonym detection.")
        df['is_synonymous_event'] = 0
        df.to_csv(output_file, index=False)
        print(f"Run 2: Processed dataset saved to: {output_file}")
        print(f"Run 2: Final dataset shape: {df.shape}")
        exit()

    # TF-IDF embedding and similarity computation
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=ngram_range, lowercase=True, min_df=1)
    tfidf_matrix = vectorizer.fit_transform(unique_activities)
    similarity_matrix = cosine_similarity(tfidf_matrix)

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

    clusters = defaultdict(list)
    for idx, activity in enumerate(unique_activities):
        clusters[find(idx)].append(activity)

    # Filter clusters by size
    valid_clusters = {k: v for k, v in clusters.items() if len(v) >= min_synonym_group_size}

    # Map activities to canonical forms
    activity_to_cluster = {}
    canonical_mapping = {}
    for cluster_id, activities in valid_clusters.items():
        activity_counts = Counter(df[df['Activity_clean'].isin(activities)]['Activity_clean'])
        canonical = activity_counts.most_common(1)[0][0]
        for activity in activities:
            activity_to_cluster[activity] = cluster_id
            canonical_mapping[activity] = canonical

    df['SynonymGroup'] = df['Activity_clean'].map(activity_to_cluster).fillna(-1).astype(int)
    df['canonical_activity'] = df['Activity_clean'].map(canonical_mapping).fillna(df['Activity_clean'])
    df['is_synonymous_event'] = np.where(
        (df['SynonymGroup'] != -1) & (df['Activity_clean'] != df['canonical_activity']), 1, 0
    )

    # Detection metrics
    if label_column in df.columns:
        y_true = df[label_column].notna().astype(int)
        y_pred = df['is_synonymous_event']
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        print("=== Detection Performance Metrics ===")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}")
    else:
        print("No ground-truth labels available for evaluation.")

    # Replace activities with canonical forms
    df['Activity'] = df['canonical_activity']

    # Save the final dataset
    final_df = df.drop(columns=['original_activity', 'Activity_clean', 'canonical_activity', 'SynonymGroup', 'is_synonymous_event'])
    final_df.to_csv(output_file, index=False)

    # Summary
    print(f"Run 2: Processed dataset saved to: {output_file}")
    print(f"Run 2: Final dataset shape: {final_df.shape}")
    print(f"Run 2: Total synonym clusters found: {len(valid_clusters)}")
    print(f"Run 2: Total synonymous events replaced: {df['is_synonymous_event'].sum()}")
    print(f"Run 2: Unique activities before: {len(unique_activities)}, after: {final_df['Activity'].nunique()}")
except Exception as e:
    print(f"An error occurred: {e}")