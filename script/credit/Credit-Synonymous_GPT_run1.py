# Generated script for Credit-Synonymous - Run 1
# Generated on: 2025-11-13T16:46:40.428417
# Model: gpt-4o-2024-11-20

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
import re

# Configuration parameters
input_file = 'data/credit/Credit-Synonymous.csv'
output_file = 'data/credit/credit_synonymous_cleaned_run1.csv'
time_threshold = 2.0
require_same_resource = False
min_matching_events = 2
max_mismatches = 1
activity_suffix_pattern = r"(_signed\d*|_\d+)$"
similarity_threshold = 0.8
case_sensitive = False
use_fuzzy_matching = False
ngram_range = (1, 3)
min_synonym_group_size = 2

# Load and validate the dataset
try:
    df = pd.read_csv(input_file)
    print(f"Run 1: Original dataset shape: {df.shape}")
    
    # Normalize column names
    if 'CaseID' in df.columns and 'Case' not in df.columns:
        df.rename(columns={'CaseID': 'Case'}, inplace=True)
    if 'Activity' not in df.columns:
        raise ValueError("The dataset must contain an 'Activity' column.")
    
    # Store original values
    df['original_activity'] = df['Activity']
    df['Activity'] = df['Activity'].astype(str).fillna('')
    
    # Parse timestamp if exists
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    
    # Sort by Case and Timestamp if both exist
    if 'Case' in df.columns and 'Timestamp' in df.columns:
        df.sort_values(by=['Case', 'Timestamp'], inplace=True)
    
    print(f"Run 1: Dataset shape after loading and validation: {df.shape}")
    print(f"Run 1: Number of unique activities: {df['Activity'].nunique()}")
except Exception as e:
    print(f"Error loading or validating dataset: {e}")
    exit(1)

# Normalize activity labels
def normalize_activity(activity):
    activity = activity.lower() if not case_sensitive else activity
    activity = re.sub(r'[^a-zA-Z0-9\s]', '', activity)  # Remove non-alphanumeric except spaces
    activity = re.sub(r'\s+', ' ', activity).strip()  # Collapse multiple spaces
    return activity if activity else 'empty_activity'

df['Activity_clean'] = df['Activity'].apply(normalize_activity)

# TF-IDF embedding and similarity
unique_activities = df['Activity_clean'].unique()
if len(unique_activities) < 2:
    print("Run 1: Not enough unique activities to perform clustering.")
    df['is_synonymous_event'] = 0
else:
    try:
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=ngram_range, lowercase=True, min_df=1)
        tfidf_matrix = vectorizer.fit_transform(unique_activities)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        print(f"Run 1: TF-IDF matrix shape: {tfidf_matrix.shape}")
        print(f"Run 1: Number of unique activities: {len(unique_activities)}")
    except Exception as e:
        print(f"Error during TF-IDF computation: {e}")
        exit(1)

    # Cluster using union-find
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
        root = find(idx)
        clusters[root].append(activity)

    # Filter clusters by minimum size
    clusters = {k: v for k, v in clusters.items() if len(v) >= min_synonym_group_size}

    print(f"Run 1: Number of synonym clusters discovered: {len(clusters)}")

    # Select canonical form
    activity_to_canonical = {}
    for cluster in clusters.values():
        activity_counts = Counter(df[df['Activity_clean'].isin(cluster)]['Activity_clean'])
        canonical = activity_counts.most_common(1)[0][0]
        for activity in cluster:
            activity_to_canonical[activity] = canonical

    # Map activities to canonical forms
    df['SynonymGroup'] = df['Activity_clean'].map(lambda x: find(unique_activities.tolist().index(x)) if x in activity_to_canonical else -1)
    df['canonical_activity'] = df['Activity_clean'].map(activity_to_canonical).fillna(df['Activity_clean'])
    df['is_synonymous_event'] = np.where(df['Activity_clean'] != df['canonical_activity'], 1, 0)

# Save final dataset
try:
    df_final = df.drop(columns=['original_activity', 'Activity_clean', 'canonical_activity', 'SynonymGroup', 'is_synonymous_event'], errors='ignore')
    df_final.to_csv(output_file, index=False)
    print(f"Run 1: Processed dataset saved to: {output_file}")
    print(f"Run 1: Final dataset shape: {df_final.shape}")
    print(f"Run 1: Dataset: credit")
    print(f"Run 1: Task type: synonymous")
except Exception as e:
    print(f"Error saving the processed dataset: {e}")
    exit(1)