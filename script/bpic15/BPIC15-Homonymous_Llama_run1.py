# Generated script for BPIC15-Homonymous - Run 1
# Generated on: 2025-11-13T14:20:06.858690
# Model: meta-llama/Llama-3.1-8B-Instruct

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import normalize
import re

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
input_file = './data/bpic15/BPIC15-Homonymous.csv'
output_file = 'data/bpic15/bpic15_homonymous_cleaned_run1.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
homonymous_suffix = ':homonymous'

df = pd.read_csv(input_file)

# Ensure required columns exist
required_columns = [case_column, activity_column, timestamp_column]
if label_column in df.columns:
    required_columns.append(label_column)
for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Required column '{column}' not found in the input file.")

# Normalize CaseID → Case if needed (handle naming variations)
df[case_column] = df[case_column].str.lower()

# Identify Homonymous Activities
df['ishomonymous'] = df[activity_column].str.endswith(homonymous_suffix)
df['BaseActivity'] = df[activity_column].str.replace(homonymous_suffix, '')

# Preprocess Activity Names
df['ProcessedActivity'] = df[activity_column].str.lower()
df['ProcessedActivity'] = df['ProcessedActivity'].str.replace(homonymous_suffix, '')
df['ProcessedActivity'] = df['ProcessedActivity'].str.replace('_', ' ')
df['ProcessedActivity'] = df['ProcessedActivity'].str.replace('-', ' ')
df['ProcessedActivity'] = df['ProcessedActivity'].str.replace(r'\s+', ' ')

# Vectorize Activities
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1,3))
unique_activities = df['ProcessedActivity'].unique()
vectorized_activities = vectorizer.fit_transform(unique_activities)
dense_vectorized_activities = vectorized_activities.toarray()

# Cluster Similar Activities
clustering = AgglomerativeClustering(
    n_clusters=None,
    metric='cosine',
    linkage='average',
    distance_threshold=1 - similarity_threshold
)
cluster_labels = clustering.fit_predict(dense_vectorized_activities)

# Majority Voting Within Clusters
activity_fixed = []
for cluster_label in np.unique(cluster_labels):
    cluster_activities = unique_activities[cluster_labels == cluster_label]
    base_activities = df.loc[df['ProcessedActivity'].isin(cluster_activities), 'BaseActivity'].value_counts().index[0]
    activity_fixed.extend([base_activities] * len(cluster_activities))
df['Activity_fixed'] = np.where(df['ishomonymous'], activity_fixed, df['Activity'])

# Calculate Detection Metrics
if label_column in df.columns:
    y_true = np.where(df[label_column].notnull(), 1, 0)
    y_pred = df['ishomonymous']
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Precision threshold (≥ 0.6) met: {'met' if precision >= 0.6 else 'not met'}")
else:
    print("=== Detection Performance Metrics ===")
    print(f"Precision: 0.0000")
    print(f"Recall: 0.0000")
    print(f"F1-Score: 0.0000")
    print("No labels available for metric calculation.")

# Integrity Check
clean_activities = df[df['ishomonymous'] == 0]
unchanged_homonymous = clean_activities[clean_activities['BaseActivity'] == clean_activities['Activity_fixed']].shape[0]
corrected_homonymous = clean_activities[clean_activities['BaseActivity'] != clean_activities['Activity_fixed']].shape[0]
print(f"Integrity Check:")
print(f"Clean activities modified: {unchanged_homonymous}")
print(f"Homonymous activities corrected: {corrected_homonymous}")

# Save Output
df.to_csv(output_file, index=False)

# Summary Statistics
print(f"Run 1: Processed dataset saved to: {output_file}")
print(f"Run 1: Final dataset shape: {df.shape}")
print(f"Run 1: Dataset: bpic15")
print(f"Run 1: Task type: homonymous")