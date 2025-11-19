# Generated script for Credit-Homonymous - Run 3
# Generated on: 2025-11-13T16:19:54.586873
# Model: meta-llama/Llama-3.1-8B-Instruct

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import normalize
import numpy as np
import re
import os

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
input_file = 'data/credit/Credit-Homonymous.csv'
output_file = 'data/credit/credit_homonymous_cleaned_run3.csv'
df = pd.read_csv(input_file)

# Ensure required columns exist
required_columns = ['Case', 'Activity', 'Timestamp']
for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing required column: {column}")

# Normalize CaseID → Case if needed (handle naming variations)
df['Case'] = df['Case'].str.lower()

# Identify Homonymous Activities
homonymous_suffix = ':homonymous'
df['ishomonymous'] = df['Activity'].str.endswith(homonymous_suffix)
df['BaseActivity'] = df['Activity'].str.replace(homonymous_suffix, '')

# Preprocess Activity Names
df['ProcessedActivity'] = df['Activity'].apply(lambda x: re.sub(r'[_\-]+', ' ', x).lower())
df['ProcessedActivity'] = df['ProcessedActivity'].apply(lambda x: x.strip())

# Vectorize Activities
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1,3))
unique_activities = df['ProcessedActivity'].unique()
vectorized_activities = vectorizer.fit_transform(unique_activities)
dense_vectorized_activities = vectorized_activities.toarray()

# Cluster Similar Activities
linkage_method = 'average'
clustering = AgglomerativeClustering(
    n_clusters=None,
    metric='cosine',
    linkage=linkage_method,
    distance_threshold=1 - similarity_threshold
)
cluster_labels = clustering.fit_predict(dense_vectorized_activities)

# Majority Voting Within Clusters
activity_fixed = []
for cluster_label in np.unique(cluster_labels):
    cluster_df = df[df['ProcessedActivity'].isin(unique_activities[cluster_labels == cluster_label])]
    BaseActivity_counts = cluster_df['BaseActivity'].value_counts()
    most_frequent_BaseActivity = BaseActivity_counts.index[0]
    activity_fixed.extend([most_frequent_BaseActivity] * len(cluster_df))

# Update Activity_fixed column
df['Activity_fixed'] = activity_fixed

# Calculate Detection Metrics
label_column = 'label'
if label_column in df.columns:
    y_true = df[label_column].notnull().astype(int)
    y_pred = df['ishomonymous'].astype(int)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Precision threshold (≥ 0.6) met/not met")
else:
    print("No labels available for metric calculation.")

# Integrity Check
clean_activities = df[df['ishomonymous'] == 0]
unchanged_homonymous = clean_activities[clean_activities['BaseActivity'] == clean_activities['Activity_fixed']].shape[0]
corrected_homonymous = clean_activities[clean_activities['BaseActivity'] != clean_activities['Activity_fixed']].shape[0]
print(f"Integrity Check:")
print(f"Clean activities modified: {unchanged_homonymous}")
print(f"Homonymous activities corrected: {corrected_homonymous}")

# Save Output
df_output = df[['Case', 'Timestamp', 'Resource', 'Activity', 'Activity_fixed', 'ishomonymous']]
if label_column in df.columns:
    df_output[label_column] = df[label_column]
df_output['Timestamp'] = pd.to_datetime(df_output['Timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
df_output.to_csv(output_file, index=False)

# Summary Statistics
print(f"Run 3: Processed dataset saved to: {output_file}")
print(f"Run 3: Final dataset shape: {df.shape}")
print(f"Run 3: Dataset: credit")
print(f"Run 3: Task type: homonymous")
print(f"Run 3: Total number of events: {df.shape[0]}")
print(f"Run 3: Number of homonymous events detected: {df['ishomonymous'].sum()}")
print(f"Run 3: Unique activities before fixing: {df['Activity'].nunique()}")
print(f"Run 3: Unique activities after fixing: {df['Activity_fixed'].nunique()}")