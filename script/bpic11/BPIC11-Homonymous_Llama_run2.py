# Generated script for BPIC11-Homonymous - Run 2
# Generated on: 2025-11-13T11:40:37.522037
# Model: meta-llama/Llama-3.1-8B-Instruct

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
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
input_file = 'data/bpic11/BPIC11-Homonymous.csv'
output_file = 'data/bpic11/bpic11_homonymous_cleaned_run2.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
homonymous_suffix = ':homonymous'

df = pd.read_csv(input_file)
print(f"Run 2: Original dataset shape: {df.shape}")

# Ensure required columns exist
required_columns = [case_column, activity_column, timestamp_column]
if not all(col in df.columns for col in required_columns):
    raise ValueError("Required columns not found in the dataset")

# Normalize CaseID → Case if needed (handle naming variations)
df[case_column] = df[case_column].str.lower()

# Identify Homonymous Activities
df['ishomonymous'] = df[activity_column].str.endswith(homonymous_suffix)
df['BaseActivity'] = df[activity_column].str.replace(homonymous_suffix, '')

# Preprocess Activity Names
df['ProcessedActivity'] = df['BaseActivity'].str.lower()
df['ProcessedActivity'] = df['ProcessedActivity'].str.replace(r'_', ' ')
df['ProcessedActivity'] = df['ProcessedActivity'].str.replace(r'-', ' ')
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
    cluster_df = df[df['ProcessedActivity'].isin(unique_activities[cluster_labels == cluster_label])]
    activity_fixed.extend(cluster_df['BaseActivity'].value_counts().index[0])

df['Activity_fixed'] = df['ishomonymous'].apply(lambda x: activity_fixed[cluster_labels[np.where(df['ProcessedActivity'] == df['ProcessedActivity'][np.where(df['ishomonymous'] == x)[0][0]])[0]]] if x == 1 else df['BaseActivity'])

# Calculate Detection Metrics
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
    print(f"Precision threshold (≥ 0.6) met: {'Yes' if precision >= 0.6 else 'No'}")
else:
    print(f"=== Detection Performance Metrics ===")
    print(f"Precision: 0.0000")
    print(f"Recall: 0.0000")
    print(f"F1-Score: 0.0000")
    print("No labels available for metric calculation.")

# Integrity Check
clean_activities = df[df['ishomonymous'] == 0]
unchanged_homonymous = clean_activities[clean_activities['BaseActivity'] == clean_activities['Activity_fixed']]
corrected_homonymous = clean_activities[clean_activities['BaseActivity'] != clean_activities['Activity_fixed']]
print(f"Clean activities modified: {len(corrected_homonymous)}")
print(f"Homonymous activities unchanged: {len(unchanged_homonymous)}")
print(f"Homonymous activities corrected: {len(corrected_homonymous)}")

# Save Output
df_output = df[['Case', 'Timestamp', 'Resource', 'Activity', 'Activity_fixed', 'ishomonymous']]
if label_column in df.columns:
    df_output[label_column] = df[label_column]
df_output['Timestamp'] = pd.to_datetime(df_output['Timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
df_output.to_csv(output_file, index=False)

# Summary Statistics
print(f"Run 2: Processed dataset saved to: {output_file}")
print(f"Run 2: Final dataset shape: {df_output.shape}")
print(f"Run 2: Dataset: bpic11")
print(f"Run 2: Task type: homonymous")