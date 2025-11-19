# Generated script for Credit-Homonymous - Run 2
# Generated on: 2025-11-13T16:19:52.776108
# Model: meta-llama/Llama-3.1-8B-Instruct

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import normalize

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
output_file = 'data/credit/credit_homonymous_cleaned_run2.csv'
df = pd.read_csv(input_file)

# Ensure required columns exist: Case, Activity, Timestamp
required_columns = ['Case', 'Activity', 'Timestamp']
for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Required column '{column}' not found in the input file.")

# Normalize CaseID → Case if needed (handle naming variations)
df['Case'] = df['Case'].str.lower()

# Identify Homonymous Activities
homonymous_suffix = ':homonymous'
df['ishomonymous'] = df['Activity'].str.endswith(homonymous_suffix)
df['BaseActivity'] = df['Activity'].str.replace(homonymous_suffix, '')

# Preprocess Activity Names
df['ProcessedActivity'] = df['Activity'].str.lower().str.replace(homonymous_suffix, '')
df['ProcessedActivity'] = df['ProcessedActivity'].str.replace('_', ' ')
df['ProcessedActivity'] = df['ProcessedActivity'].str.replace('-', ' ')
df['ProcessedActivity'] = df['ProcessedActivity'].str.replace(r'\s+', ' ')

# Vectorize Activities
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1,3))
unique_activities = df['ProcessedActivity'].unique()
vectorized_activities = vectorizer.fit_transform(unique_activities)
dense_vectorized_activities = normalize(vectorized_activities.toarray())

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
    cluster_mask = cluster_labels == cluster_label
    cluster_activities = df['ProcessedActivity'][cluster_mask]
    base_activities = df['BaseActivity'][cluster_mask]
    activity_fixed.extend([base_activities.value_counts().index[0]] * len(cluster_activities))

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
    print(f"Precision threshold (≥ 0.6) met: {precision >= 0.6}")
else:
    print("=== Detection Performance Metrics ===")
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
output_columns = ['Case', 'Timestamp', 'Resource', 'Activity', 'Activity_fixed', 'ishomonymous']
if label_column in df.columns:
    output_columns.append(label_column)
df_output = df[output_columns]
df_output['Timestamp'] = pd.to_datetime(df_output['Timestamp'])
df_output['Timestamp'] = df_output['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
df_output.to_csv(output_file, index=False)

# Summary Statistics
print(f"Total number of events: {len(df)}")
print(f"Number of homonymous events detected: {len(df[df['ishomonymous'] == 1])}")
print(f"Unique activities before fixing: {len(df['Activity'].unique())}")
print(f"Unique activities after fixing: {len(df['Activity_fixed'].unique())}")
print(f"Output file path: {output_file}")