# Generated script for BPIC11-Homonymous - Run 1
# Generated on: 2025-11-13T11:40:34.789137
# Model: meta-llama/Llama-3.1-8B-Instruct

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import precision_score, recall_score, f1_score
from datetime import datetime
import numpy as np

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
output_file = 'data/bpic11/bpic11_homonymous_cleaned_run1.csv'
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
if not all(col in df.columns for col in required_columns):
    raise ValueError("Required columns are missing from the input file")

# Normalize CaseID → Case if needed (handle naming variations)
df[case_column] = df[case_column].str.lower()

# Identify Homonymous Activities
df['ishomonymous'] = df[activity_column].str.endswith(homonymous_suffix)
df['BaseActivity'] = df[activity_column].str.replace(homonymous_suffix, '')

# Preprocess Activity Names
df['ProcessedActivity'] = df[activity_column].str.lower().str.replace(homonymous_suffix, '')
df['ProcessedActivity'] = df['ProcessedActivity'].str.replace(r'_', ' ').str.replace(r'-', ' ').str.replace(r'\s+', ' ')

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
for cluster in np.unique(cluster_labels):
    cluster_df = df[df['ProcessedActivity'].isin(unique_activities[cluster_labels == cluster])]
    BaseActivity_counts = cluster_df['BaseActivity'].value_counts()
    canonical_activity = BaseActivity_counts.index[0]
    activity_fixed.extend([canonical_activity] * len(cluster_df))
df['Activity_fixed'] = activity_fixed

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
    print(f"Precision threshold (≥ 0.6) met: {precision >= 0.6}")
else:
    print("No labels available for metric calculation.")
    print(f"Precision: 0.0000")
    print(f"Recall: 0.0000")
    print(f"F1-Score: 0.0000")

# Integrity Check
clean_activities = df[df['ishomonymous'] == 0]
unchanged_clean_activities = clean_activities[clean_activities['BaseActivity'] == clean_activities['Activity_fixed']]
unchanged_homonymous_activities = df[df['ishomonymous'] == 1][df['BaseActivity'] == df['Activity_fixed']]
corrected_homonymous_activities = df[df['ishomonymous'] == 1][df['BaseActivity'] != df['Activity_fixed']]
print(f"Clean activities modified: {len(unchanged_clean_activities)}")
print(f"Homonymous activities unchanged: {len(unchanged_homonymous_activities)}")
print(f"Homonymous activities corrected: {len(corrected_homonymous_activities)}")

# Save Output
df_output = df[['Case', 'Timestamp', 'Resource', 'Activity', 'Activity_fixed', 'ishomonymous']]
if label_column in df.columns:
    df_output[label_column] = df[label_column]
df_output['Timestamp'] = df_output['Timestamp'].apply(lambda x: datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))
df_output.to_csv(output_file, index=False)

# Summary Statistics
print(f"Total number of events: {len(df)}")
print(f"Number of homonymous events detected: {len(df[df['ishomonymous'] == 1])}")
print(f"Unique activities before fixing: {len(df['Activity'].unique())}")
print(f"Unique activities after fixing: {len(df['Activity_fixed'].unique())}")
print(f"Output file path: {output_file}")