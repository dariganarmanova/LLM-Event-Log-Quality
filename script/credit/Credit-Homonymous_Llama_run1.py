# Generated script for Credit-Homonymous - Run 1
# Generated on: 2025-11-13T16:19:50.465196
# Model: meta-llama/Llama-3.1-8B-Instruct

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
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
input_file = 'data/credit/Credit-Homonymous.csv'
output_file = 'data/credit/credit_homonymous_cleaned_run1.csv'
df = pd.read_csv(input_file)

# Ensure required columns exist: Case, Activity, Timestamp
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
def preprocess_activity(activity):
    activity = activity.lower()
    activity = re.sub(r'[_\-]', ' ', activity)
    activity = re.sub(r'\s+', ' ', activity)
    return activity.strip()

df['ProcessedActivity'] = df['Activity'].apply(preprocess_activity)

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
    BaseActivity_counts = df.loc[df['ishomonymous'] == 1, 'BaseActivity'].value_counts()
    BaseActivity_most_frequent = BaseActivity_counts.index[BaseActivity_counts.idxmax()]
    activity_fixed.extend([BaseActivity_most_frequent] * len(cluster_activities))

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
    print(f"Precision threshold (≥ 0.6) met: {'met' if precision >= 0.6 else 'not met'}")
else:
    print("No labels available for metric calculation.")

# Integrity Check
clean_activities = df.loc[df['ishomonymous'] == 0]
unchanged_homonymous = clean_activities.loc[clean_activities['BaseActivity'] == clean_activities['Activity_fixed']].shape[0]
corrected_homonymous = clean_activities.loc[clean_activities['BaseActivity'] != clean_activities['Activity_fixed']].shape[0]
print(f"Clean activities modified: {corrected_homonymous}")
print(f"Homonymous activities unchanged: {unchanged_homonymous}")

# Save Output
output_columns = ['Case', 'Timestamp', 'Resource', 'Activity', 'Activity_fixed', 'ishomonymous']
if label_column in df.columns:
    output_columns.append(label_column)
df_output = df[output_columns]
df_output['Timestamp'] = pd.to_datetime(df_output['Timestamp'])
df_output['Timestamp'] = df_output['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
df_output.to_csv(output_file, index=False)

# Summary Statistics
print(f"Run 1: Total number of events: {df.shape[0]}")
print(f"Run 1: Number of homonymous events detected: {df['ishomonymous'].sum()}")
print(f"Run 1: Unique activities before fixing: {len(unique_activities)}")
print(f"Run 1: Unique activities after fixing: {df['ProcessedActivity'].nunique()}")
print(f"Run 1: Output file path: {output_file}")