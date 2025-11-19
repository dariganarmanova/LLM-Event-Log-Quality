# Generated script for BPIC11-Homonymous - Run 2
# Generated on: 2025-11-13T11:41:16.271764
# Model: gpt-4o-2024-11-20

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import precision_score, recall_score, f1_score
import re
from datetime import datetime

# Configuration parameters
time_threshold = 2.0
require_same_resource = False
min_matching_events = 2
max_mismatches = 1
activity_suffix_pattern = r"(_signed\d*|_\d+)$"
similarity_threshold = 0.8
linkage_method = 'average'
homonymous_suffix = ':homonymous'
input_file = 'data/bpic11/BPIC11-Homonymous.csv'
output_file = 'data/bpic11/bpic11_homonymous_cleaned_run2.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'

# Load the data
try:
    df = pd.read_csv(input_file)
    print(f"Run 2: Original dataset shape: {df.shape}")
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# Ensure required columns exist
required_columns = {case_column, activity_column, timestamp_column}
if not required_columns.issubset(df.columns):
    print(f"Error: Missing required columns. Found: {df.columns}. Required: {required_columns}")
    exit()

# Step 2: Identify Homonymous Activities
df['ishomonymous'] = df[activity_column].str.endswith(homonymous_suffix).astype(int)
df['BaseActivity'] = df[activity_column].str.replace(homonymous_suffix, '', regex=False)

# Step 3: Preprocess Activity Names
def preprocess_activity(activity):
    activity = activity.lower()
    activity = re.sub(homonymous_suffix, '', activity)
    activity = re.sub(r'[_\-]', ' ', activity)
    activity = re.sub(r'\s+', ' ', activity).strip()
    return activity

df['ProcessedActivity'] = df['BaseActivity'].apply(preprocess_activity)

# Step 4: Vectorize Activities
unique_activities = df['ProcessedActivity'].unique()
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
try:
    activity_vectors = vectorizer.fit_transform(unique_activities).toarray()
except Exception as e:
    print(f"Error during vectorization: {e}")
    exit()

# Step 5: Cluster Similar Activities
try:
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric='cosine',
        linkage=linkage_method,
        distance_threshold=1 - similarity_threshold
    )
    cluster_labels = clustering.fit_predict(activity_vectors)
except Exception as e:
    print(f"Error during clustering: {e}")
    exit()

# Map clusters back to activities
cluster_map = {unique_activities[i]: cluster_labels[i] for i in range(len(unique_activities))}
df['ClusterID'] = df['ProcessedActivity'].map(cluster_map)

# Step 6: Majority Voting Within Clusters
canonical_names = {}
for cluster_id in np.unique(cluster_labels):
    cluster_activities = df[df['ClusterID'] == cluster_id]['BaseActivity']
    most_common = cluster_activities.value_counts().idxmax()
    canonical_names[cluster_id] = most_common

df['Activity_fixed'] = df['ClusterID'].map(canonical_names)

# Step 7: Calculate Detection Metrics
if label_column in df.columns:
    y_true = df[label_column].notna().astype(int)
    y_pred = df['ishomonymous']
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print("=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Precision threshold (≥ 0.6): {'met' if precision >= 0.6 else 'not met'}")
else:
    print("=== Detection Performance Metrics ===")
    print("No labels available for metric calculation.")

# Step 8: Integrity Check
clean_modified = df[(df['ishomonymous'] == 0) & (df['BaseActivity'] != df['Activity_fixed'])].shape[0]
unchanged_homonymous = df[(df['ishomonymous'] == 1) & (df['BaseActivity'] == df['Activity_fixed'])].shape[0]
corrected_homonymous = df[(df['ishomonymous'] == 1) & (df['BaseActivity'] != df['Activity_fixed'])].shape[0]

print("=== Integrity Check ===")
print(f"Clean activities modified: {clean_modified}")
print(f"Homonymous activities unchanged: {unchanged_homonymous}")
print(f"Homonymous activities corrected: {corrected_homonymous}")

# Step 9: Save Output
df[timestamp_column] = pd.to_datetime(df[timestamp_column]).dt.strftime('%Y-%m-%d %H:%M:%S')
output_columns = [case_column, timestamp_column, 'Activity', 'Activity_fixed', 'ishomonymous']
if label_column in df.columns:
    output_columns.append(label_column)

try:
    df[output_columns].to_csv(output_file, index=False)
    print(f"Run 2: Processed dataset saved to: {output_file}")
except Exception as e:
    print(f"Error saving file: {e}")
    exit()

# Step 10: Summary Statistics
print("=== Summary Statistics ===")
print(f"Total number of events: {df.shape[0]}")
print(f"Number of homonymous events detected: {df['ishomonymous'].sum()}")
print(f"Unique activities before fixing: {df['BaseActivity'].nunique()}")
print(f"Unique activities after fixing: {df['Activity_fixed'].nunique()}")
print(f"Run 2: Final dataset shape: {df.shape}")