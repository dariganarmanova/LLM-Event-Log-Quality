# Generated script for BPIC15-Homonymous - Run 1
# Generated on: 2025-11-13T14:23:22.409769
# Model: gpt-4o-2024-11-20

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import precision_score, recall_score, f1_score
import re

# Configuration parameters
input_file = 'data/bpic15/BPIC15-Homonymous.csv'
output_file = 'data/bpic15/bpic15_homonymous_cleaned_run1.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
homonymous_suffix = ':homonymous'
similarity_threshold = 0.8
linkage_method = 'average'

# Load the data
try:
    df = pd.read_csv(input_file)
    print(f"Run 1: Original dataset shape: {df.shape}")
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# Ensure required columns exist
required_columns = {case_column, activity_column, timestamp_column}
if not required_columns.issubset(df.columns):
    print(f"Error: Missing required columns. Found columns: {df.columns}")
    exit()

# Step 2: Identify homonymous activities
df['ishomonymous'] = df[activity_column].str.endswith(homonymous_suffix).astype(int)
df['BaseActivity'] = df[activity_column].str.replace(homonymous_suffix, '', regex=False)

# Step 3: Preprocess activity names
def preprocess_activity(activity):
    activity = activity.lower()
    activity = re.sub(r'[_\-]', ' ', activity)  # Replace underscores and hyphens with spaces
    activity = re.sub(r'\s+', ' ', activity).strip()  # Collapse multiple spaces
    return activity

df['ProcessedActivity'] = df['BaseActivity'].apply(preprocess_activity)

# Step 4: Vectorize activities
unique_activities = df['ProcessedActivity'].unique()
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
activity_vectors = vectorizer.fit_transform(unique_activities).toarray()

# Step 5: Cluster similar activities
clustering = AgglomerativeClustering(
    n_clusters=None,
    metric='cosine',
    linkage=linkage_method,
    distance_threshold=1 - similarity_threshold
)
cluster_labels = clustering.fit_predict(activity_vectors)

# Map cluster labels back to activities
cluster_map = {activity: cluster for activity, cluster in zip(unique_activities, cluster_labels)}
df['ClusterID'] = df['ProcessedActivity'].map(cluster_map)

# Step 6: Majority voting within clusters
canonical_map = {}
for cluster_id in np.unique(cluster_labels):
    cluster_activities = df[df['ClusterID'] == cluster_id]['BaseActivity']
    most_common_activity = cluster_activities.value_counts().idxmax()
    canonical_map[cluster_id] = most_common_activity

df['Activity_fixed'] = df['ClusterID'].map(canonical_map)

# Step 7: Calculate detection metrics
if label_column in df.columns:
    y_true = df[label_column].notnull().astype(int)
    y_pred = df['ishomonymous']
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print("=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Precision threshold (≥ 0.6) {'met' if precision >= 0.6 else 'not met'}")
else:
    print("=== Detection Performance Metrics ===")
    print("No labels available for metric calculation.")

# Step 8: Integrity check
clean_modified = df[(df['ishomonymous'] == 0) & (df['BaseActivity'] != df['Activity_fixed'])].shape[0]
homonymous_unchanged = df[(df['ishomonymous'] == 1) & (df['BaseActivity'] == df['Activity_fixed'])].shape[0]
homonymous_corrected = df[(df['ishomonymous'] == 1) & (df['BaseActivity'] != df['Activity_fixed'])].shape[0]

print("=== Integrity Check ===")
print(f"Clean activities modified (should be 0): {clean_modified}")
print(f"Homonymous activities unchanged: {homonymous_unchanged}")
print(f"Homonymous activities corrected: {homonymous_corrected}")

# Step 9: Save output
df[timestamp_column] = pd.to_datetime(df[timestamp_column]).dt.strftime('%Y-%m-%d %H:%M:%S')
columns_to_save = [case_column, timestamp_column, activity_column, 'Activity_fixed', 'ishomonymous']
if label_column in df.columns:
    columns_to_save.append(label_column)

try:
    df.to_csv(output_file, columns=columns_to_save, index=False)
    print(f"Run 1: Processed dataset saved to: {output_file}")
except Exception as e:
    print(f"Error saving file: {e}")
    exit()

# Step 10: Summary statistics
print("=== Summary Statistics ===")
print(f"Total number of events: {df.shape[0]}")
print(f"Number of homonymous events detected: {df['ishomonymous'].sum()}")
print(f"Unique activities before fixing: {len(unique_activities)}")
print(f"Unique activities after fixing: {df['Activity_fixed'].nunique()}")
print(f"Run 1: Final dataset shape: {df.shape}")