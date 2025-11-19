# Generated script for BPIC15-Homonymous - Run 1
# Generated on: 2025-11-13T14:21:14.539999
# Model: deepseek-ai/DeepSeek-V3-0324

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

# Step 1: Load CSV
df = pd.read_csv(input_file)
print(f"Run 1: Original dataset shape: {df.shape}")

# Ensure required columns exist
required_columns = [case_column, activity_column, timestamp_column]
if not all(col in df.columns for col in required_columns):
    raise ValueError(f"Missing required columns. Expected: {required_columns}")

# Step 2: Identify Homonymous Activities
df['ishomonymous'] = df[activity_column].str.endswith(homonymous_suffix).astype(int)
df['BaseActivity'] = df[activity_column].str.replace(homonymous_suffix, '', regex=False)

# Step 3: Preprocess Activity Names
def preprocess_activity(activity):
    activity = activity.lower()
    activity = re.sub(r'[:].*$', '', activity)  # Remove any suffix
    activity = re.sub(r'[_-]', ' ', activity)  # Replace underscores/hyphens with spaces
    activity = re.sub(r'\s+', ' ', activity).strip()  # Normalize whitespace
    return activity

df['ProcessedActivity'] = df[activity_column].apply(preprocess_activity)

# Step 4: Vectorize Activities
unique_activities = df['ProcessedActivity'].unique()
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
X = vectorizer.fit_transform(unique_activities).toarray()

# Step 5: Cluster Similar Activities
clustering = AgglomerativeClustering(
    n_clusters=None,
    metric='cosine',
    linkage=linkage_method,
    distance_threshold=1 - similarity_threshold
)
cluster_labels = clustering.fit_predict(X)

# Create a mapping from processed activity to cluster ID
activity_to_cluster = dict(zip(unique_activities, cluster_labels))
df['ClusterID'] = df['ProcessedActivity'].map(activity_to_cluster)

# Step 6: Majority Voting Within Clusters
# For each cluster, find the most frequent BaseActivity (before preprocessing)
cluster_to_canonical = {}
for cluster_id in df['ClusterID'].unique():
    cluster_activities = df[df['ClusterID'] == cluster_id]['BaseActivity']
    most_common = cluster_activities.mode()[0]
    cluster_to_canonical[cluster_id] = most_common

# Assign canonical names
df['Activity_fixed'] = df['ClusterID'].map(cluster_to_canonical)

# Only update homonymous activities
df.loc[df['ishomonymous'] == 0, 'Activity_fixed'] = df.loc[df['ishomonymous'] == 0, 'BaseActivity']

# Step 7: Calculate Detection Metrics (if label column exists)
if label_column in df.columns:
    y_true = df[label_column].notna().astype(int)
    y_pred = df['ishomonymous']
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print("=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("Precision threshold (≥ 0.6) met" if precision >= 0.6 else "Precision threshold (≥ 0.6) not met")
else:
    print("No labels available for metric calculation.")

# Step 8: Integrity Check
clean_modified = sum((df['ishomonymous'] == 0) & (df['BaseActivity'] != df['Activity_fixed']))
homonymous_unchanged = sum((df['ishomonymous'] == 1) & (df['BaseActivity'] == df['Activity_fixed']))
homonymous_corrected = sum((df['ishomonymous'] == 1) & (df['BaseActivity'] != df['Activity_fixed']))
print("\n=== Integrity Check ===")
print(f"Clean activities modified (should be 0): {clean_modified}")
print(f"Homonymous activities unchanged: {homonymous_unchanged}")
print(f"Homonymous activities corrected: {homonymous_corrected}")

# Step 9: Save Output
output_columns = [case_column, timestamp_column, 'Resource', activity_column, 'Activity_fixed', 'ishomonymous']
if label_column in df.columns:
    output_columns.append(label_column)
df_output = df[output_columns]
df_output.to_csv(output_file, index=False)

# Step 10: Summary Statistics
print("\n=== Summary Statistics ===")
print(f"Total number of events: {len(df)}")
print(f"Number of homonymous events detected: {df['ishomonymous'].sum()}")
print(f"Unique activities before fixing: {df[activity_column].nunique()}")
print(f"Unique activities after fixing: {df['Activity_fixed'].nunique()}")
print(f"Output file path: {output_file}")

print(f"\nRun 1: Processed dataset saved to: {output_file}")
print(f"Run 1: Final dataset shape: {df.shape}")
print(f"Run 1: Dataset: bpic15")
print(f"Run 1: Task type: homonymous")