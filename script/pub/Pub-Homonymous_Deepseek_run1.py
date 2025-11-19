# Generated script for Pub-Homonymous - Run 1
# Generated on: 2025-11-14T13:32:20.320682
# Model: deepseek-ai/DeepSeek-V3-0324

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import precision_score, recall_score, f1_score
import re

# Configuration parameters
input_file = 'data/pub/Pub-Homonymous.csv'
output_file = 'data/pub/pub_homonymous_cleaned_run1.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
homonymous_suffix = ':homonymous'
similarity_threshold = 0.8
linkage_method = 'average'

# Load CSV
df = pd.read_csv(input_file)
print(f"Run 1: Original dataset shape: {df.shape}")

# Normalize column names if needed
df.columns = df.columns.str.strip()
if 'CaseID' in df.columns and case_column not in df.columns:
    df.rename(columns={'CaseID': case_column}, inplace=True)

# Check required columns
required_columns = {case_column, activity_column, timestamp_column}
if not required_columns.issubset(df.columns):
    missing = required_columns - set(df.columns)
    raise ValueError(f"Missing required columns: {missing}")

# Identify Homonymous Activities
df['ishomonymous'] = df[activity_column].str.endswith(homonymous_suffix).astype(int)
df['BaseActivity'] = df[activity_column].str.replace(homonymous_suffix, '', regex=False)

# Preprocess Activity Names
def preprocess_activity(activity):
    activity = str(activity).lower()
    activity = re.sub(r'(_signed\d*|_\d+)$', '', activity)  # Remove numeric suffixes
    activity = re.sub(r'[-_]', ' ', activity)  # Replace underscores/hyphens with spaces
    activity = re.sub(r'\s+', ' ', activity).strip()  # Collapse multiple spaces
    return activity

df['ProcessedActivity'] = df['BaseActivity'].apply(preprocess_activity)

# Vectorize Activities
unique_activities = df['ProcessedActivity'].unique()
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
X = vectorizer.fit_transform(unique_activities).toarray()

# Cluster Similar Activities
clustering = AgglomerativeClustering(
    n_clusters=None,
    metric='cosine',
    linkage=linkage_method,
    distance_threshold=1 - similarity_threshold
)
cluster_labels = clustering.fit_predict(X)

# Create mapping from processed activity to cluster ID
activity_to_cluster = dict(zip(unique_activities, cluster_labels))
df['ClusterID'] = df['ProcessedActivity'].map(activity_to_cluster)

# Majority Voting Within Clusters
cluster_to_canonical = {}
for cluster_id in df['ClusterID'].unique():
    cluster_activities = df[df['ClusterID'] == cluster_id]
    if len(cluster_activities) > 0:
        # Get the most frequent BaseActivity in the cluster
        canonical = cluster_activities['BaseActivity'].mode()[0]
        cluster_to_canonical[cluster_id] = canonical

# Assign canonical names
df['Activity_fixed'] = df['ClusterID'].map(cluster_to_canonical)
# Only update homonymous activities
df.loc[df['ishomonymous'] == 0, 'Activity_fixed'] = df.loc[df['ishomonymous'] == 0, 'BaseActivity']

# Calculate Detection Metrics
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
    print("=== Detection Performance Metrics ===")
    print("No labels available for metric calculation.")

# Integrity Check
clean_modified = df[(df['ishomonymous'] == 0) & (df['BaseActivity'] != df['Activity_fixed'])].shape[0]
homonymous_unchanged = df[(df['ishomonymous'] == 1) & (df['BaseActivity'] == df['Activity_fixed'])].shape[0]
homonymous_corrected = df[(df['ishomonymous'] == 1) & (df['BaseActivity'] != df['Activity_fixed'])].shape[0]

print("\n=== Integrity Check ===")
print(f"Clean activities modified: {clean_modified} (should be 0)")
print(f"Homonymous activities unchanged: {homonymous_unchanged}")
print(f"Homonymous activities corrected: {homonymous_corrected}")

# Save Output
output_columns = [case_column, timestamp_column, activity_column, 'Activity_fixed', 'ishomonymous']
if 'Resource' in df.columns:
    output_columns.append('Resource')
if label_column in df.columns:
    output_columns.append(label_column)

df_output = df[output_columns].copy()
df_output[timestamp_column] = pd.to_datetime(df_output[timestamp_column]).dt.strftime('%Y-%m-%d %H:%M:%S')
df_output.to_csv(output_file, index=False)

# Summary Statistics
print("\n=== Summary Statistics ===")
print(f"Total number of events: {len(df)}")
print(f"Number of homonymous events detected: {df['ishomonymous'].sum()}")
print(f"Unique activities before fixing: {df[activity_column].nunique()}")
print(f"Unique activities after fixing: {df['Activity_fixed'].nunique()}")
print(f"Output file path: {output_file}")

# REQUIRED: Print summary
print(f"\nRun 1: Processed dataset saved to: {output_file}")
print(f"Run 1: Final dataset shape: {df_output.shape}")
print(f"Run 1: Dataset: pub")
print(f"Run 1: Task type: homonymous")