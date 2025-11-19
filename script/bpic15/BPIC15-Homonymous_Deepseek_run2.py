# Generated script for BPIC15-Homonymous - Run 2
# Generated on: 2025-11-13T14:21:56.133859
# Model: deepseek-ai/DeepSeek-V3-0324

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import precision_score, recall_score, f1_score
import re

# Configuration parameters
input_file = 'data/bpic15/BPIC15-Homonymous.csv'
output_file = 'data/bpic15/bpic15_homonymous_cleaned_run2.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
homonymous_suffix = ':homonymous'
similarity_threshold = 0.5
linkage_method = 'average'

# Load CSV
df = pd.read_csv(input_file)
print(f"Run 2: Original dataset shape: {df.shape}")

# Ensure required columns exist
required_columns = [case_column, activity_column, timestamp_column]
if not all(col in df.columns for col in required_columns):
    raise ValueError("Missing required columns in the input file")

# Identify Homonymous Activities
df['ishomonymous'] = df[activity_column].str.endswith(homonymous_suffix).astype(int)
df['BaseActivity'] = df[activity_column].str.replace(f'{homonymous_suffix}$', '', regex=True)

# Preprocess Activity Names
def preprocess_activity(activity):
    activity = activity.lower()
    activity = re.sub(f'{homonymous_suffix}$', '', activity)
    activity = re.sub(r'[_-]', ' ', activity)
    activity = re.sub(r'\s+', ' ', activity).strip()
    return activity

df['ProcessedActivity'] = df[activity_column].apply(preprocess_activity)

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
df['cluster_id'] = df['ProcessedActivity'].map(activity_to_cluster)

# Majority Voting Within Clusters
def get_canonical_name(group):
    if group['ishomonymous'].sum() == 0:
        return group['BaseActivity'].iloc[0]
    most_common = group['BaseActivity'].mode()
    if len(most_common) > 0:
        return most_common[0]
    return group['BaseActivity'].iloc[0]

canonical_names = df.groupby('cluster_id').apply(get_canonical_name).to_dict()
df['Activity_fixed'] = df['cluster_id'].map(canonical_names)

# Only modify homonymous activities
df.loc[df['ishomonymous'] == 0, 'Activity_fixed'] = df.loc[df['ishomonymous'] == 0, 'BaseActivity']

# Calculate Detection Metrics (if label column exists)
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
    print("Precision: 0.0000")
    print("Recall: 0.0000")
    print("F1-Score: 0.0000")
    print("No labels available for metric calculation.")

# Integrity Check
clean_modified = sum((df['ishomonymous'] == 0) & (df['BaseActivity'] != df['Activity_fixed']))
homonymous_unchanged = sum((df['ishomonymous'] == 1) & (df['BaseActivity'] == df['Activity_fixed']))
homonymous_corrected = sum((df['ishomonymous'] == 1) & (df['BaseActivity'] != df['Activity_fixed']))
print("\n=== Integrity Check ===")
print(f"Clean activities modified: {clean_modified} (should be 0)")
print(f"Homonymous activities unchanged: {homonymous_unchanged}")
print(f"Homonymous activities corrected: {homonymous_corrected}")

# Save Output
output_columns = [case_column, timestamp_column, 'Resource', activity_column, 'Activity_fixed', 'ishomonymous']
if label_column in df.columns:
    output_columns.append(label_column)
df_output = df[output_columns].copy()
df_output[timestamp_column] = pd.to_datetime(df_output[timestamp_column]).dt.strftime('%Y-%m-%d %H:%M:%S')
df_output.to_csv(output_file, index=False)

# Summary Statistics
unique_activities_before = df[activity_column].nunique()
unique_activities_after = df['Activity_fixed'].nunique()
print("\n=== Summary Statistics ===")
print(f"Total number of events: {len(df)}")
print(f"Number of homonymous events detected: {df['ishomonymous'].sum()}")
print(f"Unique activities before fixing: {unique_activities_before}")
print(f"Unique activities after fixing: {unique_activities_after}")
print(f"Output file path: {output_file}")
print(f"Run 2: Processed dataset saved to: {output_file}")
print(f"Run 2: Final dataset shape: {df_output.shape}")
print(f"Run 2: Dataset: bpic15")
print(f"Run 2: Task type: homonymous")