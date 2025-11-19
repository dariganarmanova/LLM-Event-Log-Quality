# Generated script for Pub-Homonymous - Run 2
# Generated on: 2025-11-14T13:32:49.896296
# Model: deepseek-ai/DeepSeek-V3-0324

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import precision_score, recall_score, f1_score
import re

# Configuration
input_file = 'data/pub/Pub-Homonymous.csv'
output_file = 'data/pub/pub_homonymous_cleaned_run2.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
homonymous_suffix = ':homonymous'
similarity_threshold = 0.8
linkage_method = 'average'

# Load CSV
df = pd.read_csv(input_file)
print(f"Run 2: Original dataset shape: {df.shape}")

# Normalize column names (CaseID -> Case if needed)
if 'CaseID' in df.columns and case_column not in df.columns:
    df.rename(columns={'CaseID': case_column}, inplace=True)

# Check required columns
required_columns = [case_column, activity_column, timestamp_column]
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' not found in input file")

# Identify homonymous activities
df['ishomonymous'] = df[activity_column].str.endswith(homonymous_suffix).astype(int)
df['BaseActivity'] = df[activity_column].str.replace(fr'{homonymous_suffix}$', '', regex=True)

# Preprocess activity names
def preprocess_activity(activity):
    activity = str(activity).lower()
    activity = re.sub(r'[:].*homonymous.*$', '', activity)  # Remove homonymous suffix if present
    activity = re.sub(r'[-_]', ' ', activity)  # Replace underscores/hyphens with spaces
    activity = re.sub(r'\s+', ' ', activity).strip()  # Normalize whitespace
    return activity

df['ProcessedActivity'] = df[activity_column].apply(preprocess_activity)

# Vectorize unique activities
unique_activities = df['ProcessedActivity'].unique()
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
X = vectorizer.fit_transform(unique_activities).toarray()

# Cluster similar activities
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

# Majority voting within clusters to determine canonical names
cluster_groups = df[df['ishomonymous'] == 1].groupby('cluster_id')
cluster_majority = cluster_groups['BaseActivity'].agg(lambda x: x.mode()[0] if len(x) > 0 else None)
cluster_majority = cluster_majority.to_dict()

# Assign canonical names
def get_canonical_name(row):
    if row['ishomonymous'] == 1:
        return cluster_majority.get(row['cluster_id'], row['BaseActivity'])
    return row['BaseActivity']

df['Activity_fixed'] = df.apply(get_canonical_name, axis=1)

# Calculate detection metrics if label column exists
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

# Integrity check
clean_modified = sum((df['ishomonymous'] == 0) & (df['BaseActivity'] != df['Activity_fixed']))
homonymous_unchanged = sum((df['ishomonymous'] == 1) & (df['BaseActivity'] == df['Activity_fixed']))
homonymous_corrected = sum((df['ishomonymous'] == 1) & (df['BaseActivity'] != df['Activity_fixed']))
print("\n=== Integrity Check ===")
print(f"Clean activities modified: {clean_modified} (should be 0)")
print(f"Homonymous activities unchanged: {homonymous_unchanged}")
print(f"Homonymous activities corrected: {homonymous_corrected}")

# Prepare output columns
output_columns = [case_column, timestamp_column]
if 'Resource' in df.columns:
    output_columns.append('Resource')
output_columns.extend([activity_column, 'Activity_fixed', 'ishomonymous'])
if label_column in df.columns:
    output_columns.append(label_column)

# Save output
df_output = df[output_columns].copy()
df_output[timestamp_column] = pd.to_datetime(df_output[timestamp_column]).dt.strftime('%Y-%m-%d %H:%M:%S')
df_output.to_csv(output_file, index=False)

# Summary statistics
unique_before = df[activity_column].nunique()
unique_after = df['Activity_fixed'].nunique()
print("\n=== Summary Statistics ===")
print(f"Total number of events: {len(df)}")
print(f"Number of homonymous events detected: {df['ishomonymous'].sum()}")
print(f"Unique activities before fixing: {unique_before}")
print(f"Unique activities after fixing: {unique_after}")
print(f"Output file path: {output_file}")

# Required run info
print(f"\nRun 2: Processed dataset saved to: {output_file}")
print(f"Run 2: Final dataset shape: {df_output.shape}")
print(f"Run 2: Dataset: pub")
print(f"Run 2: Task type: homonymous")