# Generated script for Credit-Homonymous - Run 2
# Generated on: 2025-11-13T16:27:40.309898
# Model: deepseek-ai/DeepSeek-V3-0324

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import precision_score, recall_score, f1_score
import re

# Configuration parameters
input_file = 'data/credit/Credit-Homonymous.csv'
output_file = 'data/credit/credit_homonymous_cleaned_run2.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
homonymous_suffix = ':homonymous'
similarity_threshold = 0.8
linkage_method = 'average'

# Load CSV
df = pd.read_csv(input_file)

# Normalize column names (CaseID → Case if needed)
df.columns = df.columns.str.replace('CaseID', 'Case')

# Ensure required columns exist
required_columns = [case_column, activity_column, timestamp_column]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

# Identify Homonymous Activities
df['ishomonymous'] = df[activity_column].str.endswith(homonymous_suffix).astype(int)
df['BaseActivity'] = df[activity_column].str.replace(f'{homonymous_suffix}$', '', regex=True)

# Preprocess Activity Names
def preprocess_activity(activity):
    activity = str(activity).lower()
    activity = re.sub(r'[:].*homonymous$', '', activity)  # Remove homonymous suffix if present
    activity = re.sub(r'[_-]', ' ', activity)  # Replace underscores/hyphens with spaces
    activity = re.sub(r'\s+', ' ', activity).strip()  # Collapse multiple spaces
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
    else:
        return group['BaseActivity'].mode()[0]

canonical_names = df.groupby('cluster_id').apply(get_canonical_name).to_dict()
df['Activity_fixed'] = df['cluster_id'].map(canonical_names)

# Ensure clean activities are not modified
clean_mask = df['ishomonymous'] == 0
df.loc[clean_mask, 'Activity_fixed'] = df.loc[clean_mask, 'BaseActivity']

# Calculate Detection Metrics (if label column exists)
if label_column in df.columns:
    y_true = df[label_column].notna().astype(int)
    y_pred = df['ishomonymous']
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Precision threshold (≥ 0.6) {'met' if precision >= 0.6 else 'not met'}")
else:
    print("=== Detection Performance Metrics ===")
    print("Precision: 0.0000")
    print("Recall: 0.0000")
    print("F1-Score: 0.0000")
    print("No labels available for metric calculation.")

# Integrity Check
clean_modified = sum((clean_mask) & (df['BaseActivity'] != df['Activity_fixed']))
homonymous_unchanged = sum((~clean_mask) & (df['BaseActivity'] == df['Activity_fixed']))
homonymous_corrected = sum((~clean_mask) & (df['BaseActivity'] != df['Activity_fixed']))
print("\n=== Integrity Check ===")
print(f"Clean activities modified (should be 0): {clean_modified}")
print(f"Homonymous activities unchanged: {homonymous_unchanged}")
print(f"Homonymous activities corrected: {homonymous_corrected}")

# Prepare output columns
output_columns = [case_column, timestamp_column]
if 'Resource' in df.columns:
    output_columns.append('Resource')
output_columns.extend([activity_column, 'Activity_fixed', 'ishomonymous'])
if label_column in df.columns:
    output_columns.append(label_column)

# Format timestamp
df[timestamp_column] = pd.to_datetime(df[timestamp_column]).dt.strftime('%Y-%m-%d %H:%M:%S')

# Save Output
df[output_columns].to_csv(output_file, index=False)

# Summary Statistics
print("\n=== Summary Statistics ===")
print(f"Total number of events: {len(df)}")
print(f"Number of homonymous events detected: {df['ishomonymous'].sum()}")
print(f"Unique activities before fixing: {df[activity_column].nunique()}")
print(f"Unique activities after fixing: {df['Activity_fixed'].nunique()}")
print(f"Output file path: {output_file}")

# Required print statements
print(f"\nRun 2: Processed dataset saved to: {output_file}")
print(f"Run 2: Final dataset shape: {df.shape}")
print(f"Run 2: Dataset: credit")
print(f"Run 2: Task type: homonymous")