# Generated script for Credit-Homonymous - Run 1
# Generated on: 2025-11-13T16:26:59.432364
# Model: deepseek-ai/DeepSeek-V3-0324

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import precision_score, recall_score, f1_score
import re

# Configuration
input_file = 'data/credit/Credit-Homonymous.csv'
output_file = 'data/credit/credit_homonymous_cleaned_run1.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
homonymous_suffix = ':homonymous'
similarity_threshold = 0.8
linkage_method = 'average'

# Load data
df = pd.read_csv(input_file)
print(f"Run 1: Original dataset shape: {df.shape}")

# Normalize column names
df.columns = df.columns.str.strip()
df = df.rename(columns=lambda x: 'Case' if x.lower() == 'caseid' else x)

# Check required columns
required_cols = [case_column, activity_column, timestamp_column]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

# Identify homonymous activities
df['ishomonymous'] = df[activity_column].str.endswith(homonymous_suffix).astype(int)
df['BaseActivity'] = df[activity_column].str.replace(fr'{homonymous_suffix}$', '', regex=True)

# Preprocess activity names
def preprocess_activity(activity):
    activity = str(activity).lower()
    activity = re.sub(r'[:].*homonymous', '', activity)  # Remove homonymous suffix if present
    activity = re.sub(r'[_-]', ' ', activity)  # Replace underscores/hyphens with spaces
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

# Create mapping from activity to cluster
activity_to_cluster = dict(zip(unique_activities, cluster_labels))
df['cluster'] = df['ProcessedActivity'].map(activity_to_cluster)

# Majority voting for canonical names
canonical_names = {}
for cluster_id in df['cluster'].unique():
    cluster_activities = df[df['cluster'] == cluster_id]
    if cluster_id == -1:  # Noise cluster (shouldn't happen with our settings)
        continue
    # Get most frequent BaseActivity in cluster (excluding homonymous suffix)
    most_common = cluster_activities['BaseActivity'].value_counts().idxmax()
    canonical_names[cluster_id] = most_common

# Assign canonical names
df['Activity_fixed'] = df['BaseActivity']
df.loc[df['ishomonymous'] == 1, 'Activity_fixed'] = df.loc[df['ishomonymous'] == 1, 'cluster'].map(canonical_names)

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
    print("No labels available for metric calculation.")

# Integrity checks
clean_modified = sum((df['ishomonymous'] == 0) & (df['BaseActivity'] != df['Activity_fixed']))
homonymous_unchanged = sum((df['ishomonymous'] == 1) & (df['BaseActivity'] == df['Activity_fixed']))
homonymous_corrected = sum((df['ishomonymous'] == 1) & (df['BaseActivity'] != df['Activity_fixed']))

print("\n=== Integrity Check Results ===")
print(f"Clean activities modified: {clean_modified} (should be 0)")
print(f"Homonymous activities unchanged: {homonymous_unchanged}")
print(f"Homonymous activities corrected: {homonymous_corrected}")

# Prepare output
output_cols = [case_column, timestamp_column]
if 'Resource' in df.columns:
    output_cols.append('Resource')
output_cols.extend([activity_column, 'Activity_fixed', 'ishomonymous'])
if label_column in df.columns:
    output_cols.append(label_column)

# Format timestamp
df[timestamp_column] = pd.to_datetime(df[timestamp_column]).dt.strftime('%Y-%m-%d %H:%M:%S')

# Save output
df[output_cols].to_csv(output_file, index=False)

# Summary statistics
print("\n=== Summary Statistics ===")
print(f"Total number of events: {len(df)}")
print(f"Number of homonymous events detected: {df['ishomonymous'].sum()}")
print(f"Unique activities before fixing: {df[activity_column].nunique()}")
print(f"Unique activities after fixing: {df['Activity_fixed'].nunique()}")
print(f"Output file path: {output_file}")

# Required run info
print(f"\nRun 1: Processed dataset saved to: {output_file}")
print(f"Run 1: Final dataset shape: {df.shape}")
print(f"Run 1: Dataset: credit")
print(f"Run 1: Task type: homonymous")