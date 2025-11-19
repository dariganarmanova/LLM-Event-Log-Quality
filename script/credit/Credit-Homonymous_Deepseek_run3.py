# Generated script for Credit-Homonymous - Run 3
# Generated on: 2025-11-13T16:28:20.912818
# Model: deepseek-ai/DeepSeek-V3-0324

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

# Configuration parameters
input_file = 'data/credit/Credit-Homonymous.csv'
output_file = 'data/credit/credit_homonymous_cleaned_run3.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
homonymous_suffix = ':homonymous'
similarity_threshold = 0.8
linkage_method = 'average'

# Step 1: Load CSV
df = pd.read_csv(input_file)
print(f"Run 3: Original dataset shape: {df.shape}")

# Normalize column names if needed
df.columns = df.columns.str.strip()
if 'CaseID' in df.columns and case_column not in df.columns:
    df.rename(columns={'CaseID': case_column}, inplace=True)

# Step 2: Identify Homonymous Activities
df['ishomonymous'] = df[activity_column].str.endswith(homonymous_suffix).astype(int)
df['BaseActivity'] = df[activity_column].str.replace(f'{homonymous_suffix}$', '', regex=True)

# Step 3: Preprocess Activity Names
def preprocess_activity(activity):
    activity = str(activity).lower()
    activity = re.sub(r'(_signed\d*|_\d+)$', '', activity)  # Remove suffix pattern
    activity = re.sub(r'[:_-]', ' ', activity)  # Replace special chars with space
    activity = re.sub(r'\s+', ' ', activity).strip()  # Normalize whitespace
    return activity

df['ProcessedActivity'] = df['BaseActivity'].apply(preprocess_activity)

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

# Create mapping from processed activity to cluster ID
activity_to_cluster = dict(zip(unique_activities, cluster_labels))
df['cluster_id'] = df['ProcessedActivity'].map(activity_to_cluster)

# Step 6: Majority Voting Within Clusters
activity_fixed = df['BaseActivity'].copy()

for cluster_id in df['cluster_id'].unique():
    cluster_mask = df['cluster_id'] == cluster_id
    homonymous_mask = df['ishomonymous'] == 1
    cluster_activities = df.loc[cluster_mask & homonymous_mask, 'BaseActivity']
    
    if len(cluster_activities) > 0:
        most_common = cluster_activities.mode()[0]
        activity_fixed.loc[cluster_mask & homonymous_mask] = most_common

df['Activity_fixed'] = activity_fixed

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
    print("=== Detection Performance Metrics ===")
    print("No labels available for metric calculation.")

# Step 8: Integrity Check
clean_modified = sum((df['ishomonymous'] == 0) & (df['BaseActivity'] != df['Activity_fixed']))
homonymous_unchanged = sum((df['ishomonymous'] == 1) & (df['BaseActivity'] == df['Activity_fixed']))
homonymous_corrected = sum((df['ishomonymous'] == 1) & (df['BaseActivity'] != df['Activity_fixed']))

print("\n=== Integrity Check ===")
print(f"Clean activities modified: {clean_modified} (should be 0)")
print(f"Homonymous activities unchanged: {homonymous_unchanged}")
print(f"Homonymous activities corrected: {homonymous_corrected}")

# Step 9: Save Output
output_columns = [case_column, timestamp_column]
if 'Resource' in df.columns:
    output_columns.append('Resource')
output_columns.extend([activity_column, 'Activity_fixed', 'ishomonymous'])
if label_column in df.columns:
    output_columns.append(label_column)

df_output = df[output_columns].copy()
df_output[timestamp_column] = pd.to_datetime(df_output[timestamp_column]).dt.strftime('%Y-%m-%d %H:%M:%S')
df_output.to_csv(output_file, index=False)

# Step 10: Summary Statistics
print("\n=== Summary Statistics ===")
print(f"Total number of events: {len(df)}")
print(f"Number of homonymous events detected: {df['ishomonymous'].sum()}")
print(f"Unique activities before fixing: {df[activity_column].nunique()}")
print(f"Unique activities after fixing: {df['Activity_fixed'].nunique()}")
print(f"Output file path: {output_file}")

# Required prints
print(f"\nRun 3: Processed dataset saved to: {output_file}")
print(f"Run 3: Final dataset shape: {df_output.shape}")
print(f"Run 3: Dataset: credit")
print(f"Run 3: Task type: homonymous")