# Generated script for BPIC11-Homonymous - Run 2
# Generated on: 2025-11-13T11:42:46.608448
# Model: deepseek-ai/DeepSeek-V3-0324

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import precision_score, recall_score, f1_score
import re

# Configuration parameters
input_file = 'data/bpic11/BPIC11-Homonymous.csv'
output_file = 'data/bpic11/bpic11_homonymous_cleaned_run2.csv'
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

# Normalize column names if needed
df.rename(columns={'CaseID': 'Case'}, inplace=True, errors='ignore')

# Identify Homonymous Activities
df['ishomonymous'] = df[activity_column].str.endswith(homonymous_suffix).astype(int)
df['BaseActivity'] = df[activity_column].str.replace(f'{homonymous_suffix}$', '', regex=True)

# Preprocess Activity Names
df['ProcessedActivity'] = df['BaseActivity'].str.lower()
df['ProcessedActivity'] = df['ProcessedActivity'].str.replace(r'[_-]', ' ', regex=True)
df['ProcessedActivity'] = df['ProcessedActivity'].str.replace(r'\s+', ' ', regex=True).str.strip()

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
clusters = clustering.fit_predict(X)

# Create cluster mapping
activity_to_cluster = dict(zip(unique_activities, clusters))
df['cluster'] = df['ProcessedActivity'].map(activity_to_cluster)

# Majority Voting Within Clusters
cluster_to_canonical = {}
for cluster_id in df['cluster'].unique():
    cluster_activities = df[df['cluster'] == cluster_id]
    if cluster_activities['ishomonymous'].sum() > 0:  # Only consider homonymous activities for majority voting
        homonymous_in_cluster = cluster_activities[cluster_activities['ishomonymous'] == 1]
        if not homonymous_in_cluster.empty:
            most_common = homonymous_in_cluster['BaseActivity'].mode()[0]
            cluster_to_canonical[cluster_id] = most_common.lower().replace('_', ' ').replace('-', ' ').strip()
        else:
            cluster_to_canonical[cluster_id] = cluster_activities['BaseActivity'].iloc[0].lower().replace('_', ' ').replace('-', ' ').strip()
    else:
        cluster_to_canonical[cluster_id] = cluster_activities['BaseActivity'].iloc[0].lower().replace('_', ' ').replace('-', ' ').strip()

# Assign canonical names
df['Activity_fixed'] = df['cluster'].map(cluster_to_canonical)
df.loc[df['ishomonymous'] == 0, 'Activity_fixed'] = df.loc[df['ishomonymous'] == 0, 'BaseActivity']

# Integrity Check
clean_modified = sum((df['ishomonymous'] == 0) & (df['BaseActivity'] != df['Activity_fixed']))
homonymous_unchanged = sum((df['ishomonymous'] == 1) & (df['BaseActivity'] == df['Activity_fixed']))
homonymous_corrected = sum((df['ishomonymous'] == 1) & (df['BaseActivity'] != df['Activity_fixed']))

print("=== Integrity Check ===")
print(f"Clean activities modified: {clean_modified} (should be 0)")
print(f"Homonymous activities unchanged: {homonymous_unchanged}")
print(f"Homonymous activities corrected: {homonymous_corrected}")

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
    print("No labels available for metric calculation.")

# Save Output
output_columns = [case_column, timestamp_column, 'Resource', activity_column, 'Activity_fixed', 'ishomonymous']
if label_column in df.columns:
    output_columns.append(label_column)
df_output = df[output_columns]
df_output.to_csv(output_file, index=False)

# Summary Statistics
print("\n=== Summary Statistics ===")
print(f"Total number of events: {len(df)}")
print(f"Number of homonymous events detected: {df['ishomonymous'].sum()}")
print(f"Unique activities before fixing: {df[activity_column].nunique()}")
print(f"Unique activities after fixing: {df['Activity_fixed'].nunique()}")
print(f"Output file path: {output_file}")

# Required prints
print(f"\nRun 2: Processed dataset saved to: {output_file}")
print(f"Run 2: Final dataset shape: {df.shape}")
print(f"Run 2: Dataset: bpic11")
print(f"Run 2: Task type: homonymous")