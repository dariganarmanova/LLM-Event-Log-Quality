# Generated script for BPIC15-Homonymous - Run 3
# Generated on: 2025-11-13T14:22:37.116501
# Model: deepseek-ai/DeepSeek-V3-0324

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration parameters
input_file = 'data/bpic15/BPIC15-Homonymous.csv'
output_file = 'data/bpic15/bpic15_homonymous_cleaned_run3.csv'
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

# Step 2: Identify Homonymous Activities
df['ishomonymous'] = df[activity_column].str.endswith(homonymous_suffix).astype(int)
df['BaseActivity'] = df[activity_column].str.replace(homonymous_suffix, '', regex=False)

# Step 3: Preprocess Activity Names
def preprocess_activity(activity):
    activity = activity.lower()
    activity = activity.replace(homonymous_suffix, '')
    activity = re.sub(r'[_-]', ' ', activity)
    activity = re.sub(r'\s+', ' ', activity).strip()
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

# Create a mapping from processed activity to cluster label
activity_to_cluster = dict(zip(unique_activities, cluster_labels))
df['Cluster'] = df['ProcessedActivity'].map(activity_to_cluster)

# Step 6: Majority Voting Within Clusters
cluster_to_canonical = {}
for cluster_id in df['Cluster'].unique():
    cluster_activities = df[df['Cluster'] == cluster_id]
    if cluster_activities['ishomonymous'].sum() > 0:
        # Only consider homonymous activities for majority voting
        homonymous_activities = cluster_activities[cluster_activities['ishomonymous'] == 1]
        if not homonymous_activities.empty:
            canonical = homonymous_activities['BaseActivity'].mode()[0]
            cluster_to_canonical[cluster_id] = canonical
        else:
            # If no homonymous activities in cluster, use the most frequent activity
            canonical = cluster_activities['BaseActivity'].mode()[0]
            cluster_to_canonical[cluster_id] = canonical
    else:
        # If no homonymous activities in cluster, use the most frequent activity
        canonical = cluster_activities['BaseActivity'].mode()[0]
        cluster_to_canonical[cluster_id] = canonical

# Assign canonical names
df['Activity_fixed'] = df['Cluster'].map(cluster_to_canonical)

# Ensure clean activities are not modified
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
    print("=== Detection Performance Metrics ===")
    print("Precision: 0.0000")
    print("Recall: 0.0000")
    print("F1-Score: 0.0000")
    print("No labels available for metric calculation.")

# Step 8: Integrity Check
clean_modified = df[(df['ishomonymous'] == 0) & (df['BaseActivity'] != df['Activity_fixed'])].shape[0]
homonymous_unchanged = df[(df['ishomonymous'] == 1) & (df['BaseActivity'] == df['Activity_fixed'])].shape[0]
homonymous_corrected = df[(df['ishomonymous'] == 1) & (df['BaseActivity'] != df['Activity_fixed'])].shape[0]
print("\n=== Integrity Check ===")
print(f"Clean activities modified: {clean_modified} (should be 0)")
print(f"Homonymous activities unchanged: {homonymous_unchanged}")
print(f"Homonymous activities corrected: {homonymous_corrected}")

# Step 9: Save Output
output_columns = [case_column, timestamp_column, activity_column, 'Activity_fixed', 'ishomonymous']
if label_column in df.columns:
    output_columns.append(label_column)
df_output = df[output_columns]
df_output.to_csv(output_file, index=False)

# Step 10: Summary Statistics
print("\n=== Summary Statistics ===")
print(f"Total number of events: {df.shape[0]}")
print(f"Number of homonymous events detected: {df['ishomonymous'].sum()}")
print(f"Unique activities before fixing: {df[activity_column].nunique()}")
print(f"Unique activities after fixing: {df['Activity_fixed'].nunique()}")
print(f"Output file path: {output_file}")
print(f"Run 3: Processed dataset saved to: {output_file}")
print(f"Run 3: Final dataset shape: {df_output.shape}")
print(f"Run 3: Dataset: bpic15")
print(f"Run 3: Task type: homonymous")