# Generated script for Pub-Homonymous - Run 3
# Generated on: 2025-11-14T13:31:10.717668
# Model: gpt-4o-2024-11-20

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import precision_score, recall_score, f1_score
import re

# Configuration parameters
time_threshold = 2.0
require_same_resource = False
min_matching_events = 2
max_mismatches = 1
activity_suffix_pattern = r"(_signed\d*|_\d+)$"
similarity_threshold = 0.8
linkage_method = 'average'
homonymous_suffix = ':homonymous'

# File paths
input_file = 'data/pub/Pub-Homonymous.csv'
output_file = 'data/pub/pub_homonymous_cleaned_run3.csv'

# Load the data
try:
    df = pd.read_csv(input_file)
    print(f"Run 3: Original dataset shape: {df.shape}")
except Exception as e:
    print(f"Error loading input file: {e}")
    exit()

# Ensure required columns exist
required_columns = ['Case', 'Activity', 'Timestamp']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# Step 2: Identify homonymous activities
df['ishomonymous'] = df['Activity'].str.endswith(homonymous_suffix).astype(int)
df['BaseActivity'] = df['Activity'].str.replace(homonymous_suffix, '', regex=False)

# Step 3: Preprocess activity names
def preprocess_activity(activity):
    activity = activity.lower()  # Convert to lowercase
    activity = re.sub(homonymous_suffix, '', activity)  # Remove homonymous suffix
    activity = re.sub(r'[_\-]+', ' ', activity)  # Replace underscores/hyphens with spaces
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
clusters = clustering.fit_predict(activity_vectors)

# Map unique activities to their cluster IDs
activity_to_cluster = dict(zip(unique_activities, clusters))
df['ClusterID'] = df['ProcessedActivity'].map(activity_to_cluster)

# Step 6: Majority voting within clusters
cluster_to_canonical = {}
for cluster_id in np.unique(clusters):
    cluster_activities = df[df['ClusterID'] == cluster_id]['BaseActivity']
    most_common_activity = cluster_activities.value_counts().idxmax()
    cluster_to_canonical[cluster_id] = most_common_activity

df['Activity_fixed'] = df.apply(
    lambda row: cluster_to_canonical[row['ClusterID']] if row['ishomonymous'] == 1 else row['BaseActivity'],
    axis=1
)

# Step 7: Calculate detection metrics
if 'label' in df.columns:
    y_true = df['label'].notnull().astype(int)
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
    print("Precision: 0.0000")
    print("Recall: 0.0000")
    print("F1-Score: 0.0000")

# Step 8: Integrity check
clean_activities_modified = df[(df['ishomonymous'] == 0) & (df['BaseActivity'] != df['Activity_fixed'])].shape[0]
unchanged_homonymous = df[(df['ishomonymous'] == 1) & (df['BaseActivity'] == df['Activity_fixed'])].shape[0]
corrected_homonymous = df[(df['ishomonymous'] == 1) & (df['BaseActivity'] != df['Activity_fixed'])].shape[0]

print("=== Integrity Check ===")
print(f"Clean activities modified: {clean_activities_modified}")
print(f"Unchanged homonymous activities: {unchanged_homonymous}")
print(f"Corrected homonymous activities: {corrected_homonymous}")

# Step 9: Save output
output_columns = ['Case', 'Timestamp', 'Activity', 'Activity_fixed', 'ishomonymous']
if 'Resource' in df.columns:
    output_columns.append('Resource')
if 'label' in df.columns:
    output_columns.append('label')

df['Timestamp'] = pd.to_datetime(df['Timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
df[output_columns].to_csv(output_file, index=False)

# Step 10: Summary statistics
print("=== Summary Statistics ===")
print(f"Total number of events: {df.shape[0]}")
print(f"Number of homonymous events detected: {df['ishomonymous'].sum()}")
print(f"Unique activities before fixing: {df['Activity'].nunique()}")
print(f"Unique activities after fixing: {df['Activity_fixed'].nunique()}")
print(f"Run 3: Processed dataset saved to: {output_file}")
print(f"Run 3: Final dataset shape: {df.shape}")