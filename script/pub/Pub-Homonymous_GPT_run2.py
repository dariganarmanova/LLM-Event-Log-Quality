# Generated script for Pub-Homonymous - Run 2
# Generated on: 2025-11-14T13:30:42.822837
# Model: gpt-4o-2024-11-20

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import precision_score, recall_score, f1_score
import re

# Configuration parameters
input_file = 'data/pub/Pub-Homonymous.csv'
output_file = 'data/pub/pub_homonymous_cleaned_run2.csv'
similarity_threshold = 0.8
linkage_method = 'average'
homonymous_suffix = ':homonymous'

# Load the data
try:
    df = pd.read_csv(input_file)
    print(f"Run 2: Original dataset shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: File not found at {input_file}")
    exit(1)
except Exception as e:
    print(f"Error loading file: {e}")
    exit(1)

# Ensure required columns exist
required_columns = ['Case', 'Activity', 'Timestamp']
for col in required_columns:
    if col not in df.columns:
        print(f"Error: Missing required column '{col}' in the dataset.")
        exit(1)

# Step 2: Identify homonymous activities
df['ishomonymous'] = df['Activity'].str.endswith(homonymous_suffix).astype(int)
df['BaseActivity'] = df['Activity'].str.replace(homonymous_suffix, '', regex=False)

# Step 3: Preprocess activity names
def preprocess_activity(activity):
    activity = activity.lower()
    activity = re.sub(r'[_\-]', ' ', activity)  # Replace underscores and hyphens with spaces
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
cluster_labels = clustering.fit_predict(activity_vectors)

# Map cluster labels back to activities
cluster_map = dict(zip(unique_activities, cluster_labels))
df['ClusterID'] = df['ProcessedActivity'].map(cluster_map)

# Step 6: Majority voting within clusters
canonical_map = {}
for cluster_id in np.unique(cluster_labels):
    cluster_activities = df[df['ClusterID'] == cluster_id]['BaseActivity']
    canonical_activity = cluster_activities.mode()[0]  # Most frequent activity in the cluster
    canonical_map[cluster_id] = canonical_activity

df['Activity_fixed'] = df.apply(
    lambda row: canonical_map[row['ClusterID']] if row['ishomonymous'] == 1 else row['BaseActivity'],
    axis=1
)

# Step 7: Calculate detection metrics (if label column exists)
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

# Step 8: Integrity check
clean_modified = df[(df['ishomonymous'] == 0) & (df['BaseActivity'] != df['Activity_fixed'])].shape[0]
homonymous_unchanged = df[(df['ishomonymous'] == 1) & (df['BaseActivity'] == df['Activity_fixed'])].shape[0]
homonymous_corrected = df[(df['ishomonymous'] == 1) & (df['BaseActivity'] != df['Activity_fixed'])].shape[0]

print("=== Integrity Check ===")
print(f"Clean activities modified (should be 0): {clean_modified}")
print(f"Homonymous activities unchanged: {homonymous_unchanged}")
print(f"Homonymous activities corrected: {homonymous_corrected}")

# Step 9: Save output
columns_to_save = ['Case', 'Timestamp', 'Activity', 'Activity_fixed', 'ishomonymous']
if 'Resource' in df.columns:
    columns_to_save.insert(2, 'Resource')
if 'label' in df.columns:
    columns_to_save.append('label')

df['Timestamp'] = pd.to_datetime(df['Timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
df[columns_to_save].to_csv(output_file, index=False)

# Step 10: Summary statistics
print("=== Summary Statistics ===")
print(f"Total number of events: {df.shape[0]}")
print(f"Number of homonymous events detected: {df['ishomonymous'].sum()}")
print(f"Unique activities before fixing: {df['Activity'].nunique()}")
print(f"Unique activities after fixing: {df['Activity_fixed'].nunique()}")
print(f"Run 2: Processed dataset saved to: {output_file}")
print(f"Run 2: Final dataset shape: {df.shape}")