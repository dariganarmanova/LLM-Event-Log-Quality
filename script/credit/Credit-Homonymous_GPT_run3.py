# Generated script for Credit-Homonymous - Run 3
# Generated on: 2025-11-13T16:20:34.019683
# Model: gpt-4o-2024-11-20

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration parameters
time_threshold = 2.0
require_same_resource = False
min_matching_events = 2
max_mismatches = 1
activity_suffix_pattern = r"(_signed\d*|_\d+)$"
similarity_threshold = 0.8
linkage_method = 'average'
homonymous_suffix = ':homonymous'

# Input and output file paths
input_file = 'data/credit/Credit-Homonymous.csv'
output_file = 'data/credit/credit_homonymous_cleaned_run3.csv'

# Load the data
try:
    df = pd.read_csv(input_file)
    print(f"Run 3: Original dataset shape: {df.shape}")
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# Ensure required columns exist
required_columns = ['Case', 'Activity', 'Timestamp']
for col in required_columns:
    if col not in df.columns:
        print(f"Missing required column: {col}")
        exit()

# Step 2: Identify homonymous activities
df['ishomonymous'] = df['Activity'].str.endswith(homonymous_suffix).astype(int)
df['BaseActivity'] = df['Activity'].str.replace(homonymous_suffix, '', regex=False)

# Step 3: Preprocess activity names
df['ProcessedActivity'] = df['BaseActivity'].str.lower()
df['ProcessedActivity'] = df['ProcessedActivity'].str.replace(r'[_\-]', ' ', regex=True)
df['ProcessedActivity'] = df['ProcessedActivity'].str.replace(r'\s+', ' ', regex=True).str.strip()

# Step 4: Vectorize activities
unique_activities = df['ProcessedActivity'].unique()
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
activity_vectors = vectorizer.fit_transform(unique_activities).toarray()

# Step 5: Cluster similar activities
try:
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric='cosine',
        linkage=linkage_method,
        distance_threshold=1 - similarity_threshold
    )
    cluster_labels = clustering.fit_predict(activity_vectors)
except Exception as e:
    print(f"Error during clustering: {e}")
    exit()

# Map unique activities to their cluster labels
activity_to_cluster = dict(zip(unique_activities, cluster_labels))
df['ClusterID'] = df['ProcessedActivity'].map(activity_to_cluster)

# Step 6: Majority voting within clusters
cluster_to_canonical = {}
for cluster_id in np.unique(cluster_labels):
    cluster_activities = df[df['ClusterID'] == cluster_id]['BaseActivity']
    canonical_name = cluster_activities.mode().iloc[0]
    cluster_to_canonical[cluster_id] = canonical_name

df['Activity_fixed'] = df.apply(
    lambda row: cluster_to_canonical[row['ClusterID']] if row['ishomonymous'] else row['BaseActivity'], axis=1
)

# Step 7: Calculate detection metrics
if 'label' in df.columns:
    df['y_true'] = df['label'].notnull().astype(int)
    df['y_pred'] = df['ishomonymous']
    precision = precision_score(df['y_true'], df['y_pred'], zero_division=0)
    recall = recall_score(df['y_true'], df['y_pred'], zero_division=0)
    f1 = f1_score(df['y_true'], df['y_pred'], zero_division=0)
    print("=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Precision threshold (≥ 0.6) {'met' if precision >= 0.6 else 'not met'}")
else:
    print("=== Detection Performance Metrics ===")
    print("No labels available for metric calculation.")
    precision = recall = f1 = 0.0

# Step 8: Integrity check
clean_modified = df[(df['ishomonymous'] == 0) & (df['BaseActivity'] != df['Activity_fixed'])].shape[0]
unchanged_homonymous = df[(df['ishomonymous'] == 1) & (df['BaseActivity'] == df['Activity_fixed'])].shape[0]
corrected_homonymous = df[(df['ishomonymous'] == 1) & (df['BaseActivity'] != df['Activity_fixed'])].shape[0]

print("=== Integrity Check ===")
print(f"Clean activities modified: {clean_modified}")
print(f"Homonymous activities unchanged: {unchanged_homonymous}")
print(f"Homonymous activities corrected: {corrected_homonymous}")

# Step 9: Save output
output_columns = ['Case', 'Timestamp', 'Activity', 'Activity_fixed', 'ishomonymous']
if 'Resource' in df.columns:
    output_columns.insert(2, 'Resource')
if 'label' in df.columns:
    output_columns.append('label')

df['Timestamp'] = pd.to_datetime(df['Timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
df[output_columns].to_csv(output_file, index=False)

# Step 10: Summary statistics
print("=== Summary Statistics ===")
print(f"Total number of events: {df.shape[0]}")
print(f"Number of homonymous events detected: {df['ishomonymous'].sum()}")
print(f"Unique activities before fixing: {len(df['Activity'].unique())}")
print(f"Unique activities after fixing: {len(df['Activity_fixed'].unique())}")
print(f"Run 3: Processed dataset saved to: {output_file}")
print(f"Run 3: Final dataset shape: {df.shape}")