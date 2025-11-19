# Generated script for Pub-Homonymous - Run 3
# Generated on: 2025-11-14T13:30:04.122119
# Model: meta-llama/Llama-3.1-8B-Instruct

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import normalize
import numpy as np
import re

# Configuration parameters
time_threshold = 2.0
require_same_resource = False
min_matching_events = 2
max_mismatches = 1
activity_suffix_pattern = r"(_signed\d*|_\d+)$"
similarity_threshold = 0.8
case_sensitive = False
use_fuzzy_matching = False
linkage_method = 'average'

# Load the data
input_file = 'data/pub/Pub-Homonymous.csv'
output_file = 'data/pub/pub_homonymous_cleaned_run3.csv'
df = pd.read_csv(input_file)

# Ensure required columns exist
required_columns = ['Case', 'Activity', 'Timestamp']
if not all(col in df.columns for col in required_columns):
    raise ValueError("Missing required columns")

# Normalize CaseID → Case if needed (handle naming variations)
df['Case'] = df['Case'].str.lower()

# Identify Homonymous Activities
homonymous_suffix = ':homonymous'
df['ishomonymous'] = df['Activity'].str.endswith(homonymous_suffix)
df['BaseActivity'] = df['Activity'].str.replace(homonymous_suffix, '')

# Preprocess Activity Names
def preprocess_activity(activity):
    activity = activity.lower()
    activity = re.sub(r'[_\-]', ' ', activity)
    activity = re.sub(r'\s+', ' ', activity)
    return activity.strip()

df['ProcessedActivity'] = df['BaseActivity'].apply(preprocess_activity)

# Vectorize Activities
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1,3))
unique_activities = df['ProcessedActivity'].unique()
vectorized_activities = vectorizer.fit_transform(unique_activities)
vectorized_activities = normalize(vectorized_activities).toarray()

# Cluster Similar Activities
clustering = AgglomerativeClustering(
    n_clusters=None,
    metric='cosine',
    linkage=linkage_method,
    distance_threshold=1 - similarity_threshold
)
cluster_labels = clustering.fit_predict(vectorized_activities)

# Majority Voting Within Clusters
activity_fixed = []
for cluster_label in np.unique(cluster_labels):
    cluster_activities = unique_activities[cluster_labels == cluster_label]
    cluster_base_activities = df['BaseActivity'][cluster_labels == cluster_label]
    most_frequent_base_activity = pd.Series(cluster_base_activities).value_counts().index[0]
    activity_fixed.extend([most_frequent_base_activity] * len(cluster_activities))

# Update Activity_fixed column
df['Activity_fixed'] = activity_fixed

# Calculate Detection Metrics
label_column = 'label'
if label_column in df.columns:
    y_true = df[label_column].notnull().astype(int)
    y_pred = df['ishomonymous'].astype(int)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Precision threshold (≥ 0.6) met: {precision >= 0.6}")
else:
    print("No labels available for metric calculation.")

# Integrity Check
clean_activities = df[df['ishomonymous'] == 0]
unchanged_homonymous = clean_activities[clean_activities['BaseActivity'] == clean_activities['Activity_fixed']].shape[0]
corrected_homonymous = clean_activities[clean_activities['BaseActivity'] != clean_activities['Activity_fixed']].shape[0]
print(f"Integrity Check:")
print(f"Clean activities modified: {unchanged_homonymous}")
print(f"Homonymous activities unchanged: {corrected_homonymous}")

# Save Output
output_columns = ['Case', 'Timestamp', 'Resource', 'Activity', 'Activity_fixed', 'ishomonymous']
if label_column in df.columns:
    output_columns.append(label_column)
df_output = df[output_columns]
df_output['Timestamp'] = pd.to_datetime(df_output['Timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
df_output.to_csv(output_file, index=False)

# Summary Statistics
print(f"Run 3: Processed dataset saved to: {output_file}")
print(f"Run 3: Final dataset shape: {df.shape}")
print(f"Run 3: Dataset: pub")
print(f"Run 3: Task type: homonymous")