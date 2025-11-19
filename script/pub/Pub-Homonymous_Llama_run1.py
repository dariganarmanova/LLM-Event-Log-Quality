# Generated script for Pub-Homonymous - Run 1
# Generated on: 2025-11-14T13:29:59.507462
# Model: meta-llama/Llama-3.1-8B-Instruct

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import normalize
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

# Load the data
input_file = 'data/pub/Pub-Homonymous.csv'
output_file = 'data/pub/pub_homonymous_cleaned_run1.csv'
df = pd.read_csv(input_file)

# Ensure required columns exist
required_columns = ['Case', 'Activity', 'Timestamp']
for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing required column: {column}")

# Normalize CaseID → Case if needed (handle naming variations)
df['Case'] = df['Case'].str.lower()

# Identify Homonymous Activities
homonymous_suffix = ':homonymous'
df['ishomonymous'] = df['Activity'].str.endswith(homonymous_suffix)
df['BaseActivity'] = df['Activity'].str.replace(homonymous_suffix, '')

# Preprocess Activity Names
def preprocess_activity(activity):
    activity = activity.lower()
    activity = re.sub(r'_\d+', ' ', activity)
    activity = re.sub(r'\d+', '', activity)
    activity = re.sub(r'\W+', ' ', activity)
    activity = activity.strip()
    return activity

df['ProcessedActivity'] = df['BaseActivity'].apply(preprocess_activity)

# Vectorize Activities
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1,3))
vectorized_activities = vectorizer.fit_transform(df['ProcessedActivity'].unique())

# Cluster Similar Activities
clustering = AgglomerativeClustering(
    n_clusters=None,
    metric='cosine',
    linkage='average',
    distance_threshold=1 - similarity_threshold
)
cluster_labels = clustering.fit_predict(normalize(vectorized_activities.toarray()))

# Majority Voting Within Clusters
activity_fixed = []
for cluster in np.unique(cluster_labels):
    cluster_df = df[df['ProcessedActivity'].map(lambda x: cluster_labels[np.where(vectorized_activities.toarray() == vectorizer.transform([x]).toarray())[0][0]]) == cluster]
    canonical_activity = cluster_df['BaseActivity'].value_counts().index[0]
    activity_fixed.extend([canonical_activity] * len(cluster_df))

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
    print(f"Precision threshold (≥ 0.6) met: {'Yes' if precision >= 0.6 else 'No'}")
else:
    print("No labels available for metric calculation.")

# Integrity Check
clean_activities = df[df['ishomonymous'] == 0]
unchanged_homonymous = clean_activities[clean_activities['BaseActivity'] == clean_activities['Activity_fixed']].shape[0]
corrected_homonymous = clean_activities[clean_activities['BaseActivity'] != clean_activities['Activity_fixed']].shape[0]
print(f"Clean activities modified: {unchanged_homonymous}")
print(f"Homonymous activities corrected: {corrected_homonymous}")

# Save Output
output_columns = ['Case', 'Timestamp', 'Resource', 'Activity', 'Activity_fixed', 'ishomonymous']
if label_column in df.columns:
    output_columns.append(label_column)
df_output = df[output_columns]
df_output['Timestamp'] = pd.to_datetime(df_output['Timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
df_output.to_csv(output_file, index=False)

# Summary Statistics
print(f"Total number of events: {df.shape[0]}")
print(f"Number of homonymous events detected: {df['ishomonymous'].sum()}")
print(f"Unique activities before fixing: {df['BaseActivity'].nunique()}")
print(f"Unique activities after fixing: {df['Activity_fixed'].nunique()}")
print(f"Output file path: {output_file}")

print(f"Run 1: Processed dataset saved to: {output_file}")
print(f"Run 1: Final dataset shape: {df.shape}")
print(f"Run 1: Dataset: pub")
print(f"Run 1: Task type: homonymous")