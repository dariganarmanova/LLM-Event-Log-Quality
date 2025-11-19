# Generated script for BPIC11-Homonymous - Run 3
# Generated on: 2025-11-18T22:38:41.761352
# Model: grok-4-fast

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
activity_suffix_pattern = r'(_signed\d*|_\d+)$'
similarity_threshold = 0.8
case_sensitive = False
use_fuzzy_matching = False
linkage_method = 'average'
homonymous_suffix = ':homonymous'
input_file = 'data/bpic11/BPIC11-Homonymous.csv'
output_file = 'data/bpic11/bpic11_homonymous_cleaned_run3.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
resource_column = 'Resource'

# Load the data
df = pd.read_csv(input_file)
print(f"Run 3: Original dataset shape: {df.shape}")

# Ensure required columns exist (assume they do as per config)
# Normalize Case if needed
if 'Case ID' in df.columns:
    df['Case'] = df['Case ID']
elif 'case' in df.columns:
    df['Case'] = df['case']

# #2. Identify Homonymous Activities
df['ishomonymous'] = df[activity_column].str.endswith(homonymous_suffix, na=False).astype(int)
df['BaseActivity'] = df[activity_column].str.rsplit(homonymous_suffix, n=1).str[0]

# #7. Calculate Detection Metrics (BEFORE FIXING)
if label_column in df.columns:
    y_true = ((df[label_column].notna()) & (df[label_column].astype(str).str.strip() != '')).astype(int)
    y_pred = df['ishomonymous']
    if len(np.unique(y_pred)) > 1 and len(np.unique(y_true)) > 1:
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
    else:
        prec = rec = f1 = 0.0
else:
    prec = rec = f1 = 0.0
    print("No labels available for metric calculation.")

print("=== Detection Performance Metrics ===")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Precision threshold (≥ 0.6) {'met' if prec >= 0.6 else 'not met'}")

# #3. Preprocess Activity Names
def normalize_suffixes(activity):
    if pd.isna(activity):
        return activity
    return re.sub(activity_suffix_pattern, '', str(activity))

df['NormalizedBase'] = df['BaseActivity'].apply(normalize_suffixes)

def preprocess_text(text, case_sensitive):
    if pd.isna(text):
        return text
    text = str(text)
    if not case_sensitive:
        text = text.lower()
    text = re.sub(r'[_-]', ' ', text)
    text = ' '.join(text.split())
    return text

df['ProcessedActivity'] = df['NormalizedBase'].apply(lambda x: preprocess_text(x, case_sensitive))

# #4. Vectorize Activities (only for homonymous)
homo_mask = df['ishomonymous'] == 1
homo_df = df[homo_mask].copy()
if len(homo_df) > 0:
    unique_processed = homo_df['ProcessedActivity'].dropna().unique()
    if len(unique_processed) > 1:
        vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3), lowercase=False)
        vectors = vectorizer.fit_transform(unique_processed)
        dense_vectors = vectors.toarray()
        
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric='cosine',
            linkage=linkage_method,
            distance_threshold=1 - similarity_threshold
        )
        cluster_labels = clustering.fit_predict(dense_vectors)
        
        processed_to_cluster = dict(zip(unique_processed, cluster_labels))
        
        # #5 & #6. Cluster and Majority Voting
        homo_df['cluster'] = homo_df['ProcessedActivity'].map(processed_to_cluster)
        cluster_canonical = {}
        for cluster in homo_df['cluster'].unique():
            cluster_mask = homo_df['cluster'] == cluster
            cluster_norm_bases = homo_df.loc[cluster_mask, 'NormalizedBase'].value_counts()
            if len(cluster_norm_bases) > 0:
                most_frequent_norm = cluster_norm_bases.index[0]
                canonical = preprocess_text(most_frequent_norm, case_sensitive)
                cluster_canonical[cluster] = canonical
        
        # Assign to df
        df['Activity_fixed'] = df[activity_column]
        cluster_map = homo_df.set_index('ProcessedActivity')['cluster'].to_dict()
        for idx, row in homo_df.iterrows():
            if row['ProcessedActivity'] in cluster_map:
                cl = cluster_map[row['ProcessedActivity']]
                if cl in cluster_canonical:
                    df.at[idx, 'Activity_fixed'] = cluster_canonical[cl]
    else:
        # Single or no homo, normalize suffixes for them
        df['Activity_fixed'] = df[activity_column]
        df.loc[homo_mask, 'Activity_fixed'] = df.loc[homo_mask, 'NormalizedBase'].apply(lambda x: preprocess_text(x, case_sensitive))
else:
    df['Activity_fixed'] = df[activity_column]

# #8. Integrity Check
clean_modified = ((df['ishomonymous'] == 0) & (df[activity_column] != df['Activity_fixed'])).sum()
homo_unchanged = ((df['ishomonymous'] == 1) & (df['BaseActivity'] == df['Activity_fixed'])).sum()
homo_corrected = ((df['ishomonymous'] == 1) & (df['BaseActivity'] != df['Activity_fixed'])).sum()
print("=== Integrity Check ===")
print(f"Clean activities modified: {clean_modified} (should be 0)")
print(f"Homonymous activities unchanged: {homo_unchanged}")
print(f"Homonymous activities corrected: {homo_corrected}")

# Format timestamp
if timestamp_column in df.columns:
    df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')

# #9 & #10. Save Output and Summary
df.to_csv(output_file, index=False)

print("=== Summary Statistics ===")
print(f"Total number of events: {len(df)}")
print(f"Number of homonymous events detected: {df['ishomonymous'].sum()}")
print(f"Unique activities before fixing: {df[activity_column].nunique()}")
print(f"Unique activities after fixing: {df['Activity_fixed'].nunique()}")
print(f"Output file path: {output_file}")

print(f"Run 3: Processed dataset saved to: data/bpic11/bpic11_homonymous_cleaned_run3.csv")
print(f"Run 3: Final dataset shape: {df.shape}")
print(f"Run 3: Dataset: bpic11")
print(f"Run 3: Task type: homonymous")