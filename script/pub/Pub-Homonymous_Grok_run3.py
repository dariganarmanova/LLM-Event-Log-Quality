# Generated script for Pub-Homonymous - Run 3
# Generated on: 2025-11-18T18:43:56.792602
# Model: grok-4-fast

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import precision_score, recall_score, f1_score
import re

# Configuration
input_file = 'data/pub/Pub-Homonymous.csv'
output_file = 'data/pub/pub_homonymous_cleaned_run3.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
homonymous_suffix = ':homonymous'
similarity_threshold = 0.8
linkage_method = 'average'
activity_suffix_pattern = r'(_signed\d*|_\d+)$'
min_matching_events = 2

# Load data
df = pd.read_csv(input_file)
print(f"Run 3: Original dataset shape: {df.shape}")

# Ensure required columns
required = [case_column, activity_column, timestamp_column]
for col in required:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# Normalize CaseID to Case if present
if 'CaseID' in df.columns:
    df = df.rename(columns={'CaseID': case_column})

# Step 2: Identify Homonymous Activities
df['ishomonymous'] = df[activity_column].str.endswith(homonymous_suffix).astype(int)
df['BaseActivity'] = df[activity_column].str.rstrip(homonymous_suffix)

# Step 7: Calculate Detection Metrics (before fixing)
print("=== Detection Performance Metrics ===")
if label_column in df.columns:
    y_true = (~df[label_column].isna() & (df[label_column] != '')).astype(int)
    y_pred = df['ishomonymous']
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    if precision >= 0.6:
        print("Precision threshold (>= 0.6) met")
    else:
        print("Precision threshold (>= 0.6) not met")
else:
    print("Precision: 0.0000")
    print("Recall: 0.0000")
    print("F1-Score: 0.0000")
    print("No labels available for metric calculation.")
    print("Precision threshold (>= 0.6) not met")

# Steps 3-6: Preprocess, Vectorize, Cluster, Majority Voting
mask_hom = df['ishomonymous'] == 1
df['Activity_fixed'] = df[activity_column]
df.loc[~mask_hom, 'Activity_fixed'] = df.loc[~mask_hom, activity_column]

if mask_hom.sum() > 0:
    # Preprocess for homonymous only
    df.loc[mask_hom, 'Stem'] = df.loc[mask_hom, 'BaseActivity'].str.replace(activity_suffix_pattern, '', regex=True)
    df.loc[mask_hom, 'ProcessedActivity'] = (
        df.loc[mask_hom, 'Stem']
        .str.lower()
        .str.replace(r'[_-]', ' ', regex=True)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )

    unique_processed = df.loc[mask_hom, 'ProcessedActivity'].dropna().unique()
    if len(unique_processed) > 0:
        vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
        vectors = vectorizer.fit_transform(unique_processed).toarray()

        if len(unique_processed) > 1:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                metric='cosine',
                linkage=linkage_method,
                distance_threshold=1 - similarity_threshold
            )
            cluster_labels = clustering.fit_predict(vectors)
        else:
            cluster_labels = np.zeros(len(unique_processed), dtype=int)

        cluster_map = dict(zip(unique_processed, cluster_labels))
        df.loc[mask_hom, 'cluster'] = df.loc[mask_hom, 'ProcessedActivity'].map(cluster_map)

        # Majority voting per cluster
        hom_df = df[mask_hom].copy()
        def normalize_stemmed(base):
            stemmed = re.sub(activity_suffix_pattern, '', base)
            norm = stemmed.lower().replace('_', ' ').replace('-', ' ')
            return re.sub(r'\s+', ' ', norm).strip()

        for cluster_id in hom_df['cluster'].dropna().unique():
            group = hom_df[hom_df['cluster'] == cluster_id]
            if len(group) < min_matching_events:
                continue  # Leave unchanged for small clusters
            group['stemmed_normalized'] = group['BaseActivity'].apply(normalize_stemmed)
            if not group['stemmed_normalized'].empty:
                canonical = group['stemmed_normalized'].mode().iloc[0]
                mask_group = df.index.isin(group.index)
                df.loc[mask_group, 'Activity_fixed'] = canonical

# Drop temporary columns
temp_cols = ['BaseActivity', 'Stem', 'ProcessedActivity', 'cluster', 'stemmed_normalized']
for col in temp_cols:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

# Step 8: Integrity Check
clean_modified = ((df['ishomonymous'] == 0) & (df[activity_column] != df['Activity_fixed'])).sum()
hom_unchanged = ((df['ishomonymous'] == 1) & (df[activity_column].str.rstrip(homonymous_suffix) == df['Activity_fixed'])).sum()
hom_corrected = ((df['ishomonymous'] == 1) & (df[activity_column].str.rstrip(homonymous_suffix) != df['Activity_fixed'])).sum()
print("=== Integrity Check ===")
print(f"Clean activities modified: {clean_modified} (should be 0)")
print(f"Homonymous activities unchanged: {hom_unchanged}")
print(f"Homonymous activities corrected: {hom_corrected}")

# Step 9: Prepare output
# Format timestamp
df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')

# Select columns, preserving optional
output_cols = [case_column]
if 'Variant' in df.columns:
    output_cols.append('Variant')
output_cols += [timestamp_column]
if 'Resource' in df.columns:
    output_cols.append('Resource')
output_cols += [activity_column, 'Activity_fixed', 'ishomonymous']
if label_column in df.columns:
    output_cols.append(label_column)
df = df[output_cols]

# Step 10: Summary Statistics
print("=== Summary Statistics ===")
print(f"Total number of events: {len(df)}")
print(f"Number of homonymous events detected: {df['ishomonymous'].sum()}")
print(f"Unique activities before fixing: {df[activity_column].nunique()}")
print(f"Unique activities after fixing: {df['Activity_fixed'].nunique()}")
print(f"Output file path: {output_file}")

# Save
df.to_csv(output_file, index=False)

# Required prints
print(f"Run 3: Processed dataset saved to: data/pub/pub_homonymous_cleaned_run3.csv")
print(f"Run 3: Final dataset shape: {df.shape}")
print(f"Run 3: Dataset: pub")
print(f"Run 3: Task type: homonymous")