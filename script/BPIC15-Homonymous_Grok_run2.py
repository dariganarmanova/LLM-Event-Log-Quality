# Generated script for BPIC15-Homonymous - Run 2
# Generated on: 2025-11-18T21:50:14.574946
# Model: grok-4-fast

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import precision_score, recall_score, f1_score
import re

# Configuration
input_file = 'data/bpic15/BPIC15-Homonymous.csv'
output_file = 'data/bpic15/bpic15_homonymous_cleaned_run2.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
homonymous_suffix = ':homonymous'
activity_suffix_pattern = r'(_signed\d*|_\d+)$'
similarity_threshold = 0.8
linkage_method = 'average'
case_sensitive = False

# Run-specific prints
print(f"Run 2: Original dataset shape: {pd.read_csv(input_file).shape}")

# Load CSV
df = pd.read_csv(input_file)

# Normalize column names if needed
if 'Case ID' in df.columns:
    df.rename(columns={'Case ID': 'Case'}, inplace=True)

# Ensure required columns exist
required_cols = [case_column, activity_column, timestamp_column]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' not found in the dataset.")

# #2. Identify Homonymous Activities
df['ishomonymous'] = df[activity_column].str.endswith(homonymous_suffix, na=False).astype(int)
df['BaseActivity'] = df[activity_column].str.replace(homonymous_suffix, '', regex=False)

# #7. Calculate Detection Metrics (BEFORE FIXING)
if label_column in df.columns:
    y_true = (df[label_column].notna() & (df[label_column].astype(str).str.strip() != '')).astype(int)
    y_pred = df['ishomonymous']
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print("=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    met = "met" if precision >= 0.6 else "not met"
    print(f"Precision threshold (≥ 0.6) {met}")
else:
    print("=== Detection Performance Metrics ===")
    print("Precision: 0.0000")
    print("Recall: 0.0000")
    print("F1-Score: 0.0000")
    print("No labels available for metric calculation.")
    print("Precision threshold (≥ 0.6) not met")

# #3. Preprocess Activity Names (only for homonymous)
homo_mask = df['ishomonymous'] == 1
if homo_mask.sum() > 0:
    homo_df = df[homo_mask].copy()
    homo_df['StemActivity'] = homo_df['BaseActivity'].str.replace(activity_suffix_pattern, '', regex=True)
    homo_df['ProcessedActivity'] = homo_df['StemActivity'].str.replace(r'[_-]', ' ', regex=True)
    homo_df['ProcessedActivity'] = homo_df['ProcessedActivity'].str.replace(r'\s+', ' ', regex=True).str.strip()
    if not case_sensitive:
        homo_df['ProcessedActivity'] = homo_df['ProcessedActivity'].str.lower()
    unique_processed = homo_df['ProcessedActivity'].unique()
    if len(unique_processed) > 0:
        # #4. Vectorize Activities
        vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
        vectors = vectorizer.fit_transform(unique_processed)
        vectors_dense = vectors.toarray()
        # #5. Cluster Similar Activities
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric='cosine',
            linkage=linkage_method,
            distance_threshold=1 - similarity_threshold
        )
        cluster_labels = clustering.fit_predict(vectors_dense)
        processed_to_cluster = dict(zip(unique_processed, cluster_labels))
        homo_df['cluster'] = homo_df['ProcessedActivity'].map(processed_to_cluster)
        # #6. Majority Voting Within Clusters
        canonical_dict = {}
        for clus in homo_df['cluster'].unique():
            if pd.isna(clus):
                continue
            cluster_stems = homo_df.loc[homo_df['cluster'] == clus, 'StemActivity']
            if len(cluster_stems) > 0:
                mode_stem = cluster_stems.mode()
                canonical = mode_stem.iloc[0] if len(mode_stem) > 0 else cluster_stems.iloc[0]
                canonical_dict[clus] = canonical
        homo_df['Activity_fixed'] = homo_df['cluster'].map(canonical_dict)
        # Assign to df
        df.loc[~homo_mask, 'Activity_fixed'] = df.loc[~homo_mask, activity_column]
        df.loc[homo_mask, 'Activity_fixed'] = homo_df['Activity_fixed'].values
    else:
        df['Activity_fixed'] = df[activity_column]
else:
    df['Activity_fixed'] = df[activity_column]

# Drop intermediate column
df.drop('BaseActivity', axis=1, inplace=True, errors='ignore')

# #8. Integrity Check
clean_modified = ((df['ishomonymous'] == 0) & (df[activity_column] != df['Activity_fixed'])).sum()
homo_unchanged = ((df['ishomonymous'] == 1) & (df[activity_column].str.replace(homonymous_suffix, '', regex=False) == df['Activity_fixed'])).sum()
homo_corrected = df['ishomonymous'].sum() - homo_unchanged
print(f"Clean activities modified: {clean_modified} (should be 0)")
print(f"Homonymous activities unchanged: {homo_unchanged}")
print(f"Homonymous activities corrected: {homo_corrected}")

# #9. Save Output
# Format timestamp
df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')

# Keep all original columns + added
df.to_csv(output_file, index=False)

# #10. Summary Statistics
print(f"Total number of events: {len(df)}")
print(f"Number of homonymous events detected: {df['ishomonymous'].sum()}")
print(f"Unique activities before fixing: {df[activity_column].nunique()}")
print(f"Unique activities after fixing: {df['Activity_fixed'].nunique()}")
print(f"Output file path: {output_file}")

# Required prints
print(f"Run 2: Processed dataset saved to: data/bpic15/bpic15_homonymous_cleaned_run2.csv")
print(f"Run 2: Final dataset shape: {df.shape}")
print(f"Run 2: Dataset: bpic15")
print(f"Run 2: Task type: homonymous")