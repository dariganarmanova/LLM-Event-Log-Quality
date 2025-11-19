# Generated script for Pub-Homonymous - Run 1
# Generated on: 2025-11-18T18:41:39.894438
# Model: grok-4-fast

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import precision_recall_fscore_support

# Configuration
input_file = 'data/pub/Pub-Homonymous.csv'
output_file = 'data/pub/pub_homonymous_cleaned_run1.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'
homonymous_suffix = ':homonymous'
similarity_threshold = 0.8
linkage_method = 'average'
activity_suffix_pattern = r'(_signed\d*|_\d+)$'

# Load data
df = pd.read_csv(input_file)
print(f"Run 1: Original dataset shape: {df.shape}")

# Verify required columns
required = [case_column, activity_column, timestamp_column]
for col in required:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# Normalize CaseID to Case if present
if 'Case ID' in df.columns:
    df[case_column] = df['Case ID']
    df = df.drop('Case ID', axis=1)

# Step #2: Identify Homonymous Activities
df['ishomonymous'] = df[activity_column].str.endswith(homonymous_suffix, na=False).astype(int)
df['BaseActivity'] = df[activity_column].str.rstrip(homonymous_suffix)

# Step #3: Preprocess Activity Names
def preprocess_activity(text):
    if pd.isna(text):
        return text
    text = str(text).lower()
    text = re.sub(r'[_-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Create StemActivity: for homonymous, remove additional suffix pattern
df['StemActivity'] = df['BaseActivity']
mask = df['ishomonymous'] == 1
df.loc[mask, 'StemActivity'] = df.loc[mask, 'StemActivity'].str.replace(activity_suffix_pattern, '', regex=True)
df['ProcessedActivity'] = df['StemActivity'].apply(preprocess_activity)

# Step #7: Calculate Detection Metrics (BEFORE FIXING)
print("=== Detection Performance Metrics ===")
if label_column in df.columns:
    y_true = (df[label_column].notna() & (df[label_column].astype(str) != '')).astype(int)
    y_pred = df['ishomonymous'].astype(int)
    if len(y_true.unique()) > 1 or len(y_pred.unique()) > 1:
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    else:
        p = r = f1 = 0.0
    print(f"Precision: {p:.4f}")
    print(f"Recall: {r:.4f}")
    print(f"F1-Score: {f1:.4f}")
    met = "met" if p >= 0.6 else "not met"
    print(f"Precision threshold (≥ 0.6) {met}")
else:
    print("Precision: 0.0000")
    print("Recall: 0.0000")
    print("F1-Score: 0.0000")
    print("No labels available for metric calculation.")

# Steps #4-6: Vectorize, Cluster, Majority Voting
df['Activity_fixed'] = df[activity_column]
df['cluster_id'] = pd.NA

hom_mask = df['ishomonymous'] == 1
unique_hom_proc = df.loc[hom_mask, 'ProcessedActivity'].dropna().unique()

if len(unique_hom_proc) > 0:
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
    X = vectorizer.fit_transform(unique_hom_proc)
    X_dense = X.toarray()

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric='cosine',
        linkage=linkage_method,
        distance_threshold=1 - similarity_threshold
    )
    cluster_labels = clustering.fit_predict(X_dense)

    cluster_map = dict(zip(unique_hom_proc, cluster_labels))
    df['cluster_id'] = df['ProcessedActivity'].map(cluster_map)

    # Majority voting on StemActivity per cluster
    cluster_canonical = {}
    for clus in df.loc[hom_mask, 'cluster_id'].dropna().unique():
        cluster_stems = df.loc[(hom_mask) & (df['cluster_id'] == clus), 'StemActivity']
        if not cluster_stems.empty and not cluster_stems.isna().all():
            canonical = cluster_stems.mode().iloc[0]
            cluster_canonical[clus] = canonical

    # Assign canonical to homonymous in clusters
    assign_mask = hom_mask & df['cluster_id'].notna()
    df.loc[assign_mask, 'Activity_fixed'] = df.loc[assign_mask, 'cluster_id'].map(cluster_canonical)

# Step #8: Integrity Check
print("Integrity Check:")
clean_modified = ((df['ishomonymous'] == 0) & (df[activity_column] != df['Activity_fixed'])).sum()
print(f"Clean activities modified: {clean_modified} (should be 0)")
hom_unchanged = ((df['ishomonymous'] == 1) & (df['BaseActivity'] == df['Activity_fixed'])).sum()
print(f"Homonymous activities unchanged: {hom_unchanged}")
hom_corrected = ((df['ishomonymous'] == 1) & (df['BaseActivity'] != df['Activity_fixed'])).sum()
print(f"Homonymous activities corrected: {hom_corrected}")

# Step #9: Prepare Output
output_cols = [case_column, timestamp_column]
if 'Resource' in df.columns:
    output_cols.append('Resource')
output_cols += [activity_column, 'Activity_fixed', 'ishomonymous']
if label_column in df.columns:
    output_cols.append(label_column)
if 'Variant' in df.columns:
    output_cols.append('Variant')

df_out = df[output_cols].copy()

# Standardize timestamp
df_out[timestamp_column] = pd.to_datetime(df_out[timestamp_column], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')

df_out.to_csv(output_file, index=False)

# Step #10: Summary Statistics
print(f"Total number of events: {len(df)}")
print(f"Number of homonymous events detected: {df['ishomonymous'].sum()}")
print(f"Unique activities before fixing: {df[activity_column].nunique()}")
print(f"Unique activities after fixing: {df['Activity_fixed'].nunique()}")
print(f"Output file path: {output_file}")

# Required prints
print(f"Run 1: Processed dataset saved to: data/pub/pub_homonymous_cleaned_run1.csv")
print(f"Run 1: Final dataset shape: {df.shape}")
print(f"Run 1: Dataset: pub")
print(f"Run 1: Task type: homonymous")