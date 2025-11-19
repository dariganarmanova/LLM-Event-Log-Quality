# Generated script for BPIC15-Homonymous - Run 1
# Generated on: 2025-11-18T21:49:06.833723
# Model: grok-4-fast

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import Counter

# Configuration
input_file = 'data/bpic15/BPIC15-Homonymous.csv'
output_file = 'data/bpic15/bpic15_homonymous_cleaned_run1.csv'
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

# Normalize column names if needed
if 'Case ID' in df.columns:
    df = df.rename(columns={'Case ID': case_column})

# Ensure required columns exist
required_columns = [case_column, activity_column, timestamp_column]
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' not found in the dataset.")

# Step 2: Identify homonymous activities
df['ishomonymous'] = df[activity_column].apply(lambda x: 1 if str(x).endswith(homonymous_suffix) else 0)

# Create BaseActivity
def get_base_activity(activity):
    activity_str = str(activity)
    if activity_str.endswith(homonymous_suffix):
        return activity_str[:-len(homonymous_suffix)]
    return activity_str

df['BaseActivity'] = df[activity_column].apply(get_base_activity)

# Step 3: Preprocess activity names
def preprocess_activity(activity):
    act = str(activity).lower()
    act = re.sub(r'[_-]', ' ', act)
    act = re.sub(r'\s+', ' ', act).strip()
    return act

df['ProcessedActivity'] = df['BaseActivity'].apply(preprocess_activity)

# Step 4 & 5 & 6: Vectorize, cluster, and majority voting for homonymous only
hom_df = df[df['ishomonymous'] == 1]
df['Activity_fixed'] = df[activity_column].where(df['ishomonymous'] == 0, np.nan)

if len(hom_df) > 0:
    unique_processed_hom = hom_df['ProcessedActivity'].dropna().unique()
    if len(unique_processed_hom) > 0:
        vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3), lowercase=False)
        vectors = vectorizer.fit_transform(unique_processed_hom).toarray()

        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric='cosine',
            linkage=linkage_method,
            distance_threshold=1 - similarity_threshold
        )
        cluster_labels = clustering.fit_predict(vectors)

        unique_to_cluster = {unique_processed_hom[i]: cluster_labels[i] for i in range(len(unique_processed_hom))}

        # Get canonical candidate function
        def get_canonical_candidate(base_act):
            stripped = re.sub(activity_suffix_pattern, '', str(base_act))
            return preprocess_activity(stripped)

        # Temporary for hom_df
        hom_df_temp = hom_df.copy()
        hom_df_temp['cluster'] = hom_df_temp['ProcessedActivity'].map(unique_to_cluster)

        canonicals = {}
        processed_to_canonical = {}
        for clus in set(hom_df_temp['cluster'].dropna()):
            if pd.isna(clus):
                continue
            cluster_mask = hom_df_temp['cluster'] == clus
            cluster_bases = hom_df_temp.loc[cluster_mask, 'BaseActivity'].values
            if len(cluster_bases) > 0:
                base_counts = Counter(cluster_bases)
                most_freq_base = base_counts.most_common(1)[0][0]
                canonical = get_canonical_candidate(most_freq_base)
                canonicals[clus] = canonical
                cluster_procs = hom_df_temp.loc[cluster_mask, 'ProcessedActivity'].unique()
                for proc in cluster_procs:
                    processed_to_canonical[proc] = canonical

        # Assign to df
        mask_hom = df['ishomonymous'] == 1
        df.loc[mask_hom, 'Activity_fixed'] = df.loc[mask_hom, 'ProcessedActivity'].map(processed_to_canonical).fillna(df.loc[mask_hom, 'BaseActivity'])

# Step 7: Calculate Detection Metrics (BEFORE FIXING, but detection is ishomonymous)
has_label = label_column in df.columns
if has_label:
    y_true = (df[label_column].notna() & (df[label_column] != '')).astype(int)
    y_pred = df['ishomonymous'].astype(int)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print("=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Precision threshold (≥ 0.6) {'met' if precision >= 0.6 else 'not met'}")
else:
    print("=== Detection Performance Metrics ===")
    print("Precision: 0.0000")
    print("Recall: 0.0000")
    print("F1-Score: 0.0000")
    print("No labels available for metric calculation.")

# Step 8: Integrity Check
clean_modified = ((df['ishomonymous'] == 0) & (df[activity_column] != df['Activity_fixed'])).sum()
hom_unchanged = ((df['ishomonymous'] == 1) & (df['BaseActivity'] == df['Activity_fixed'])).sum()
hom_corrected = ((df['ishomonymous'] == 1) & (df['BaseActivity'] != df['Activity_fixed'])).sum()
print("=== Integrity Check ===")
print(f"Clean activities modified: {clean_modified} (should be 0)")
print(f"Homonymous activities unchanged: {hom_unchanged}")
print(f"Homonymous activities corrected: {hom_corrected}")

# Step 9: Prepare output
df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')

output_cols = [case_column, timestamp_column]
if 'Resource' in df.columns:
    output_cols.append('Resource')
output_cols += [activity_column, 'Activity_fixed', 'ishomonymous']
if label_column in df.columns:
    output_cols.append(label_column)
if 'Variant' in df.columns:
    output_cols = ['Variant'] + output_cols

df_output = df[output_cols].copy()
df_output.to_csv(output_file, index=False)

# Step 10: Summary Statistics
total_events = len(df)
num_hom = df['ishomonymous'].sum()
unique_before = df[activity_column].nunique()
unique_after = df['Activity_fixed'].nunique()
print("=== Summary Statistics ===")
print(f"Total number of events: {total_events}")
print(f"Number of homonymous events detected: {num_hom}")
print(f"Unique activities before fixing: {unique_before}")
print(f"Unique activities after fixing: {unique_after}")
print(f"Output file path: {output_file}")

print(f"Run 1: Processed dataset saved to: {output_file}")
print(f"Run 1: Final dataset shape: {df_output.shape}")
print(f"Run 1: Dataset: bpic15")
print(f"Run 1: Task type: homonymous")