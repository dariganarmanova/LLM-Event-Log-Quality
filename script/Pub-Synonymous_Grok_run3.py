# Generated script for Pub-Synonymous - Run 3
# Generated on: 2025-11-18T18:53:59.414718
# Model: grok-4-fast

import pandas as pd
import re
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_fscore_support

# Configuration
input_file = 'data/pub/Pub-Synonymous.csv'
dataset_name = 'pub'
similarity_threshold = 0.8
min_synonym_group_size = 2
ngram_range = (1, 3)
save_detection_file = False
label_column = 'label'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
output_file = 'data/pub/pub_synonymous_cleaned_run3.csv'

print("Run Number: 3")
print(f"Dataset Name: {dataset_name}")
print("Task Type: synonymous")

# Step 1: Load and Validate
df = pd.read_csv(input_file)
if 'CaseID' in df.columns and case_column not in df.columns:
    df[case_column] = df['CaseID']
if activity_column not in df.columns:
    raise ValueError(f"{activity_column} column is missing.")
df['original_activity'] = df[activity_column].astype(str).fillna('')
df[activity_column] = df['original_activity']
if timestamp_column in df.columns:
    df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce')
if case_column in df.columns and timestamp_column in df.columns:
    df = df.sort_values([case_column, timestamp_column]).reset_index(drop=True)
print(f"Run 3: Original dataset shape: {df.shape}")
print(df.head())
print(f"Number of unique Activity values: {df['original_activity'].nunique()}")

# Step 2: Normalize Activity Labels
def normalize_activity(activity):
    activity = str(activity).lower()
    activity = re.sub(r'[^\w\s]', '', activity)
    activity = ' '.join(activity.split()).strip()
    if not activity:
        return 'empty_activity'
    return activity

df['Activity_clean'] = df['original_activity'].apply(normalize_activity)

# Step 3: TF-IDF Embedding and Similarity
unique_activities = sorted(df['Activity_clean'].unique())
if len(unique_activities) < 2:
    df['SynonymGroup'] = -1
    df['canonical_activity'] = df['original_activity']
    df['is_synonymous_event'] = 0
    num_clusters = 0
else:
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=ngram_range, lowercase=True, min_df=1)
    tfidf_matrix = vectorizer.fit_transform(unique_activities)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Unique activity count: {len(unique_activities)}")

    # Step 4: Cluster Using Union-Find
    n = len(unique_activities)
    parent = list(range(n))
    rank = [0] * n

    def find(parent, i):
        if parent[i] != i:
            parent[i] = find(parent, parent[i])
        return parent[i]

    def union(parent, rank, x, y):
        px = find(parent, x)
        py = find(parent, y)
        if px != py:
            if rank[px] > rank[py]:
                parent[py] = px
            elif rank[px] < rank[py]:
                parent[px] = py
            else:
                parent[py] = px
                rank[px] += 1

    for i in range(n):
        for j in range(i + 1, n):
            if similarity_matrix[i, j] >= similarity_threshold:
                union(parent, rank, i, j)

    clusters = defaultdict(list)
    for i in range(n):
        root = find(parent, i)
        clusters[root].append(unique_activities[i])

    valid_clusters = {k: v for k, v in clusters.items() if len(v) >= min_synonym_group_size}
    num_clusters = len(valid_clusters)
    print(f"Number of synonym clusters discovered: {num_clusters}")

    # Step 5: Select Canonical Form and Assign
    cluster_id_map = {}
    canonicals = {}
    cid = 0
    for root, members in valid_clusters.items():
        mask = df['Activity_clean'].isin(members)
        if mask.sum() == 0:
            continue
        cluster_originals_freq = df.loc[mask, 'original_activity'].value_counts()
        can_original = cluster_originals_freq.index[0]
        canonicals[cid] = can_original
        for mem in members:
            cluster_id_map[mem] = cid
        cid += 1

    df['SynonymGroup'] = df['Activity_clean'].map(cluster_id_map).fillna(-1).astype(int)
    df['canonical_activity'] = df['original_activity']
    for cidd, can_orig in canonicals.items():
        mask_c = df['SynonymGroup'] == cidd
        df.loc[mask_c, 'canonical_activity'] = can_orig
    df['is_synonymous_event'] = ((df['SynonymGroup'] != -1) & (df['original_activity'] != df['canonical_activity'])).astype(int)

# Step 6: Calculate Detection Metrics
print("=== Detection Performance Metrics ===")
if label_column in df.columns:
    y_true = (df[label_column].notna() & (df[label_column].astype(str) != '')).astype(int)
    y_pred = df['is_synonymous_event']
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
else:
    print("Precision: 0.0000")
    print("Recall: 0.0000")
    print("F1-Score: 0.0000")
    print("No ground-truth labels available for evaluation.")

# Step 7: Integrity Check
print(f"Total synonym clusters found: {num_clusters}")
print(f"Total events flagged as synonyms: {(df['is_synonymous_event'] == 1).sum()}")
print(f"Total canonical/clean events: {len(df) - (df['is_synonymous_event'] == 1).sum()}")

# Step 8: Fix Activities
df[activity_column] = df['canonical_activity']

# Step 10: Create Final Fixed Dataset
helper_columns = ['original_activity', 'Activity_clean', 'canonical_activity', 'SynonymGroup', 'is_synonymous_event']
final_df = df.drop(columns=[col for col in helper_columns if col in df.columns], errors='ignore')

# Step 11: Save Output and Summary
final_df.to_csv(output_file, index=False)

before_unique = df['original_activity'].nunique()
after_unique = final_df[activity_column].nunique()
reduction = before_unique - after_unique
rate = (reduction / before_unique * 100) if before_unique > 0 else 0
syn_replaced = (df['is_synonymous_event'] == 1).sum()
replace_rate = (syn_replaced / len(df) * 100) if len(df) > 0 else 0

print(f"Total rows: {len(final_df)}")
print(f"Synonym clusters found: {num_clusters}")
print(f"Synonymous events replaced: {syn_replaced}")
print(f"Replacement rate: {replace_rate:.2f}%")
print(f"Unique activities before: {before_unique} → after: {after_unique}")
print(f"Activity reduction: {reduction} ({rate:.2f}%)")
print(f"Output file path: {output_file}")

print("Sample transformations (up to 10):")
changes = df[df['is_synonymous_event'] == 1][['original_activity', 'canonical_activity']].drop_duplicates()
for _, row in changes.head(10).iterrows():
    print(f"'{row['original_activity']}' → '{row['canonical_activity']}'")

print(f"Run 3: Processed dataset saved to: {output_file}")
print(f"Run 3: Final dataset shape: {final_df.shape}")
print(f"Run 3: Dataset: {dataset_name}")
print(f"Run 3: Task type: synonymous")