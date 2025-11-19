# Generated script for Pub-Distorted - Run 3
# Generated on: 2025-11-18T17:31:50.244184
# Model: meta-llama/Llama-3.1-8B-Instruct

import pandas as pd
import re
from sklearn.metrics import precision_score, recall_score, f1_score

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
input_file = 'data/pub/Pub-Distorted.csv'
df = pd.read_csv(input_file)

# Normalize CaseID → Case if needed (handle naming variations)
df['Case'] = df['Case'].apply(lambda x: x.strip().lower() if case_sensitive else x.strip().lower())

# Store original Activity values in `original_activity` column for reference
df['original_activity'] = df['Activity']

# Create `isdistorted` column: 1 if Activity ends with `distorted_suffix`, else 0
df['isdistorted'] = df['Activity'].apply(lambda x: 1 if re.search(activity_suffix_pattern, x) else 0)

# Create `BaseActivity` column: Activity with distorted suffix removed
df['BaseActivity'] = df['Activity'].apply(lambda x: re.sub(activity_suffix_pattern, '', x))

# Preprocess Activity Names
def preprocess_activity(activity):
    if case_sensitive:
        return activity
    else:
        activity = activity.lower()
        activity = re.sub(r'[^a-z0-9\s]', '', activity)
        activity = re.sub(r'\s+', ' ', activity)
        activity = activity.strip()
        return activity

df['ProcessedActivity'] = df['Activity'].apply(preprocess_activity)

# Filter out activities shorter than min_length characters (too short for meaningful comparison)
min_length = 4
df = df[df['ProcessedActivity'].apply(lambda x: len(x) >= min_length)]

# Calculate Jaccard N-gram Similarity
def jaccard_similarity(val1, val2):
    ngram_size = 3
    val1_ngrams = set([val1[i:i+ngram_size] for i in range(len(val1) - ngram_size + 1)])
    val2_ngrams = set([val2[i:i+ngram_size] for i in range(len(val2) - ngram_size + 1)])
    intersection = val1_ngrams.intersection(val2_ngrams)
    union = val1_ngrams.union(val2_ngrams)
    if len(union) == 0:
        return 0.0
    return len(intersection) / len(union)

# Find Similar Pairs
similar_pairs = []
unique_activities = df['ProcessedActivity'].unique()
for i in range(len(unique_activities)):
    for j in range(i+1, len(unique_activities)):
        val1 = unique_activities[i]
        val2 = unique_activities[j]
        similarity = jaccard_similarity(val1, val2)
        if similarity >= similarity_threshold:
            similar_pairs.append((val1, val2))

# Cluster Similar Activities (Union-Find)
def find(parent, i):
    if parent[i] != i:
        parent[i] = find(parent, parent[i])
    return parent[i]

def union(parent, rank, x, y):
    x_root = find(parent, x)
    y_root = find(parent, y)
    if x_root == y_root:
        return
    if rank[x_root] < rank[y_root]:
        parent[x_root] = y_root
    elif rank[x_root] > rank[y_root]:
        parent[y_root] = x_root
    else:
        parent[y_root] = x_root
        rank[x_root] += 1

parent = {i: i for i in range(len(unique_activities))}
rank = {i: 0 for i in range(len(unique_activities))}
for pair in similar_pairs:
    union(parent, rank, unique_activities.index(pair[0]), unique_activities.index(pair[1]))

# Build clusters
clusters = {}
for i in range(len(unique_activities)):
    root = find(parent, i)
    if root not in clusters:
        clusters[root] = []
    clusters[root].append(unique_activities[i])

# Majority Voting Within Clusters
canonical_activities = {}
for cluster in clusters.values():
    if len(cluster) >= min_matching_events:
        cluster_activities = df[df['ProcessedActivity'].isin(cluster)]['original_activity'].value_counts()
        canonical_activity = cluster_activities.index[0]
        canonical_activities.update({act: canonical_activity for act in cluster})

# Mark distorted activities
df['canonical_activity'] = df['original_activity'].map(canonical_activities)
df['is_distorted'] = df['original_activity'].apply(lambda x: 1 if x != df['canonical_activity'] else 0)

# Calculate Detection Metrics (BEFORE FIXING)
if 'label' in df.columns:
    def normalize_label(label):
        if pd.isnull(label) or label == '':
            return 0
        elif 'distorted' in label.lower():
            return 1
        else:
            return 0

    y_true = df['label'].apply(normalize_label)
    y_pred = df['is_distorted']
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"✓/✗ Precision threshold (≥ 0.6) met/not met")

# Integrity Check
total_clusters = len(clusters)
total_distorted = df['is_distorted'].sum()
clean_activities = df[df['is_distorted'] == 0].shape[0]
print(f"Total distortion clusters detected: {total_clusters}")
print(f"Total activities marked as distorted: {total_distorted}")
print(f"Activities to be fixed: {clean_activities}")

# Fix Activities
df['Activity_fixed'] = df['canonical_activity']

# Save Output
output_file = 'data/pub/pub_distorted_cleaned_run3.csv'
df.to_csv(output_file, index=False)

# Summary Statistics
print(f"Run 3: Processed dataset saved to: {output_file}")
print(f"Run 3: Final dataset shape: {df.shape}")
print(f"Run 3: Dataset: pub")
print(f"Run 3: Task type: distorted")
print(f"Run 3: Total number of events: {df.shape[0]}")
print(f"Run 3: Number of distorted events detected: {total_distorted}")
print(f"Run 3: Unique activities before fixing: {len(unique_activities)}")
print(f"Run 3: Unique activities after fixing: {len(canonical_activities)}")
print(f"Run 3: Activity reduction count and percentage: {(len(unique_activities) - len(canonical_activities))}/{len(unique_activities)}")
print(f"Run 3: Output file path: {output_file}")
print(f"Run 3: Sample of up to 10 transformations:")
print(df[['original_activity', 'canonical_activity']].head(10))