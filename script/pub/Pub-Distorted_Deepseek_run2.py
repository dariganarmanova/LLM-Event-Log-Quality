# Generated script for Pub-Distorted - Run 2
# Generated on: 2025-11-13T17:45:23.002325
# Model: deepseek-ai/DeepSeek-V3-0324

import pandas as pd
import re
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score

def generate_ngrams(text, n):
    return set([text[i:i+n] for i in range(len(text)-n+1)])

def jaccard_similarity(set1, set2):
    if not set1 and not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union

class UnionFind:
    def __init__(self):
        self.parent = {}
    
    def find(self, item):
        if item not in self.parent:
            self.parent[item] = item
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item
    
    def union(self, item1, item2):
        root1 = self.find(item1)
        root2 = self.find(item2)
        if root1 != root2:
            self.parent[root2] = root1

def normalize_activity_name(activity):
    activity = str(activity).lower()
    activity = re.sub(r'[^a-z0-9\s]', '', activity)
    activity = re.sub(r'\s+', ' ', activity).strip()
    return activity

def process_distorted_activities(df, ngram_size, similarity_threshold, min_length):
    df['original_activity'] = df['Activity']
    df['ProcessedActivity'] = df['Activity'].apply(normalize_activity_name)
    
    unique_activities = df[df['ProcessedActivity'].str.len() >= min_length]['ProcessedActivity'].unique()
    uf = UnionFind()
    
    for i in range(len(unique_activities)):
        for j in range(i+1, len(unique_activities)):
            act1 = unique_activities[i]
            act2 = unique_activities[j]
            ngrams1 = generate_ngrams(act1, ngram_size)
            ngrams2 = generate_ngrams(act2, ngram_size)
            similarity = jaccard_similarity(ngrams1, ngrams2)
            if similarity >= similarity_threshold:
                uf.union(act1, act2)
    
    clusters = defaultdict(list)
    for act in unique_activities:
        root = uf.find(act)
        clusters[root].append(act)
    
    clusters = {k: v for k, v in clusters.items() if len(v) >= 2}
    
    activity_mapping = {}
    for cluster in clusters.values():
        original_activities = df[df['ProcessedActivity'].isin(cluster)]['original_activity']
        most_common = original_activities.value_counts().idxmax()
        for act in cluster:
            activity_mapping[act] = most_common
    
    df['canonical_activity'] = df['ProcessedActivity'].map(activity_mapping).fillna(df['original_activity'])
    df['is_distorted'] = (df['original_activity'] != df['canonical_activity']).astype(int)
    
    if 'label' in df.columns:
        y_true = df['label'].fillna('').str.contains('distorted', case=False).astype(int)
        y_pred = df['is_distorted']
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        print("=== Detection Performance Metrics ===")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print("✓/✗ Precision threshold (≥ 0.6) met/not met")
    else:
        print("No labels available for metric calculation")
    
    print(f"Total distortion clusters detected: {len(clusters)}")
    print(f"Total activities marked as distorted: {df['is_distorted'].sum()}")
    print(f"Activities to be fixed: {df['is_distorted'].sum()}")
    
    df['Activity_fixed'] = df['canonical_activity']
    
    unique_before = df['original_activity'].nunique()
    unique_after = df['Activity_fixed'].nunique()
    print(f"Unique activities before fixing: {unique_before}")
    print(f"Unique activities after fixing: {unique_after}")
    print(f"Activity reduction count: {unique_before - unique_after}")
    print(f"Activity reduction percentage: {(unique_before - unique_after)/unique_before*100:.2f}%")
    
    sample_transformations = df[df['is_distorted'] == 1][['original_activity', 'Activity_fixed']].drop_duplicates().head(10)
    for _, row in sample_transformations.iterrows():
        print(f"{row['original_activity']} → {row['Activity_fixed']}")
    
    return df

df = pd.read_csv('data/pub/Pub-Distorted.csv')
print(f"Run 2: Original dataset shape: {df.shape}")

df = process_distorted_activities(df, ngram_size=3, similarity_threshold=0.56, min_length=4)

output_columns = ['Case', 'Timestamp', 'Activity', 'Activity_fixed']
if 'Variant' in df.columns:
    output_columns.append('Variant')
if 'Resource' in df.columns:
    output_columns.append('Resource')
if 'label' in df.columns:
    output_columns.append('label')

df[output_columns].to_csv('data/pub/pub_distorted_cleaned_run2.csv', index=False)

print(f"Run 2: Processed dataset saved to: data/pub/pub_distorted_cleaned_run2.csv")
print(f"Run 2: Final dataset shape: {df.shape}")
print(f"Run 2: Dataset: pub")
print(f"Run 2: Task type: distorted")