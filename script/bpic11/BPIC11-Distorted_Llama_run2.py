# Generated script for BPIC11-Distorted - Run 2
# Generated on: 2025-11-13T11:27:46.914274
# Model: meta-llama/Llama-3.1-8B-Instruct

import pandas as pd
import re
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk import download
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import os

# Download required NLTK resources
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')

def load_data(input_file):
    try:
        df = pd.read_csv(input_file)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_activity(activity):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    activity = re.sub(r'[^a-zA-Z0-9\s]', '', activity)
    activity = activity.lower()
    tokens = word_tokenize(activity)
    tokens = [token for token in tokens if token not in stop_words]
    pos_tags = pos_tag(tokens)
    lemmatized_tokens = []
    for token, pos in pos_tags:
        if pos.startswith('NN'):
            lemmatized_tokens.append(lemmatizer.lemmatize(token))
        else:
            lemmatized_tokens.append(token)
    activity = ' '.join(lemmatized_tokens)
    return activity

def calculate_jaccard_similarity(val1, val2, ngram_size):
    val1_ngrams = set(ngrams(val1, ngram_size))
    val2_ngrams = set(ngrams(val2, ngram_size))
    intersection = val1_ngrams.intersection(val2_ngrams)
    union = val1_ngrams.union(val2_ngrams)
    similarity = len(intersection) / len(union)
    return similarity

def find_similar_pairs(df, ngram_size, similarity_threshold):
    similar_pairs = []
    unique_activities = df['ProcessedActivity'].unique()
    for i in range(len(unique_activities)):
        for j in range(i+1, len(unique_activities)):
            val1 = unique_activities[i]
            val2 = unique_activities[j]
            similarity = calculate_jaccard_similarity(val1, val2, ngram_size)
            if similarity >= similarity_threshold:
                similar_pairs.append((val1, val2))
    return similar_pairs

def union_find(similar_pairs):
    parent = {}
    rank = {}
    for activity in similar_pairs:
        parent[activity[0]] = activity[0]
        rank[activity[0]] = 0
        parent[activity[1]] = activity[1]
        rank[activity[1]] = 0

    def find(activity):
        if parent[activity] != activity:
            parent[activity] = find(parent[activity])
        return parent[activity]

    def union(activity1, activity2):
        root1 = find(activity1)
        root2 = find(activity2)
        if root1 != root2:
            if rank[root1] > rank[root2]:
                parent[root2] = root1
            else:
                parent[root1] = root2
                if rank[root1] == rank[root2]:
                    rank[root2] += 1

    for pair in similar_pairs:
        union(pair[0], pair[1])

    clusters = {}
    for activity in parent:
        root = find(activity)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(activity)

    return clusters

def majority_voting(clusters):
    canonical_forms = {}
    for cluster in clusters:
        if len(clusters[cluster]) > 1:
            original_activities = df.loc[df['ProcessedActivity'].isin(clusters[cluster]), 'Activity'].tolist()
            canonical_form = max(set(original_activities), key = original_activities.count)
            canonical_forms[cluster] = canonical_form
    return canonical_forms

def calculate_detection_metrics(df, label_column):
    if label_column in df.columns:
        def normalize_label(label):
            if pd.isnull(label) or label == np.nan:
                return 0
            elif 'distorted' in label.lower():
                return 1
            else:
                return 0

        y_true = df[label_column].apply(normalize_label)
        y_pred = df['is_distorted']
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        print(f"=== Detection Performance Metrics ===")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"✓/✗ Precision threshold (≥ 0.6) met/not met")
    else:
        print("No labels available for metric calculation")

def integrity_check(df):
    distortion_clusters = len([cluster for cluster in clusters if len(clusters[cluster]) > 1])
    distorted_activities = df[df['is_distorted'] == 1].shape[0]
    clean_activities = df[df['is_distorted'] == 0].shape[0]
    print(f"Total distortion clusters detected: {distortion_clusters}")
    print(f"Total activities marked as distorted: {distorted_activities}")
    print(f"Activities to be fixed: {distorted_activities + clean_activities - distortion_clusters}")

def fix_activities(df, canonical_forms):
    df['Activity_fixed'] = df['ProcessedActivity'].map(canonical_forms)
    return df

def save_output(df):
    df.to_csv('data/bpic11/bpic11_distorted_cleaned_run2.csv', index=False)

# Configuration parameters
time_threshold = 2.0
require_same_resource = False
min_matching_events = 2
max_mismatches = 1
activity_suffix_pattern = r"(_signed\d*|_\d+)$"
ngram_size = 3
similarity_threshold = 0.56
case_sensitive = False
use_fuzzy_matching = False

# Load the data
input_file = './data/bpic11/BPIC11-Distorted.csv'
df = load_data(input_file)

# Preprocess activity names
df['ProcessedActivity'] = df['Activity'].apply(preprocess_activity)

# Filter out activities shorter than min_length characters
min_length = 4
df = df[df['ProcessedActivity'].str.len() >= min_length]

# Identify distorted activities
df['isdistorted'] = df['Activity'].str.endswith(':distorted')
df['BaseActivity'] = df['Activity'].str.replace(':distorted', '')

# Calculate Jaccard n-gram similarity
similar_pairs = find_similar_pairs(df, ngram_size, similarity_threshold)

# Cluster similar activities using union-find
clusters = union_find(similar_pairs)

# Majority voting within clusters
canonical_forms = majority_voting(clusters)

# Mark distorted activities
df['is_distorted'] = df['Activity'].apply(lambda x: 1 if x in canonical_forms.values() else 0)

# Calculate detection metrics
label_column = 'label'
calculate_detection_metrics(df, label_column)

# Integrity check
integrity_check(df)

# Fix activities
df = fix_activities(df, canonical_forms)

# Save output
save_output(df)

# Print summary
print(f"Run 2: Processed dataset saved to: data/bpic11/bpic11_distorted_cleaned_run2.csv")
print(f"Run 2: Final dataset shape: {df.shape}")
print(f"Run 2: Dataset: bpic11")
print(f"Run 2: Task type: distorted")
print(f"Run 2: Output file path: data/bpic11/bpic11_distorted_cleaned_run2.csv")