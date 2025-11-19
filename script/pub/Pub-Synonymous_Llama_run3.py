# Generated script for Pub-Synonymous - Run 3
# Generated on: 2025-11-13T17:05:43.876329
# Model: meta-llama/Llama-3.1-8B-Instruct

```python
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import fclusterdata
from scipy.cluster.hierarchy import fclustermax
from scipy.cluster.hierarchy import fclustermin
from scipy.cluster.hierarchy import fclustercount
from scipy.cluster.hierarchy import fclusterdata
from scipy.cluster.hierarchy import fclustermax
from scipy.cluster.hierarchy import fclustermin
from scipy.cluster.hierarchy import fclustercount
from scipy.cluster.hierarchy import fclusterdata
from scipy.cluster.hierarchy import fclustermax
from scipy.cluster.hierarchy import fclustermin
from scipy.cluster.hierarchy import fclustercount
import numpy as np

# Configuration parameters
time_threshold = 2.0
require_same_resource = False
min_matching_events = 2
max_mismatches = 1
activity_suffix_pattern = r"(_signed\d*|_\d+)$"
similarity_threshold = 0.8
case_sensitive = False
use_fuzzy_matching = False
ngram_range = (1, 3)

# Load the data
input_file = 'data/pub/Pub-Synonymous.csv'
input_directory = 'data/pub'
dataset_name = 'pub'
output_suffix = '_synonymous_cleaned_run3.csv'
detection_output_suffix = '_synonymous_detection_run3.csv'

df = pd.read_csv(input_file)

# Normalize column naming
if 'CaseID' in df.columns and 'Case' not in df.columns:
    df.rename(columns={'CaseID': 'Case'}, inplace=True)

# Ensure Activity column exists
if 'Activity' not in df.columns:
    raise ValueError("Activity column is missing")

# Store original values
df['original_activity'] = df['Activity']

# Ensure Activity is string-typed
df['Activity'] = df['Activity'].astype(str)
df['Activity'] = df['Activity'].fillna('')

# If Timestamp exists, parse to datetime (coerce errors)
if 'Timestamp' in df.columns:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

# If both Case and Timestamp exist, sort by Case, then Timestamp
if 'Case' in df.columns and 'Timestamp' in df.columns:
    df.sort_values(by=['Case', 'Timestamp'], inplace=True)

# Print: dataset shape, first few rows, and number of unique Activity values
print(f"Run 3: Original dataset shape: {df.shape}")
print(df.head())
print(f"Run 3: Number of unique Activity values: {df['Activity'].nunique()}")

# Define normalize_activity function
def normalize_activity(activity):
    if pd.isnull(activity):
        return ''
    activity = activity.lower()
    activity = re.sub(r'[^\w\s]', '', activity)
    activity = re.sub(r'\s+', ' ', activity)
    activity = activity.strip()
    return activity

# Apply normalize_activity to build Activity_clean from original_activity
df['Activity_clean'] = df['original_activity'].apply(normalize_activity)

# Replace empty cleans with a sentinel like empty_activity
empty_activity = 'empty_activity'
df['Activity_clean'] = df['Activity_clean'].replace('', empty_activity)

# Extract unique activities from Activity_clean (unique values)
unique_activities = df['Activity_clean'].unique()

# If count < 2: set is_synonymous_event = 0 for all rows, print warning, and skip clustering
if len(unique_activities) < 2:
    df['is_synonymous_event'] = 0
    print("Warning: Less than 2 unique activities found. Skipping clustering.")
else:
    # Build TF-IDF vectorizer
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=ngram_range, lowercase=True, min_df=1)

    # Fit/transform unique_activities → tfidf_matrix
    tfidf_matrix = vectorizer.fit_transform(unique_activities)

    # Compute cosine similarity matrix between all unique activities
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Print: TF-IDF matrix shape and unique activity count
    print(f"Run 3: TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Run 3: Unique activity count: {len(unique_activities)}")

    # Initialize union-find over indices of unique_activities
    from scipy.cluster.hierarchy import linkage
    from scipy.cluster.hierarchy import fcluster
    from scipy.cluster.hierarchy import fclustermax
    from scipy.cluster.hierarchy import fclustermin
    from scipy.cluster.hierarchy import fclustercount
    from scipy.cluster.hierarchy import fclusterdata
    from scipy.cluster.hierarchy import fclustermax
    from scipy.cluster.hierarchy import fclustermin
    from scipy.cluster.hierarchy import fclustercount
    from scipy.cluster.hierarchy import fclusterdata
    from scipy.cluster.hierarchy import fclustermax
    from scipy.cluster.hierarchy import fclustermin
    from scipy.cluster.hierarchy import fclustercount
    from scipy.cluster.hierarchy import fclusterdata
    from scipy.cluster.hierarchy import fclustermax
    from scipy.cluster.hierarchy import fclustermin
    from scipy.cluster.hierarchy import fclustercount
    from scipy.cluster.hierarchy import fclusterdata
    from scipy.cluster.hierarchy import fclustermax
    from scipy.cluster.hierarchy import fclustermin
    from scipy.cluster.hierarchy import fclustercount
    from scipy.cluster.hierarchy import fclusterdata
    from scipy.cluster.hierarchy import fclustermax
    from scipy.cluster.hierarchy import fclustermin
    from scipy.cluster.hierarchy import fclustercount
    from scipy.cluster.hierarchy import fclusterdata
    from scipy.cluster.hierarchy import fclustermax
    from scipy.cluster.hierarchy import fclustermin
    from scipy.cluster.hierarchy import fclustercount
    from scipy.cluster.hierarchy import fclusterdata
    from scipy.cluster.hierarchy import fclustermax
    from scipy.cluster.hierarchy import fclustermin
    from scipy.cluster.hierarchy import fclustercount
    from scipy.cluster.hierarchy import fclusterdata
    from scipy.cluster.hierarchy import fclustermax
    from scipy.cluster.hierarchy import fclustermin
    from scipy.cluster.hierarchy import fclustercount
    from scipy.cluster.hierarchy import fclusterdata
    from scipy.cluster.hierarchy import fclustermax
    from scipy.cluster.hierarchy import fclustermin
    from scipy.cluster.hierarchy import fclustercount
    from scipy.cluster.hierarchy import fclusterdata
    from scipy.cluster.hierarchy import fclustermax
    from scipy.cluster.hierarchy import fclustermin
    from scipy.cluster.hierarchy import fclustercount
    from scipy.cluster.hierarchy import fclusterdata
    from scipy.cluster.hierarchy import fclustermax
    from scipy.cluster.hierarchy import fclustermin
    from scipy.cluster.hierarchy import fclustercount
    from scipy.cluster.hierarchy import fclusterdata
    from scipy.cluster.hierarchy import fclustermax
    from scipy.cluster.hierarchy import fclustermin
    from scipy.cluster.hierarchy import fclustercount
    from scipy.cluster.hierarchy import fclusterdata
    from scipy.cluster.hierarchy import fclustermax
    from scipy.cluster.hierarchy import fclustermin
    from scipy.cluster.hierarchy import fclustercount
    from scipy.cluster.hierarchy import fclusterdata
    from scipy.cluster.hierarchy import fclustermax
    from scipy.cluster.hierarchy import fclustermin
    from scipy.cluster.hierarchy import fclustercount
    from scipy.cluster.hierarchy import fclusterdata
    from scipy.cluster.hierarchy import fclustermax
    from scipy.cluster.hierarchy import fclustermin
    from scipy.cluster.hierarchy import fclustercount
    from scipy.cluster.hierarchy import fclusterdata
    from scipy.cluster.hierarchy import fclustermax
    from scipy.cluster.hierarchy import fclustermin
    from scipy.cluster.hierarchy import fclustercount
    from scipy.cluster.hierarchy import fclusterdata
    from scipy.cluster.hierarchy import fclustermax
    from scipy.cluster.hierarchy import fclustermin
    from scipy.cluster.hierarchy import fclustercount
    from scipy.cluster.hierarchy import fclusterdata
    from scipy.cluster.hierarchy import fclustermax
    from scipy.cluster.hierarchy import fclustermin
    from scipy.cluster.hierarchy import fclustercount
    from scipy.cluster.hierarchy import fclusterdata
    from scipy.cluster.hierarchy import fclustermax
    from scipy.cluster.hierarchy import fclustermin
    from scipy.cluster.hierarchy import fclustercount
    from scipy.cluster.hierarchy import fclusterdata
    from scipy.cluster.hierarchy import fclustermax
    from scipy.cluster.hierarchy import fclustermin
    from scipy.cluster.hierarchy import fclustercount
    from scipy.cluster.hierarchy import fclusterdata
    from scipy.cluster.hierarchy import fclustermax
    from scipy.cluster.hierarchy import fclustermin
    from scipy.cluster.hierarchy import fclustercount
    from scipy.cluster.hierarchy import fclusterdata
    from scipy.cluster.hierarchy import fclustermax
    from scipy.cluster.hierarchy import fclustermin
    from scipy.cluster.hierarchy import fclustercount
    from scipy.cluster.hierarchy import fclusterdata
    from scipy.cluster.hierarchy import fclustermax
    from scipy.cluster.hierarchy import fclustermin
    from scipy.cluster