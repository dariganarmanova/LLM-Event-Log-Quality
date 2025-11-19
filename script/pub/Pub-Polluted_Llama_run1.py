# Generated script for Pub-Polluted - Run 1
# Generated on: 2025-11-14T13:38:17.849827
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
aggressive_token_limit = 3
min_variants = 2
normalization_strategy = 'aggressive'
save_detection_file = False

# Load the data
input_file = 'data/pub/Pub-Polluted.csv'
df = pd.read_csv(input_file)
print(f"Run 1: Original dataset shape: {df.shape}")

# Normalize column names for 'Case'/'CaseID' if present (rename to 'Case')
df = df.rename(columns={'CaseID': 'Case'})

# Store original activities in 'original_activity' column for reference
df['original_activity'] = df['Activity']

# Print: Number of unique activities before fixing
print(f"Run 1: Unique activities before fixing: {df['Activity'].nunique()}")

# Define aggressive normalization function
def aggressive_normalize(activity):
    # Lowercase
    activity = activity.lower()
    
    # Replace punctuation with spaces
    activity = re.sub(r'[^\w\s]', ' ', activity)
    
    # Remove alphanumeric ID-like tokens
    activity = re.sub(r'\b\w\d+\b', '', activity)
    
    # Remove long digit strings
    activity = re.sub(r'\d{5,}', '', activity)
    
    # Collapse whitespace
    activity = re.sub(r'\s+', ' ', activity).strip()
    
    # Token limiting
    tokens = activity.split()
    activity = ' '.join(tokens[:aggressive_token_limit])
    
    # Return the joined tokens; for NaN return empty string
    return activity if pd.notna(activity) else ''

# Apply normalization
df['BaseActivity'] = df['Activity'].apply(aggressive_normalize)
df = df.dropna(subset=['BaseActivity'])  # Drop rows with empty 'BaseActivity'
print(f"Run 1: Unique base activities count: {df['BaseActivity'].nunique()}")

# Detect polluted groups
polluted_bases = df.groupby('BaseActivity')['Activity'].nunique().apply(lambda x: x > min_variants).index
print(f"Run 1: Number of polluted groups found: {len(polluted_bases)}")

# Flag polluted events
df['is_polluted_label'] = df['BaseActivity'].isin(polluted_bases)
print(f"Run 1: Polluted events count: {df['is_polluted_label'].sum()}")
print(f"Run 1: Clean events count: {~df['is_polluted_label'].sum()}")
print(f"Run 1: Pollution rate: {(df['is_polluted_label'].sum() / len(df)) * 100:.2f}%")

# Calculate detection metrics (BEFORE FIXING)
if 'label' in df.columns:
    y_true = (df['label'].notna()) | (~df['label'].isnull())
    y_pred = df['is_polluted_label']
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"✓ Precision threshold (≥ 0.6) met" if precision >= 0.6 else "✗ Precision threshold (≥ 0.6) not met")
else:
    print("No ground-truth labels found, skipping evaluation.")

# Integrity check
total_polluted_bases = len(polluted_bases)
total_events_flagged = df['is_polluted_label'].sum()
total_events_clean = ~df['is_polluted_label'].sum()
print(f"Run 1: Total polluted bases detected: {total_polluted_bases}")
print(f"Run 1: Total events flagged as polluted: {total_events_flagged}")
print(f"Run 1: Total clean events: {total_events_clean}")

# Fix activities
df.loc[df['is_polluted_label'], 'Activity'] = df.loc[df['is_polluted_label'], 'BaseActivity']

# Save fixed output
df_fixed = df.drop(['original_activity', 'BaseActivity', 'is_polluted_label'], axis=1)
output_file = f"data/pub/pub_polluted_cleaned_run1.csv"
df_fixed.to_csv(output_file, index=False)
print(f"Run 1: Processed dataset saved to: {output_file}")
print(f"Run 1: Final dataset shape: {df_fixed.shape}")
print(f"Run 1: Dataset: pub")
print(f"Run 1: Task type: polluted")

# Print summary statistics
print(f"Run 1: Normalization strategy: {normalization_strategy}")
print(f"Run 1: Total rows: {df_fixed.shape[0]}")
print(f"Run 1: Labels replaced count: {df_fixed['Activity'].ne(df['Activity']).sum()}")
print(f"Run 1: Replacement rate: {(df_fixed['Activity'].ne(df['Activity']).sum() / df.shape[0]) * 100:.2f}%")
print(f"Run 1: Unique activities before → after: {df['Activity'].nunique()} → {df_fixed['Activity'].nunique()}")
print(f"Run 1: Activity reduction count and percentage: {df['Activity'].nunique() - df_fixed['Activity'].nunique()} ({((df['Activity'].nunique() - df_fixed['Activity'].nunique()) / df['Activity'].nunique()) * 100:.2f}%)")

# Print up to 10 sample transformations
print("Run 1: Sample transformations:")
for i in range(min(10, df_fixed.shape[0])):
    print(f"Original: {df.loc[i, 'original_activity']} → {df_fixed.loc[i, 'Activity']}")