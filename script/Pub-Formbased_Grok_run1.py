# Generated script for Pub-Formbased - Run 1
# Generated on: 2025-11-18T18:38:42.191667
# Model: grok-4-fast

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

# Algorithm Configuration Parameters
time_threshold = 2.0
require_same_resource = False
min_matching_events = 2
max_mismatches = 1
activity_suffix_pattern = r"(_signed\d*|_\d+)$"
similarity_threshold = 0.8
case_sensitive = False
use_fuzzy_matching = False

# Load the data
input_file = 'data/pub/Pub-Formbased.csv'
df = pd.read_csv(input_file)
print(f"Run 1: Original dataset shape: {df.shape}")

# Rename common column variants
column_mapping = {
    'CaseID': 'Case',
    'case': 'Case',
    'Activity': 'Activity',
    'activity': 'Activity',
    'event': 'Activity',
    'Timestamp': 'Timestamp',
    'time': 'Timestamp',
    'timestamp': 'Timestamp',
    'label': 'label'
}
df = df.rename(columns=column_mapping)

# Ensure required columns exist
required = ['Case', 'Activity', 'Timestamp']
for col in required:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# Check for optional columns
has_label = 'label' in df.columns
has_variant = 'Variant' in df.columns
has_resource = 'Resource' in df.columns

# Convert Timestamp to datetime and standardize format
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
df['Timestamp'] = df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

# Sort by Case and Timestamp
df = df.sort_values(['Case', 'Timestamp']).reset_index(drop=True)

# Step 2: Identify Flattened Events
df['group_key'] = df['Case'].astype(str) + '_' + df['Timestamp'].astype(str)
group_sizes = df.groupby('group_key').size()
df['is_flattened'] = df['group_key'].map(lambda x: 1 if group_sizes.get(x, 0) >= min_matching_events else 0)

# Step 6: Calculate Detection Metrics (BEFORE FIXING)
print("=== Detection Performance Metrics ===")
if has_label:
    y_true = ((df['label'].notna()) & (df['label'] != '')).astype(int)
    y_pred = df['is_flattened']
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    prec_mark = '✓' if precision >= 0.6 else '✗'
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"{prec_mark} Precision threshold (>= 0.6) met")
else:
    print("Precision: 0.0000")
    print("Recall: 0.0000")
    print("F1-Score: 0.0000")
    print("✗ No labels available for metric calculation.")

# Step 3: Preprocess Flattened Groups
normal_events = df[df['is_flattened'] == 0].copy()
flattened_events = df[df['is_flattened'] == 1].copy()

# Step 7: Integrity Check
num_flattened_groups = len(group_sizes[group_sizes >= min_matching_events])
total_flattened_events = len(flattened_events)
percentage = (total_flattened_events / len(df)) * 100 if len(df) > 0 else 0
print(f"Total flattened groups detected: {num_flattened_groups}")
print(f"Total events marked as flattened: {total_flattened_events}")
print(f"Percentage of flattened events: {percentage:.2f}%")

# Step 4: Merge Flattened Activities
merged_df = pd.DataFrame()
if len(flattened_events) > 0:
    merge_cols = ['Case', 'Timestamp']
    def merge_activities(x):
        if case_sensitive:
            return ';'.join(sorted(x))
        else:
            return ';'.join(sorted(x, key=str.lower))
    agg_dict = {'Activity': merge_activities}
    if has_label:
        agg_dict['label'] = 'first'
    if has_variant:
        agg_dict['Variant'] = 'first'
    if has_resource:
        agg_dict['Resource'] = 'first'
    merged_df = flattened_events.groupby(merge_cols).agg(agg_dict).reset_index()

# Step 5: Combine and Sort
final_df = pd.concat([normal_events, merged_df], ignore_index=True)
final_df = final_df.sort_values(['Case', 'Timestamp']).reset_index(drop=True)
# Drop helper columns
cols_to_drop = ['group_key', 'is_flattened']
for col in cols_to_drop:
    if col in final_df.columns:
        final_df = final_df.drop(col, axis=1)

# Step 10: Summary Statistics
original_events = len(df)
final_events = len(final_df)
num_merged = num_flattened_groups
unique_acts_before = df['Activity'].nunique()
unique_acts_after = final_df['Activity'].nunique()
reduction = ((original_events - final_events) / original_events * 100) if original_events > 0 else 0
print(f"Total number of events: {original_events}")
print(f"Number of flattened (merged) events detected: {num_merged}")
print(f"Number of unique activities before: {unique_acts_before} vs after: {unique_acts_after}")
print(f"Total reduction percentage: {reduction:.2f}%")
print("Sample of up to 10 merged activities (;-separated):")
merged_samples = final_df[final_df['Activity'].str.contains(';', na=False)]['Activity'].head(10).tolist()
for act in merged_samples:
    print(f"  - {act}")
if not merged_samples:
    print("  - No merged activities found.")

# Step 9: Save Output
output_path = 'data/pub/pub_form_based_cleaned_run1.csv'
# Include all relevant columns: required + optional
final_df.to_csv(output_path, index=False)

# Required prints
print(f"Run 1: Processed dataset saved to: data/pub/pub_form_based_cleaned_run1.csv")
print(f"Run 1: Final dataset shape: {final_df.shape}")
print(f"Run 1: Dataset: pub")
print(f"Run 1: Task type: form_based")