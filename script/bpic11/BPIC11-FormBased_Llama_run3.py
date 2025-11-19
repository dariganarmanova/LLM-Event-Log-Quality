# Generated script for BPIC11-FormBased - Run 3
# Generated on: 2025-11-13T11:17:49.444399
# Model: meta-llama/Llama-3.1-8B-Instruct

import pandas as pd
import numpy as np
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

# Input configuration
input_file = 'data/bpic11/BPIC11-FormBased.csv'
input_directory = 'data/bpic11'
dataset_name = 'bpic11'
output_suffix = '_form_based_cleaned_run3'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Time'
label_column = 'label'

# Load the data
try:
    df = pd.read_csv(input_file)
except FileNotFoundError:
    print("Error: Input file not found.")
    exit()

# Ensure all required columns exist; rename common variants
df = df.rename(columns={'CaseID': case_column, 'Timestamp': timestamp_column})
required_columns = [case_column, timestamp_column, activity_column, label_column]
if not all(col in df.columns for col in required_columns):
    print("Error: Required columns not found in the input file.")
    exit()

# Convert the Timestamp column to datetime and standardize as YYYY-MM-DD HH:MM:SS
df[timestamp_column] = pd.to_datetime(df[timestamp_column], format='%H:%M:%S')
df[timestamp_column] = df[timestamp_column].dt.strftime('%Y-%m-%d %H:%M:%S')

# Sort by Case and Timestamp to maintain chronological order
df = df.sort_values(by=[case_column, timestamp_column])

# Identify flattened events
group_key = df.groupby([case_column, timestamp_column]).groups.keys()
df['is_flattened'] = df.groupby([case_column, timestamp_column]).grouper.group_info[0] >= min_matching_events

# Preprocess flattened groups
normal_events = df[df['is_flattened'] == 0]
flattened_events = df[df['is_flattened'] == 1]

# Merge flattened activities
merged_activities = flattened_events.groupby([case_column, timestamp_column])[activity_column].apply(lambda x: ';'.join(sorted(x.unique())))
merged_events = pd.DataFrame({'Case': flattened_events[case_column].unique(), 
                              'Timestamp': flattened_events[timestamp_column].unique(), 
                              'Activity': merged_activities,
                              'label': flattened_events[label_column].unique()})
merged_events = merged_events.reset_index(drop=True)

# Combine and sort
final_df = pd.concat([normal_events, merged_events]).sort_values(by=[case_column, timestamp_column]).reset_index(drop=True)
final_df = final_df.drop(columns=['is_flattened'])

# Calculate detection metrics
if label_column in final_df.columns:
    y_true = final_df[label_column].notnull().astype(int)
    y_pred = final_df['is_flattened'].astype(int)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"✓/✗ Precision threshold (≥ 0.6) met/not met")
else:
    print("No labels available for metric calculation.")

# Integrity check
total_flattened_groups = len(final_df[final_df['is_flattened'] == 1].groupby([case_column, timestamp_column]).groups.keys())
total_flattened_events = final_df[final_df['is_flattened'] == 1].shape[0]
percentage_flattened_events = (total_flattened_events / final_df.shape[0]) * 100
print(f"Total flattened groups detected: {total_flattened_groups}")
print(f"Total events marked as flattened: {total_flattened_events}")
print(f"Percentage of flattened events: {percentage_flattened_events:.2f}%")

# Fix events
final_df = final_df.drop(columns=['is_flattened'])

# Save output
output_path = input_directory + '/' + dataset_name + output_suffix + '.csv'
final_df.to_csv(output_path, index=False)

# Summary statistics
total_events = final_df.shape[0]
unique_activities_before = df[activity_column].nunique()
unique_activities_after = final_df[activity_column].nunique()
total_reduction_percentage = ((unique_activities_before - unique_activities_after) / unique_activities_before) * 100
print(f"Total number of events: {total_events}")
print(f"Number of flattened (merged) events detected: {total_flattened_events}")
print(f"Number of unique activities before merging: {unique_activities_before}")
print(f"Number of unique activities after merging: {unique_activities_after}")
print(f"Total reduction percentage: {total_reduction_percentage:.2f}%")
print(f"Output file path: {output_path}")

# Print sample of up to 10 merged activities
merged_activities_sample = final_df[final_df['is_flattened'] == 1][activity_column].unique()
print(f"Sample of up to 10 merged activities: {merged_activities_sample[:10]}")

print(f"Run 3: Processed dataset saved to: {output_path}")
print(f"Run 3: Final dataset shape: {final_df.shape}")
print(f"Run 3: Dataset: {dataset_name}")
print(f"Run 3: Task type: form_based")