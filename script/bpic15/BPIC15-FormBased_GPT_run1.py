# Generated script for BPIC15-FormBased - Run 1
# Generated on: 2025-11-13T14:24:32.557673
# Model: gpt-4o-2024-11-20

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration parameters
input_file = 'data/bpic15/BPIC15-FormBased.csv'
output_file = 'data/bpic15/bpic15_form_based_cleaned_run1.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'

# Load the data
try:
    df = pd.read_csv(input_file)
    print(f"Run 1: Original dataset shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: File not found at {input_file}")
    exit()
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# Ensure required columns exist
required_columns = {case_column, activity_column, timestamp_column}
if not required_columns.issubset(df.columns):
    print(f"Error: Missing required columns. Required: {required_columns}, Found: {df.columns}")
    exit()

# Convert timestamp to datetime and standardize format
df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce')
if df[timestamp_column].isnull().any():
    print("Warning: Some timestamps could not be parsed and were set to NaT.")
df = df.dropna(subset=[timestamp_column])  # Drop rows with invalid timestamps
df[timestamp_column] = df[timestamp_column].dt.strftime('%Y-%m-%d %H:%M:%S')

# Sort by case and timestamp
df = df.sort_values(by=[case_column, timestamp_column]).reset_index(drop=True)

# Identify flattened events
df['group_key'] = df[case_column].astype(str) + '_' + df[timestamp_column]
group_counts = df.groupby('group_key').size()
df['is_flattened'] = df['group_key'].map(group_counts).apply(lambda x: 1 if x >= 2 else 0)

# Split dataset into normal and flattened events
normal_events = df[df['is_flattened'] == 0].copy()
flattened_events = df[df['is_flattened'] == 1].copy()

# Merge flattened events
merged_events = (
    flattened_events.groupby([case_column, timestamp_column])
    .agg({
        activity_column: lambda x: ';'.join(sorted(x)),
        label_column: 'first' if label_column in df.columns else lambda x: None
    })
    .reset_index()
)

# Combine normal and merged events
final_df = pd.concat([normal_events[[case_column, activity_column, timestamp_column, label_column]],
                      merged_events], ignore_index=True)

# Sort final DataFrame by case and timestamp
final_df = final_df.sort_values(by=[case_column, timestamp_column]).reset_index(drop=True)

# Save the processed data
try:
    final_df.to_csv(output_file, index=False)
    print(f"Run 1: Processed dataset saved to: {output_file}")
except Exception as e:
    print(f"Error saving file: {e}")
    exit()

# Print summary statistics
total_events = len(df)
flattened_events_count = len(flattened_events)
merged_events_count = len(merged_events)
unique_activities_before = df[activity_column].nunique()
unique_activities_after = final_df[activity_column].nunique()
reduction_percentage = (1 - len(final_df) / total_events) * 100

print(f"Run 1 Summary:")
print(f"Total events: {total_events}")
print(f"Flattened events detected: {flattened_events_count}")
print(f"Merged events: {merged_events_count}")
print(f"Unique activities before merging: {unique_activities_before}")
print(f"Unique activities after merging: {unique_activities_after}")
print(f"Reduction in total events: {reduction_percentage:.2f}%")
print(f"Final dataset shape: {final_df.shape}")

# Display sample of merged activities
print("Sample of merged activities:")
print(merged_events.head(10))