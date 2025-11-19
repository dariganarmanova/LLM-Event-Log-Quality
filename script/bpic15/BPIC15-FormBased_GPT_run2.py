# Generated script for BPIC15-FormBased - Run 2
# Generated on: 2025-11-13T14:24:43.181642
# Model: gpt-4o-2024-11-20

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration parameters
input_file = 'data/bpic15/BPIC15-FormBased.csv'
output_file = 'data/bpic15/bpic15_form_based_cleaned_run2.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'

# Load the data
try:
    df = pd.read_csv(input_file)
    print(f"Run 2: Original dataset shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: File not found at {input_file}")
    exit()
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# Ensure required columns exist
required_columns = [case_column, activity_column, timestamp_column]
for col in required_columns:
    if col not in df.columns:
        print(f"Error: Missing required column '{col}' in the dataset.")
        exit()

# Convert timestamp to datetime and standardize format
df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce')
if df[timestamp_column].isnull().any():
    print("Warning: Some timestamps could not be parsed and will be dropped.")
df = df.dropna(subset=[timestamp_column])
df[timestamp_column] = df[timestamp_column].dt.strftime('%Y-%m-%d %H:%M:%S')

# Sort by case and timestamp
df = df.sort_values(by=[case_column, timestamp_column]).reset_index(drop=True)

# Identify flattened events
df['group_key'] = df[case_column].astype(str) + '_' + df[timestamp_column]
group_sizes = df.groupby('group_key').size()
df['is_flattened'] = df['group_key'].map(group_sizes)
df['is_flattened'] = (df['is_flattened'] >= 2).astype(int)

# Split dataset into normal and flattened events
normal_events = df[df['is_flattened'] == 0].copy()
flattened_events = df[df['is_flattened'] == 1].copy()

# Merge flattened activities
merged_flattened = (
    flattened_events.groupby([case_column, timestamp_column])
    .agg({
        activity_column: lambda x: ';'.join(sorted(x)),
        label_column: 'first' if label_column in df.columns else lambda x: None
    })
    .reset_index()
)

# Combine normal and merged flattened events
final_df = pd.concat([normal_events, merged_flattened], ignore_index=True)
final_df = final_df.sort_values(by=[case_column, timestamp_column]).reset_index(drop=True)

# Drop helper columns
final_df = final_df[[case_column, activity_column, timestamp_column] + ([label_column] if label_column in df.columns else [])]

# Calculate detection metrics if label column exists
if label_column in df.columns:
    y_true = (df[label_column].notnull()).astype(int)
    y_pred = df['is_flattened']
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print("=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"{'✓' if precision >= 0.6 else '✗'} Precision threshold (≥ 0.6) {'met' if precision >= 0.6 else 'not met'}")
else:
    print("No labels available for metric calculation.")
    precision, recall, f1 = 0.0, 0.0, 0.0

# Integrity check
total_flattened_groups = len(merged_flattened)
total_flattened_events = len(flattened_events)
percentage_flattened = (total_flattened_events / len(df)) * 100
print(f"Total flattened groups detected: {total_flattened_groups}")
print(f"Total events marked as flattened: {total_flattened_events}")
print(f"Percentage of flattened events: {percentage_flattened:.2f}%")

# Save the processed data
try:
    final_df.to_csv(output_file, index=False)
    print(f"Run 2: Processed dataset saved to: {output_file}")
    print(f"Run 2: Final dataset shape: {final_df.shape}")
    print(f"Run 2: Dataset: bpic15")
    print(f"Run 2: Task type: form_based")
except Exception as e:
    print(f"Error saving file: {e}")
    exit()

# Summary statistics
unique_activities_before = df[activity_column].nunique()
unique_activities_after = final_df[activity_column].nunique()
total_reduction = ((len(df) - len(final_df)) / len(df)) * 100
print(f"Total number of events: {len(df)}")
print(f"Number of flattened (merged) events detected: {total_flattened_groups}")
print(f"Number of unique activities before merging: {unique_activities_before}")
print(f"Number of unique activities after merging: {unique_activities_after}")
print(f"Total reduction percentage: {total_reduction:.2f}%")
print("Sample of merged activities:")
print(final_df[final_df[activity_column].str.contains(';')].head(10))