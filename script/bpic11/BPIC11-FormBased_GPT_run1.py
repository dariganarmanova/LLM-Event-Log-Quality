# Generated script for BPIC11-FormBased - Run 1
# Generated on: 2025-11-13T11:16:50.401746
# Model: gpt-4o-2024-11-20

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration parameters
input_file = 'data/bpic11/BPIC11-FormBased.csv'
output_file = 'data/bpic11/bpic11_form_based_cleaned_run1.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'

# Load the data
try:
    df = pd.read_csv(input_file)
    print(f"Run 1: Original dataset shape: {df.shape}")
except Exception as e:
    print(f"Error loading the file: {e}")
    exit()

# Ensure required columns exist
required_columns = [case_column, activity_column, timestamp_column]
for col in required_columns:
    if col not in df.columns:
        print(f"Missing required column: {col}")
        exit()

# Convert timestamp to datetime and standardize format
try:
    df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce')
    if df[timestamp_column].isnull().any():
        print("Warning: Some timestamps could not be converted and will be dropped.")
    df = df.dropna(subset=[timestamp_column])
except Exception as e:
    print(f"Error processing timestamps: {e}")
    exit()

# Sort by Case and Timestamp
df = df.sort_values(by=[case_column, timestamp_column]).reset_index(drop=True)

# Identify flattened events
df['group_key'] = df[case_column].astype(str) + "_" + df[timestamp_column].astype(str)
group_counts = df.groupby('group_key').size()
df['is_flattened'] = df['group_key'].map(group_counts).apply(lambda x: 1 if x >= 2 else 0)

# Split dataset into normal and flattened events
normal_events = df[df['is_flattened'] == 0].copy()
flattened_events = df[df['is_flattened'] == 1].copy()

# Merge flattened activities
merged_events = (
    flattened_events.groupby([case_column, timestamp_column])
    .agg({
        activity_column: lambda x: ';'.join(sorted(x)),
        label_column: 'first' if label_column in df.columns else lambda x: None
    })
    .reset_index()
)

# Combine normal and merged events
final_df = pd.concat([normal_events[[case_column, activity_column, timestamp_column, label_column]], merged_events], ignore_index=True)
final_df = final_df.sort_values(by=[case_column, timestamp_column]).reset_index(drop=True)

# Drop helper columns
final_df = final_df.drop(columns=['group_key', 'is_flattened'], errors='ignore')

# Save the processed data
try:
    final_df.to_csv(output_file, index=False)
    print(f"Run 1: Processed dataset saved to: {output_file}")
except Exception as e:
    print(f"Error saving the file: {e}")
    exit()

# Summary statistics
total_events = len(df)
flattened_groups = len(flattened_events['group_key'].unique())
flattened_events_count = len(flattened_events)
unique_activities_before = df[activity_column].nunique()
unique_activities_after = final_df[activity_column].nunique()
reduction_percentage = (1 - len(final_df) / total_events) * 100

print(f"Run 1: Summary Statistics")
print(f"Total events: {total_events}")
print(f"Flattened groups detected: {flattened_groups}")
print(f"Flattened events: {flattened_events_count}")
print(f"Unique activities before merging: {unique_activities_before}")
print(f"Unique activities after merging: {unique_activities_after}")
print(f"Total reduction in events: {reduction_percentage:.2f}%")
print(f"Final dataset shape: {final_df.shape}")

# Detection metrics (if label column exists)
if label_column in df.columns:
    y_true = df[label_column].notnull().astype(int)
    y_pred = df['is_flattened']
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print("=== Detection Performance Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"{'✓' if precision >= 0.6 else '✗'} Precision threshold (≥ 0.6) met")
else:
    print("No labels available for metric calculation.")
    print("Precision: 0.0000")
    print("Recall: 0.0000")
    print("F1-Score: 0.0000")