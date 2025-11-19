# Generated script for Pub-Formbased - Run 2
# Generated on: 2025-11-18T18:39:12.984708
# Model: grok-4-fast

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration
input_file = 'data/pub/Pub-Formbased.csv'
output_file = 'data/pub/pub_form_based_cleaned_run2.csv'
case_col = 'Case'
act_col = 'Activity'
ts_col = 'Timestamp'
label_col = 'label'
time_threshold = 2.0
require_same_resource = False
min_matching_events = 2
max_mismatches = 1
activity_suffix_pattern = r'(_signed\d*|_\d+)$'
similarity_threshold = 0.8
case_sensitive = False
use_fuzzy_matching = False

# Step 1: Load CSV
df = pd.read_csv(input_file)
print(f"Run 2: Original dataset shape: {df.shape}")

# Rename common variants if needed
column_mapping = {
    'CaseID': 'Case',
    'case': 'Case',
    'ActivityID': 'Activity',
    'activity': 'Activity',
    'time': 'Timestamp',
    'timestamp': 'Timestamp'
}
df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

# Ensure required columns exist
required_cols = [case_col, act_col, ts_col]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

# Convert Timestamp to datetime and standardize
df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
df[ts_col] = df[ts_col].fillna(pd.Timestamp('1900-01-01')).dt.strftime('%Y-%m-%d %H:%M:%S')

# Sort by Case and Timestamp
df = df.sort_values([case_col, ts_col]).reset_index(drop=True)

# Step 2: Identify Flattened Events
df['group_key'] = df[case_col].astype(str) + '_' + df[ts_col]
group_counts = df.groupby('group_key').size()
df['is_flattened'] = df['group_key'].map(lambda g: 1 if group_counts.get(g, 0) >= min_matching_events else 0)

# Step 6: Calculate Detection Metrics (BEFORE FIXING)
if label_col in df.columns:
    y_true = ((df[label_col].notna()) & (df[label_col].astype(str) != '')).astype(int)
    y_pred = df['is_flattened']
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec_mark = '✓' if precision >= 0.6 else '✗'
    print('=== Detection Performance Metrics ===')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-Score: {f1:.4f}')
    print(f'{prec_mark} Precision threshold (>= 0.6) met')
else:
    print('=== Detection Performance Metrics ===')
    print('Precision: 0.0000')
    print('Recall: 0.0000')
    print('F1-Score: 0.0000')
    print('No labels available for metric calculation.')

# Step 7: Integrity Check
num_flattened_groups = (group_counts >= min_matching_events).sum()
total_flattened_events = group_counts[group_counts >= min_matching_events].sum()
percentage = (total_flattened_events / len(df)) * 100 if len(df) > 0 else 0
print(f'Total flattened groups detected: {num_flattened_groups}')
print(f'Total events marked as flattened: {total_flattened_events}')
print(f'Percentage of flattened events: {percentage:.2f}%')

# Step 3: Preprocess Flattened Groups
normal_events = df[df['is_flattened'] == 0].drop(['group_key', 'is_flattened'], axis=1).copy()
flattened_events = df[df['is_flattened'] == 1].drop(['group_key', 'is_flattened'], axis=1).copy()

# Step 4: Merge Flattened Activities
if len(flattened_events) > 0:
    agg_dict = {
        act_col: lambda x: ';'.join(sorted(set(x.dropna().astype(str))))
    }
    optional_cols = [label_col, 'Variant', 'Resource']
    for opt_col in optional_cols:
        if opt_col in df.columns:
            agg_dict[opt_col] = 'first'
    merged_df = flattened_events.groupby([case_col, ts_col]).agg(agg_dict).reset_index()
else:
    # Create empty df with required columns
    merged_df = pd.DataFrame(columns=[case_col, ts_col, act_col])

# Step 5: Combine and Sort
final_df = pd.concat([normal_events, merged_df], ignore_index=True)
final_df = final_df.sort_values([case_col, ts_col]).reset_index(drop=True)

# Step 8: Fix Events (already done via merging)

# Step 10: Summary Statistics
total_events_after = len(final_df)
unique_acts_before = df[act_col].nunique()
unique_acts_after = final_df[act_col].nunique()
reduction_pct = (1 - total_events_after / len(df)) * 100 if len(df) > 0 else 0
print(f'Total number of events: {total_events_after}')
print(f'Number of flattened (merged) events detected: {num_flattened_groups}')
print(f'Number of unique activities before vs after merging: {unique_acts_before} vs {unique_acts_after}')
print(f'Total reduction percentage: {reduction_pct:.2f}%')
merged_samples = final_df[final_df[act_col].str.contains(';', na=False)][act_col].head(10).tolist()
print('Sample of up to 10 merged activities (;-separated):')
for sample in merged_samples:
    print(f'  - {sample}')
print(f'Output file path: {output_file}')

# Step 9: Save Output
# Select core columns, include optional if present
output_cols = [case_col, act_col, ts_col]
optional_output = [col for col in [label_col, 'Variant', 'Resource'] if col in final_df.columns]
output_cols.extend(optional_output)
final_df[output_cols].to_csv(output_file, index=False)

# Required prints
print(f"Run 2: Processed dataset saved to: data/pub/pub_form_based_cleaned_run2.csv")
print(f"Run 2: Final dataset shape: {final_df.shape}")
print(f"Run 2: Dataset: pub")
print(f"Run 2: Task type: form_based")