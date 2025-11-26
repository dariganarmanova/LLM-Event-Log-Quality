# Generated script for Pub-Formbased - Run 3
# Generated on: 2025-11-18T18:39:46.268120
# Model: grok-4-fast

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

input_file = 'data/pub/Pub-Formbased.csv'
case_column = 'Case'
activity_column = 'Activity'
timestamp_column = 'Timestamp'
label_column = 'label'

df = pd.read_csv(input_file)
print(f"Run 3: Original dataset shape: {df.shape}")

df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce')
df = df.sort_values([case_column, timestamp_column]).reset_index(drop=True)

df['group_size'] = df.groupby([case_column, timestamp_column])[case_column].transform('size')
is_flattened = (df['group_size'] >= 2).astype(int)

if label_column in df.columns:
    y_true = (df[label_column].notna() & (df[label_column].astype(str).str.strip() != '')).astype(int)
    prec = precision_score(y_true, is_flattened)
    rec = recall_score(y_true, is_flattened)
    f1 = f1_score(y_true, is_flattened)
    print("=== Detection Performance Metrics ===")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    if prec >= 0.6:
        print("✓ Precision threshold (>= 0.6) met")
    else:
        print("✗ Precision threshold (>= 0.6) not met")
else:
    print("=== Detection Performance Metrics ===")
    print("Precision: 0.0000")
    print("Recall: 0.0000")
    print("F1-Score: 0.0000")
    print("No labels available for metric calculation.")

flattened_groups = len(df[df['group_size'] >= 2].groupby([case_column, timestamp_column]))
total_flattened_events = (df['group_size'] >= 2).sum()
percentage = (total_flattened_events / len(df)) * 100 if len(df) > 0 else 0
print(f"Total flattened groups detected: {flattened_groups}")
print(f"Total events marked as flattened: {total_flattened_events}")
print(f"Percentage of flattened events: {percentage:.2f}%")

normal_events = df[df['group_size'] < 2].drop(columns=['group_size']).copy()
flattened_events = df[df['group_size'] >= 2].copy()

preserve_cols = [col for col in df.columns if col not in [case_column, timestamp_column, activity_column, 'group_size']]
agg_dict = {activity_column: (lambda x: ';'.join(sorted(x.dropna().astype(str).unique())))}
for col in preserve_cols:
    agg_dict[col] = 'first'
merged_df = flattened_events.groupby([case_column, timestamp_column]).agg(agg_dict).reset_index()

final_df = pd.concat([normal_events, merged_df], ignore_index=True)
final_df = final_df.sort_values([case_column, timestamp_column]).reset_index(drop=True)
final_df[timestamp_column] = final_df[timestamp_column].dt.strftime('%Y-%m-%d %H:%M:%S')

print(f"Total number of events: {len(final_df)}")
print(f"Number of flattened (merged) events detected: {flattened_groups}")
unique_before = df[activity_column].nunique()
def count_unique_after(series):
    uniques = set()
    for val in series:
        if pd.isna(val):
            continue
        acts = str(val).split(';')
        uniques.update(acts)
    return len(uniques)
unique_after = count_unique_after(final_df[activity_column])
print(f"Number of unique activities before vs after merging: {unique_before} vs {unique_after}")
reduction = ((len(df) - len(final_df)) / len(df)) * 100 if len(df) > 0 else 0
print(f"Total reduction percentage: {reduction:.2f}%")
print(f"Output file path: data/pub/pub_form_based_cleaned_run3.csv")

merged_samples = final_df[final_df[activity_column].str.contains(';', na=False)][activity_column].head(10)
if len(merged_samples) > 0:
    print("Sample of up to 10 merged activities (`;` separated):")
    for i, act in enumerate(merged_samples, 1):
        print(f"{i}. {act}")
else:
    print("No merged activities found.")

final_df.to_csv('data/pub/pub_form_based_cleaned_run3.csv', index=False)
print(f"Run 3: Processed dataset saved to: data/pub/pub_form_based_cleaned_run3.csv")
print(f"Run 3: Final dataset shape: {final_df.shape}")
print(f"Run 3: Dataset: pub")
print(f"Run 3: Task type: form_based")