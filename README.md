# LLM-Based Event Log Quality Assessment

Automated detection and repair of imperfection patterns in process mining event logs using Large Language Models (gpt-4o, Llama-3.1-8B-Instruct, DeepSeek-V3-0324, grok-4-fast).

## Overview

Event logs are fundamental data structures in process mining, but they often contain imperfections that compromise analysis quality. This project investigates whether modern Large Language Models can automatically identify and repair common data quality issues in event logs without requiring domain-specific training or fine-tuning.

### The Challenge

Process mining practitioners frequently encounter six types of event log imperfections that distort process discovery, conformance checking, and performance analysis. Manual identification and repair of these issues is time-consuming, error-prone, and requires deep domain expertise. This project explores an automated approach using LLM-generated Python code to detect and fix these quality problems.

### Our Approach

We employ a **prompt-based code generation methodology** where LLMs receive detailed task specifications and generate executable Python code that:

1. **Detects** problematic events using pattern-specific algorithms
2. **Validates** detection accuracy against ground truth labels (when available)
   - Computes precision, recall, and F1-score at this stage
3. **Repairs** the identified issues through appropriate transformations
4. **Outputs** the repaired dataset

The key innovation is that LLMs generate complete data repair pipelines on-demand, adapting to different event log structures without pre-training on process mining data.

---

## Imperfection Patterns and Repair Algorithms

### 1. **Form-Based Capture Errors**

**Problem:** Multiple events recorded at the exact same timestamp representing a single form submission that was later "flattened" into separate events during data entry.

**Example:**

```
Before:
Case=1, Time=10:00:00, Activity=enter_blood_pressure
Case=1, Time=10:00:00, Activity=enter_temperature
Case=1, Time=10:00:00, Activity=enter_weight

After (Merged):
Case=1, Time=10:00:00, Activity=enter_blood_pressure;enter_temperature;enter_weight
```

**Detection Algorithm:**

- Group events by `(Case, Timestamp)`
- Flag groups with ≥2 events as flattened
- Mark with `is_flattened = 1`

**Repair Method:** **MERGING**

- Combine all simultaneous activities alphabetically with `;` separator
- Reduce multiple rows to single merged event
- Preserve timestamp and case information

**Validation:**

- Ground truth: Events with non-empty `label` column
- Metrics calculated before merging
- Output: Fewer total rows (merged events)

---

### 2. **Distorted Labels**

**Problem:** Activity names with typos, character swaps, or case differences that represent the same action.

**Example:**

```
Before:
- Perform cehcks (typo - letters swapped)
- perform checks (correct)
- Checkf or completeness (space error)
- check for completeness (correct)

After (Normalized):
- perform checks (canonical)
- perform checks (canonical)
- check for completeness (canonical)
- check for completeness (canonical)
```

**Detection Algorithm:**

1. Preprocess: Lowercase, remove punctuation, normalize whitespace
2. Calculate **Jaccard N-gram Similarity** between all activity pairs
   - Generate character n-grams (default size=3)
   - Compute Jaccard coefficient: `|A ∩ B| / |A ∪ B|`
   - Threshold: 0.56 (pairs above threshold are similar)
3. Cluster using **Union-Find** (transitive closure of similarities)
4. Select canonical form by **majority voting** (most frequent variant)
5. Mark non-canonical variants as `is_distorted = 1`

**Repair Method:** **REPLACEMENT**

- Replace all distorted variants with canonical form
- Preserve original activities that match canonical
- Same number of rows, standardized activity names

**Validation:**

- Ground truth: Labels containing "distorted" (case-insensitive)
- Detects types: Interchange, Insert, Proximity, UpLow, Skip errors
- Metrics calculated before replacement

**Key Parameters:**

- `ngram_size = 3`: Character n-gram size for similarity
- `similarity_threshold = 0.56`: Jaccard threshold for clustering
- `min_length = 4`: Minimum activity length to compare

---

### 3. **Homonymous Labels**

**Problem:** Different activities incorrectly sharing the same label, typically with numeric suffixes or random IDs appended.

**Example:**

```
Before:
- Submit_Form_123:homonymous
- Submit_Form_456:homonymous
- Submit_Form_abc:homonymous

After (Normalized):
- submit form
- submit form
- submit form
```

**Detection Algorithm:**

1. Identify activities ending with `:homonymous` suffix
2. Remove suffix and normalize (lowercase, remove punctuation)
3. **TF-IDF Vectorization** with character n-grams (1-3 grams)
4. **Agglomerative Clustering** with cosine distance
   - Critical: Use `metric='cosine'` (not deprecated `affinity`)
   - `distance_threshold = 1 - similarity_threshold`
   - `linkage = 'average'`
5. Select canonical form by majority voting within clusters
6. Mark non-canonical activities as needing correction

**Repair Method:** **REPLACEMENT**

- Replace all homonymous variants with canonical cluster representative
- Clean activities (without `:homonymous` suffix) remain unchanged
- Reduces unique activity count

**Validation:**

- Ground truth: Events with non-empty `label` column
- Ensures clean activities are never modified
- Metrics calculated before replacement

**Key Parameters:**

- `similarity_threshold = 0.5`: Cosine similarity for clustering
- `linkage_method = 'average'`: Hierarchical clustering linkage

---

### 4. **Synonymous Labels**

**Problem:** Different textual variations representing the same logical activity (e.g., "Submit Form" vs "Form Submission" vs "submitting the form").

**Example:**

```
Before:
- Submit Form (most frequent)
- submit_form
- submitting the form
- Form Submission

After (Standardized):
- Submit Form (canonical)
- Submit Form (canonical)
- Submit Form (canonical)
- Submit Form (canonical)
```

**Detection Algorithm:**

1. Normalize activities: lowercase, remove punctuation (except spaces), collapse whitespace
2. **TF-IDF Vectorization** with character n-grams
   - Analyzer: `char_wb` (character with word boundaries)
   - Range: 1-3 grams
3. Compute **Cosine Similarity Matrix** for all unique activities
4. Cluster using **Union-Find** with similarity threshold
5. Keep clusters with ≥ `min_synonym_group_size` members
6. Select canonical by **frequency** (most common variant in dataset)
7. Mark non-canonical members as `is_synonymous_event = 1`

**Repair Method:** **REPLACEMENT**

- Map all synonym variants to their cluster's canonical form
- Unclustered activities remain unchanged
- Preserves row count, reduces unique activity labels

**Validation:**

- Ground truth: Events with non-empty `label` column
- Metrics calculated before standardization
- Tracks activity reduction count and percentage

**Key Parameters:**

- `similarity_threshold = 0.40`: Cosine similarity for synonym detection
- `min_synonym_group_size = 2`: Minimum cluster size
- `ngram_range = (1, 3)`: Character n-gram range

---

### 5. **Collateral Events**

**Problem:** Redundant events occurring within seconds of each other representing the same action (e.g., multiple "Submit" button clicks).

**Example:**

```
Before:
- 10:00:00 Submit_Form
- 10:00:01 Submit_Form_1:collateral
- 10:00:01 Submit_Form_signed:collateral
- 10:00:05 Review_Document

After (Deduplicated):
- 10:00:00 Submit_Form (KEPT)
- 10:00:05 Review_Document (KEPT)
```

**Detection Algorithm:**

1. Preprocess: Normalize activity names, remove suffixes (`_1`, `_signed`)
2. **Sliding Window Clustering** per case:
   - Start window at event `i`, track base activity and start time
   - Expand window: add events within `time_threshold` (2.0 seconds)
   - Allow `max_mismatches` (1 different activity) for noise tolerance
   - Stop if time gap exceeds threshold or mismatches exceed limit
3. Filter cluster to **dominant base activity** (most frequent)
4. Validate cluster size ≥ `min_matching_events` (2)
5. **Mark for removal:**
   - Keep unsuffixed version if exists, else keep first event
   - Mark others with `is_collateral_event = 1`
6. Assign `CollateralGroup` ID to all events in cluster

**Repair Method:** **DELETION**

- Physically remove events where `is_collateral_event == 1`
- Keep only one representative per cluster
- Output has fewer rows (duplicates removed)

**Validation:**

- Ground truth: Labels containing "collateral" (case-insensitive)
- Metrics calculated before deletion
- Ensures clean events outside clusters are never removed

**Key Parameters:**

- `time_threshold = 2.0`: Maximum seconds between cluster events
- `max_mismatches = 1`: Tolerance for different activities in window
- `min_matching_events = 2`: Minimum cluster size for valid collateral group

---

### 6. **Polluted Labels**

**Problem:** Activity names contaminated with noisy tokens like IDs, timestamps, or random codes that obscure the semantic meaning.

**Example:**

```
Before:
- Submit_Form_123
- Submit_Form_456
- Submit_Form_abc

After (Cleaned):
- submit form
- submit form
- submit form
```

**Detection Algorithm:**

1. **Aggressive Normalization** in 7 steps:
   - Lowercase conversion
   - Replace punctuation (`_ - . , ; :`) with spaces
   - Remove alphanumeric ID-like tokens (containing digits)
   - Remove long digit strings (5+ consecutive digits)
   - Collapse multiple whitespace to single space
   - Token limiting (keep first N tokens only)
   - Join remaining tokens
2. Group by normalized `BaseActivity`
3. Count unique original variants per base
4. Mark as **polluted** if `unique_variants > min_variants` (default: 2)
5. Flag with `is_polluted_label = 1`

**Repair Method:** **REPLACEMENT**

- Replace polluted activities with their `BaseActivity` (normalized form)
- Clean activities (single variant) remain unchanged
- Reduces unique activity count significantly

**Validation:**

- Ground truth: Events with non-empty `label` column
- Metrics calculated before replacement
- Tracks pollution rate and activity reduction percentage

**Key Parameters:**

- `aggressive_token_limit = 3`: Keep first N tokens after normalization
- `min_variants = 2`: Minimum variants to mark base as polluted

---

## Evaluation Methodology

### Two-Phase Approach

Each imperfection pattern is evaluated in isolation following this strict sequence:

**Phase 1: Detection and Validation (BEFORE Repair)**

1. **Detection**: Algorithm identifies problematic events using pattern-specific methods

   - Creates prediction column (e.g., `is_collateral_event`, `is_distorted`)
   - Marks events suspected to contain the target imperfection

2. **Validation**: Compare algorithm predictions against ground truth labels
   - `y_pred` = Detection algorithm's predictions (from step 1)
   - `y_true` = Ground truth labels from `label` column
   - Calculate metrics **BEFORE any repairs are applied**:
     - **Precision**: Of flagged events, how many were truly problematic?
     - **Recall**: Of all problematic events, how many were detected?
     - **F1-Score**: Harmonic mean of precision and recall
   - Threshold: Precision ≥ 0.6 required for acceptable detection

**Phase 2: Repair Application (AFTER Validation)**

3. **Repair**: Apply transformation based on detection results

   - Merge, delete, or replace events as appropriate for the pattern
   - No further metrics calculated at this stage

4. **Output**: Save repaired event log for downstream use
   - Measure activity reduction and transformation statistics
   - Original detection metrics reflect the algorithm's ability to identify issues

### Critical Note on Metrics

**The precision, recall, and F1-score measure DETECTION accuracy, not repair quality.**

- **What they measure**: "Did the algorithm correctly identify which events are problematic?"
- **What they DON'T measure**: "Did the algorithm fix the data correctly?"

This is because:

- Ground truth labels only indicate _which events_ have issues (e.g., `:distorted`, `:collateral`)
- Ground truth does NOT provide the correct repaired values
- We validate the detection step, then trust that correct detection leads to correct repairs

**Example Flow (Distorted Labels):**

```
Input: "perform checks", "perform cehcks:distorted", "perform checks"

Step 1: Detection
- Algorithm clusters similar activities
- Marks "perform cehcks" as is_distorted=1

Step 2: Validation (BEFORE fixing)
- y_true = [0, 1, 0] (from :distorted labels)
- y_pred = [0, 1, 0] (from is_distorted column)
- Precision = 1.0, Recall = 1.0, F1 = 1.0
- Metrics recorded at this stage

Step 3: Repair (AFTER validation)
- Replace "perform cehcks" → "perform checks"

Output: "perform checks", "perform checks", "perform checks"
```

### Ground Truth Labeling

Datasets include a `label` column indicating true imperfections:

- **Synthetic data**: Labels injected via FLAWD framework (e.g., `:distorted`, `:collateral`)
- **Real-world data**: Labels may be absent (metrics default to 0.0)

### Metrics Interpretation

| Metric        | Meaning                                   | Good Range |
| ------------- | ----------------------------------------- | ---------- |
| **Precision** | Correctness of detections                 | ≥ 0.6      |
| **Recall**    | Completeness of detections                | ≥ 0.5      |
| **F1-Score**  | Overall detection quality (harmonic mean) | ≥ 0.6      |

**High Precision, Low Recall:** Conservative (misses some issues but few false alarms)  
**Low Precision, High Recall:** Aggressive (catches most issues but many false positives)  
**High F1:** Balanced detection (ideal)

## Installation

### Prerequisites

- Python 3.8 or higher
- API keys for:
  - OpenAI (GPT-4)
  - Hugging Face (Llama, Deepseek)
  - xAI (Grok)

### Setup

1. Clone the repository:

```bash
git clone https://github.com/dariganarmanova/llm-event-log-quality.git
cd llm-event-log-quality
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up API keys as environment variables:

```bash
export OPENAI_API_KEY='your-openai-key'
export HUGGINGFACE_API_KEY='your-hf-token'
export XAI_API_KEY='your-xai-key'
```

---

## Repository Structure

```
.
├── data/                          # Event log datasets
│   ├── credit/                    # Synthetic Credit dataset
│   │   ├── Credit-FormBased.csv
│   │   ├── Credit-Distorted.csv
│   │   ├── Credit-Homonymous.csv
│   │   ├── Credit-Synonymous.csv
│   │   ├── Credit-Collateral.csv
│   │   └── Credit-Polluted.csv
│   ├── pub/                       # Synthetic Pub dataset
│   ├── bpic11/                    # BPIC 2011 real-world dataset
│   └── bpic15/                    # BPIC 2015 real-world dataset
├── prompt/                        # Task-specific prompts for each pattern
│   ├── form_based_prompt.txt      # Form flattening detection/merge
│   ├── distorted_prompt.txt       # Typo detection/normalization
│   ├── homonymous_prompt.txt      # Homonym detection/clustering
│   ├── synonymous_prompt.txt      # Synonym detection/unification
│   ├── collateral_prompt.txt      # Duplicate detection/removal
│   └── polluted_prompt.txt        # Noise detection/cleaning
├── results/                       # Experimental results (JSON format)
│   ├── credit/                    # Results for Credit dataset
│   ├── pub/                       # Results for Pub dataset
│   ├── bpic11/                    # Results for BPIC11 dataset
│   └── bpic15/                    # Results for BPIC15 dataset
├── script/                        # Generated Python scripts (temporary)
├── gpt_access.py                  # GPT-4 runner
├── llama_access.py                # Llama runner
├── deepseek_access.py             # DeepSeek runner
├── grok_access.py                 # Grok runner
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## Usage

### Running Experiments

Each Python script follows the same interface:

```bash
python <model>_access.py <dataset_path> <prompt_path>
```

The script will:

1. Load the event log CSV
2. Send the prompt + dataset info to the LLM API
3. Receive generated Python code
4. Execute the code in a subprocess
5. Capture detection metrics and repair results
6. Save output to `results/<dataset>/` as JSON

### Examples

#### Example 1: GPT-4 on Credit Dataset - Form-Based Pattern

```bash
python gpt_access.py data/credit/Credit-FormBased.csv prompt/form_based_prompt.txt
```

**Output:** `results/credit/Credit-FormBased_GPT_run1.json`

#### Example 2: Grok on BPIC11 - Synonymous Pattern

```bash
python grok_access.py data/bpic11/BPIC11-Synonymous.csv prompt/synonymous_prompt.txt
```

**Output:** `results/bpic11/BPIC11-Synonymous_Grok_run1.json`

#### Example 3: Running All Patterns on Credit Dataset with DeepSeek

```bash
# Form-based
python deepseek_access.py data/credit/Credit-FormBased.csv prompt/form_based_prompt.txt

# Distorted
python deepseek_access.py data/credit/Credit-Distorted.csv prompt/distorted_prompt.txt

# Homonymous
python deepseek_access.py data/credit/Credit-Homonymous.csv prompt/homonymous_prompt.txt

# Synonymous
python deepseek_access.py data/credit/Credit-Synonymous.csv prompt/synonymous_prompt.txt

# Collateral
python deepseek_access.py data/credit/Credit-Collateral.csv prompt/collateral_prompt.txt

# Polluted
python deepseek_access.py data/credit/Credit-Polluted.csv prompt/polluted_prompt.txt
```

### Running Multiple Iterations

For statistical reliability, run each experiment 3 times:

```bash
for i in {1..3}; do
  python gpt_access.py data/credit/Credit-Distorted.csv prompt/distorted_prompt.txt
done
```

Results are saved as `run1.json`, `run2.json`, `run3.json`.

---

## Understanding Results

### JSON Output Structure

```json
[
  {
    "run_id": 1,
    "prompt_id": 1,
    "model": "gpt-4",
    "temperature": 0.3,
    "dataset": "credit",
    "task_type": "distorted",
    "prompt": "Full prompt text...",
    "response": "Generated Python code...",
    "success": true,
    "output_shape": [5000, 6],
    "metrics": {
      "precision": 0.9567,
      "recall": 0.9234,
      "f1_score": 0.9398
    },
    "error": null,
    "timestamp": "2025-11-19T14:30:03.235087"
  }
]
```

**Key Fields:**

- `success`: Whether generated code executed without errors
- `metrics`: Detection performance (before repair)
  - `precision`: Accuracy of flagged events
  - `recall`: Completeness of detection
  - `f1_score`: Overall quality score
- `error`: Error message if execution failed (e.g., syntax errors, import errors)
- `output_shape`: Dimensions of repaired event log `[rows, columns]`

### Analyzing Results

#### Single Run Analysis

```python
import json
import pandas as pd

# Load result
with open('results/credit/Credit-Distorted_GPT_run1.json', 'r') as f:
    result = json.load(f)[0]

if result['success']:
    print(f"Precision: {result['metrics']['precision']:.4f}")
    print(f"Recall: {result['metrics']['recall']:.4f}")
    print(f"F1-Score: {result['metrics']['f1_score']:.4f}")
else:
    print(f"Error: {result['error']}")
```

#### Aggregate Multiple Runs

```python
import json
import numpy as np

# Load all runs
runs = []
for i in range(1, 4):
    with open(f'results/credit/Credit-Distorted_GPT_run{i}.json', 'r') as f:
        runs.extend(json.load(f))

# Filter successful runs
successful = [r for r in runs if r['success']]

if successful:
    avg_precision = np.mean([r['metrics']['precision'] for r in successful])
    avg_recall = np.mean([r['metrics']['recall'] for r in successful])
    avg_f1 = np.mean([r['metrics']['f1_score'] for r in successful])

    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1-Score: {avg_f1:.4f}")
    print(f"Success Rate: {len(successful)}/{len(runs)}")
```

---

## Key Findings

### Overall Performance

See our paper for detailed analysis and discussion.

---

## Datasets

### Synthetic Datasets

**Credit & Pub:** Event logs with artificially injected imperfections using the FLAWD (Flaw Language for Workflow Data) framework.

- **Injection rates**: 30% error rates per pattern
- **Characteristics**: Controlled scenarios, clear pattern boundaries
- **Size**:
  - Credit: 5,000 cases, 12 activities
  - Pub: 5,000 cases, 12 activities

### Real-World Datasets

**BPIC 2011:** Hospital billing process

- **Source**: Dutch academic hospital (Translated)
- **Size**: 1,143 cases, 624 activity

**BPIC 2015:** Municipality building permit process

- **Source**: Dutch municipality
- **Size**: 1,199 cases, 289 activity

---

## License

[MIT License / Apache 2.0 / GPL-3.0 - specify your choice]

---

## Contact

For questions, suggestions, or collaborations:

- **Email**: mcomuzzi@unist.ac.kr
- **GitHub Issues**: [Open an issue](https://github.com/dariganarmanova/llm-event-log-quality/issues)

---

## Acknowledgments

- **Event log datasets**: [4TU.ResearchData](https://data.4tu.nl/), BPI Challenge organizers
- **FLAWD framework**: Error injection methodology for synthetic data
- **LLM API providers**: OpenAI, HuggingFace, xAI
- **Process mining community**: For identifying and categorizing event log quality issues

---

## Future Work

- Extend to additional imperfection patterns (e.g., missing events, incorrect case assignments)
- Develop Agentic AI with an interface
- Develop ensemble approaches combining multiple LLMs
- Investigate fine-tuning strategies for process mining-specific tasks
- Create automated repair pipelines with confidence thresholds
- Explore explainability of LLM-generated repair code
