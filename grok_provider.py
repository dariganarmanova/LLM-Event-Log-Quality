#!/usr/bin/env python3
"""
Parameterized Grok CSV Processor with Clean Logging (Native xAI SDK)
Usage: python grok_processor.py <csv_file> <prompt_file> [options]
"""

import os
import subprocess
import pandas as pd
from pathlib import Path
import re
from datetime import datetime
import time
import argparse
import json

# Import xAI SDK
try:
    from xai_sdk import Client
    from xai_sdk.chat import user, system
except ImportError:
    print("[ERROR] xai_sdk not installed. Please run: pip install xai-sdk")
    exit(1)

class ParameterizedGrokProcessor:
    def __init__(self, api_key=None, model_name="grok-4-fast"):
        """Initialize the Parameterized Grok CSV Processor"""
        self.api_key = api_key or os.getenv('XAI_API_KEY')
        if not self.api_key:
            raise ValueError("xAI API token not found. Please set XAI_API_KEY environment variable.")
        
        self.model_name = model_name
        
        # Initialize the xAI client
        try:
            self.client = Client(
                api_key=self.api_key,
                timeout=3600  # Extended timeout for long-running operations
            )
            print(f"[INFO] xAI client initialized with model: {self.model_name}")
        except Exception as e:
            print(f"[ERROR] Failed to initialize xAI client: {e}")
            raise
        
        # Set up directories
        self.script_dir = Path('script')
        self.logs_dir = Path('logs')
        self.results_dir = Path('results')
        
        # Create directories if they don't exist
        self.script_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
    def extract_dataset_info(self, csv_path):
        """Extract dataset name and directory from CSV path"""
        csv_path = Path(csv_path)
        dataset_dir = csv_path.parent
        dataset_name = dataset_dir.name
        
        if not dataset_name or dataset_name == 'data':
            dataset_name = csv_path.stem.split('_')[0]
        
        return dataset_name, dataset_dir
    
    def extract_task_type(self, prompt_path):
        """Extract task type from prompt filename"""
        filename = Path(prompt_path).stem
        if filename.endswith('_prompt'):
            return filename[:-7]
        return filename
    
    def get_csv_basename(self, csv_path):
        """Get the CSV filename without extension"""
        return Path(csv_path).stem
    
    def create_output_path(self, csv_path, prompt_path, run_number, output_suffix):
        """Create dynamic output path based on dataset, task type, and run number"""
        dataset_name, dataset_dir = self.extract_dataset_info(csv_path)
        task_type = self.extract_task_type(prompt_path)
        
        output_filename = f"{dataset_name}_{task_type}{output_suffix}_run{run_number}.csv"
        output_path = dataset_dir / output_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        return output_path, dataset_name, task_type
    
    def log_clean(self, message, run_number=None):
        """Clean logging to console only"""
        if run_number:
            print(f"[INFO] Run {run_number}: {message}")
        else:
            print(f"[INFO] {message}")
    
    def log_error(self, message, run_number=None):
        """Error logging to console only"""
        if run_number:
            print(f"[ERROR] Run {run_number}: {message}")
        else:
            print(f"[ERROR] {message}")
        
    def load_prompt(self, prompt_path):
        """Load prompt from specified file"""
        prompt_path = Path(prompt_path)
        
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file '{prompt_path}' not found")
            
        with open(prompt_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        return content
    
    def create_context_prompt(self, csv_path, base_prompt, output_path, dataset_name, task_type, 
                            run_number, algorithm_params):
        """Create an enhanced prompt with configuration parameters"""
        param_config = "\n".join([f"- {key} = {value}" for key, value in algorithm_params.items()])
        
        context = f"""
CSV File Path: {csv_path}
Dataset Name: {dataset_name}
Task Type: {task_type}
Run Number: {run_number}
Required Output File: {output_path}

Algorithm Configuration Parameters:
{param_config}

{base_prompt}

IMPORTANT REQUIREMENTS:
1. Your code MUST save the final processed/cleaned dataset to: {output_path}
2. Use df.to_csv('{output_path}', index=False) to save the processed data
3. The output filename should include run number: {Path(output_path).name}
4. Use the provided algorithm parameters in your implementation
5. Print a summary of what was processed for run {run_number}
6. Print the shape of the final dataset
7. Print the path where the data was saved
8. This is run #{run_number} of the process

Configuration Usage Example:
```python
import pandas as pd

# Configuration parameters
time_threshold = {algorithm_params.get('time_threshold', 2.0)}
require_same_resource = {algorithm_params.get('require_same_resource', False)}
min_matching_events = {algorithm_params.get('min_matching_events', 2)}
max_mismatches = {algorithm_params.get('max_mismatches', 1)}
activity_suffix_pattern = r"{algorithm_params.get('activity_suffix_pattern', '(_signed\\d*|_\\d+)$')}"

# Load the data
df = pd.read_csv('{csv_path}')
print(f"Run {run_number}: Original dataset shape: {{df.shape}}")

# Your algorithm implementation here using the configuration parameters

# REQUIRED: Save the processed data
df.to_csv('{output_path}', index=False)

# REQUIRED: Print summary
print(f"Run {run_number}: Processed dataset saved to: {output_path}")
print(f"Run {run_number}: Final dataset shape: {{df.shape}}")
print(f"Run {run_number}: Dataset: {dataset_name}")
print(f"Run {run_number}: Task type: {task_type}")
```
"""
        return context
    
    def generate_code(self, prompt, max_tokens=2000, temperature=0.3, max_retries=5):
        """Generate code using xAI native SDK"""
        for attempt in range(max_retries):
            try:
                print(f"Attempt {attempt + 1}: Generating code with xAI SDK...")
            
            # Create a chat instance with temperature
                chat = self.client.chat.create(
                    model=self.model_name,
                    temperature=temperature  # Set temperature here, not in sample()
                )
            
            # Add system message using the system helper
                chat.append(system("You are an expert Python programmer specializing in data analysis and CSV manipulation. Generate clean, well-commented, and executable Python code. Always include proper imports and error handling."))
            
            # Add user message using the user helper
                chat.append(user(prompt))
            
            # Sample the response - no parameters needed
                response = chat.sample()
            
                return response.content
            
            except Exception as e:
                error_msg = str(e).lower()
                if "rate limit" in error_msg and attempt < max_retries - 1:
                    wait_time = min(120, 15 * (attempt + 1))
                    print(f"Rate limited, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                elif attempt < max_retries - 1:
                    print(f"Request failed, retrying in 15 seconds...")
                    time.sleep(15)
                    continue
                else:
                    self.log_error(f"Error generating code: {e}")
                    return None
    
        return None
    
    def extract_python_code(self, response_text):
        """Extract Python code from the response"""
        if not response_text:
            return None
            
        # Look for code blocks with python specification
        code_blocks = re.findall(r'```python\n(.*?)\n```', response_text, re.DOTALL)
        if code_blocks:
            return code_blocks[0]
        
        # Look for code blocks without language specification
        code_blocks = re.findall(r'```\n(.*?)\n```', response_text, re.DOTALL)
        if code_blocks:
            return code_blocks[0]
        
        # Look for code blocks with just ```
        code_blocks = re.findall(r'```(.*?)```', response_text, re.DOTALL)
        if code_blocks:
            code = code_blocks[0].strip()
            lines = code.split('\n')
            if lines and lines[0].strip() in ['python', 'py']:
                code = '\n'.join(lines[1:])
            return code
        
        return response_text
    
    def save_generated_script(self, code, csv_path, prompt_path, run_number):
        """Save the generated code to a file"""
        if not code:
            return None
            
        csv_basename = self.get_csv_basename(csv_path)
        script_filename = f"{csv_basename}_Grok_run{run_number}.py"
        script_path = self.script_dir / script_filename
        
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(f"# Generated script for {csv_basename} - Run {run_number}\n")
            f.write(f"# Generated on: {datetime.now().isoformat()}\n")
            f.write(f"# Model: {self.model_name}\n\n")
            f.write(code)
        
        return script_path
    
    def run_script(self, script_path, csv_path):
        """Run the generated script using subprocess"""
        try:
            cmd = ['python', str(script_path), str(csv_path)]
            
            print(f"[DEBUG] Running: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=os.getcwd()
            )
            
            # Print output for debugging
            if result.stdout:
                print(f"[DEBUG] Script STDOUT:\n{result.stdout}")
            if result.stderr:
                print(f"[DEBUG] Script STDERR:\n{result.stderr}")
            
            print(f"[DEBUG] Return code: {result.returncode}")
            
            return result.returncode == 0, result.stdout, result.stderr, result
            
        except subprocess.TimeoutExpired:
            print(f"[DEBUG] Script execution timed out")
            return False, "", "Script execution timed out", None
        except Exception as e:
            print(f"[DEBUG] Script execution error: {e}")
            return False, "", str(e), None
    
    def check_output_files(self, output_path):
        """Check if the expected output files were created"""
        print(f"[DEBUG] Checking for output file: {output_path}")
        
        if output_path.exists():
            try:
                df = pd.read_csv(output_path)
                print(f"[DEBUG] Output file found! Shape: {df.shape}")
                return True, output_path, df.shape
            except Exception as e:
                print(f"[DEBUG] Error reading output file: {e}")
                return False, output_path, None
        else:
            print(f"[DEBUG] Output file NOT found at: {output_path}")
            # Check if any files were created in the directory
            parent_dir = output_path.parent
            if parent_dir.exists():
                csv_files = list(parent_dir.glob("*.csv"))
                print(f"[DEBUG] CSV files in {parent_dir}: {[f.name for f in csv_files]}")
        
        return False, output_path, None
    
    def parse_metrics_from_stdout(self, stdout_text):
        """Parse metrics from script stdout with improved patterns."""
        metrics = {}
        
        if not stdout_text:
            return metrics
        
        # Try multiple patterns for each metric to be more robust
        
        # Precision patterns
        precision_patterns = [
            r'Precision:\s*([0-9.]+)',
            r'precision:\s*([0-9.]+)',
            r'PRECISION:\s*([0-9.]+)',
            r'Precision\s*=\s*([0-9.]+)',
        ]
        
        # Recall patterns
        recall_patterns = [
            r'Recall:\s*([0-9.]+)',
            r'recall:\s*([0-9.]+)',
            r'RECALL:\s*([0-9.]+)',
            r'Recall\s*=\s*([0-9.]+)',
        ]
        
        # F1-Score patterns
        f1_patterns = [
            r'F1-Score:\s*([0-9.]+)',
            r'F1:\s*([0-9.]+)',
            r'f1[-_]score:\s*([0-9.]+)',
            r'F1\s*=\s*([0-9.]+)',
        ]
        
        # Note patterns
        note_patterns = [
            r'Note:\s*(.+?)(?:\n|$)',
            r'NOTE:\s*(.+?)(?:\n|$)',
            r'note:\s*(.+?)(?:\n|$)',
        ]
        
        # Try each pattern
        for pattern in precision_patterns:
            match = re.search(pattern, stdout_text, re.IGNORECASE)
            if match:
                try:
                    metrics['precision'] = float(match.group(1))
                    print(f"[DEBUG] Parsed precision: {metrics['precision']}")
                    break
                except (ValueError, IndexError):
                    continue
        
        for pattern in recall_patterns:
            match = re.search(pattern, stdout_text, re.IGNORECASE)
            if match:
                try:
                    metrics['recall'] = float(match.group(1))
                    print(f"[DEBUG] Parsed recall: {metrics['recall']}")
                    break
                except (ValueError, IndexError):
                    continue
        
        for pattern in f1_patterns:
            match = re.search(pattern, stdout_text, re.IGNORECASE)
            if match:
                try:
                    metrics['f1_score'] = float(match.group(1))
                    print(f"[DEBUG] Parsed f1_score: {metrics['f1_score']}")
                    break
                except (ValueError, IndexError):
                    continue
        
        for pattern in note_patterns:
            match = re.search(pattern, stdout_text, re.IGNORECASE)
            if match:
                try:
                    metrics['note'] = match.group(1).strip()
                    print(f"[DEBUG] Parsed note: {metrics['note']}")
                    break
                except IndexError:
                    continue
        
        # Additional statistics parsing
        stats_patterns = {
            'total_clusters': r'Total clusters[^:]*:\s*(\d+)',
            'original_events': r'Original events[^:]*:\s*(\d+)',
            'fixed_events': r'Fixed events[^:]*:\s*(\d+)',
            'events_removed': r'Events removed[^:]*:\s*(\d+)',
        }
        
        for key, pattern in stats_patterns.items():
            match = re.search(pattern, stdout_text, re.IGNORECASE)
            if match:
                try:
                    metrics[key] = int(match.group(1))
                    print(f"[DEBUG] Parsed {key}: {metrics[key]}")
                except (ValueError, IndexError):
                    continue
        
        return metrics
    
    def save_result_json(self, run_number, prompt_id, dataset_name, task_type, prompt_content, 
                        response, output_shape, success, temperature, error_msg=None, stdout_text=None):
        """Save results in JSON format including metrics parsed from script stdout."""
        
        # Parse metrics from stdout
        metrics = self.parse_metrics_from_stdout(stdout_text)
        
        # Log what we found
        if metrics:
            print(f"[DEBUG] Metrics extracted: {json.dumps(metrics, indent=2)}")
        else:
            print(f"[DEBUG] No metrics found in stdout")
        
        model_short = self.model_name.split('/')[-1].replace('.', '_').replace('-', '_')
        result_file = self.results_dir / f"{model_short}_{task_type}_{dataset_name}.json"
        
        result_data = {
            "run_id": run_number,
            "prompt_id": prompt_id,
            "model": self.model_name,
            "temperature": temperature,
            "dataset": dataset_name,
            "task_type": task_type,
            "prompt": prompt_content[:200] + "..." if len(prompt_content) > 200 else prompt_content,
            "response": response[:200] + "..." if len(response) > 200 else response,
            "success": success,
            "output_shape": list(output_shape) if output_shape else None,
            "metrics": metrics if metrics else None,
            "error": error_msg,
            "timestamp": datetime.now().isoformat()
        }
        
        # Load existing results
        if result_file.exists():
            with open(result_file, 'r', encoding='utf-8') as f:
                try:
                    all_results = json.load(f)
                except json.JSONDecodeError:
                    all_results = []
        else:
            all_results = []
        
        # Append new result
        all_results.append(result_data)
        
        # Save updated results
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2)
        
        return result_file
    
    def process_csv_single_run(self, csv_file, prompt_file, run_number, algorithm_params, 
                             output_suffix, model_name=None, prompt_id=1, temperature=0.3):
        """Process a CSV file with a specified prompt for a single run"""
        if model_name:
            self.model_name = model_name
        
        # Validate inputs
        csv_path = Path(csv_file)
        prompt_path = Path(prompt_file)
        
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        
        # Extract dataset and task type information
        output_path, dataset_name, task_type = self.create_output_path(csv_path, prompt_path, 
                                                                     run_number, output_suffix)
        
        self.log_clean(f"model={self.model_name}, dataset={dataset_name}", run_number)
        
        error_msg = None
        stdout_text = None
        
        try:
            # Load the prompt
            prompt_content = self.load_prompt(prompt_path)
            
            # Create enhanced context prompt
            context_prompt = self.create_context_prompt(csv_path, prompt_content, output_path, 
                                                      dataset_name, task_type, run_number, 
                                                      algorithm_params)
            
            # Generate code
            response = self.generate_code(context_prompt, temperature=temperature)
            if response is None:
                error_msg = "Failed to generate code"
                raise Exception(error_msg)
            
            self.log_clean("Prompt executed successfully", run_number)
            
            # Extract Python code
            code = self.extract_python_code(response)
            if not code:
                error_msg = "Failed to extract code from response"
                raise Exception(error_msg)
            
            print(f"[DEBUG] Generated code preview:\n{code[:500]}...")
            
            # Save generated script
            script_path = self.save_generated_script(code, csv_path, prompt_path, run_number)
            if script_path is None:
                error_msg = "Failed to save generated script"
                raise Exception(error_msg)
            
            print(f"[DEBUG] Script saved to: {script_path}")
            
            # Run the script
            success, stdout, stderr, result = self.run_script(script_path, csv_path)
            stdout_text = stdout  # Capture stdout for metrics parsing
            
            if not success:
                error_msg = f"Script execution failed. STDERR: {stderr[:200]}"
                print(f"[DEBUG] {error_msg}")
            
            # Check if output files were created
            output_created, final_output_path, output_shape = self.check_output_files(output_path)
            
            if not output_created:
                if not error_msg:
                    error_msg = f"Output file not created at expected path: {output_path}"
            
            # Save results to JSON with stdout for metrics parsing
            result_file = self.save_result_json(
                run_number, prompt_id, dataset_name, task_type, 
                prompt_content, response, output_shape, 
                success and output_created, temperature, error_msg, stdout_text=stdout_text
            )
            
            self.log_clean(f"Saved results to {result_file}", run_number)
            
            if success and output_created:
                self.log_clean(f"Processing completed successfully! Output: {final_output_path}", run_number)
            else:
                self.log_error(f"Processing failed: {error_msg}", run_number)
            
            return success and output_created, script_path, final_output_path if output_created else None
            
        except Exception as e:
            if not error_msg:
                error_msg = str(e)
            self.log_error(f"Error during processing: {error_msg}", run_number)
            
            # Still save the result even if it failed
            try:
                result_file = self.save_result_json(
                    run_number, prompt_id, dataset_name, task_type, 
                    prompt_content if 'prompt_content' in locals() else "", 
                    response if 'response' in locals() else "", 
                    None, False, temperature, error_msg, stdout_text=stdout_text
                )
                self.log_clean(f"Saved error results to {result_file}", run_number)
            except:
                pass
            
            return False, None, None

    def process_csv_multiple_runs(self, csv_file, prompt_file, algorithm_params, output_suffix,
                                model_name=None, num_runs=3, prompt_id=1, temperature=0.3):
        """Process a CSV file with a specified prompt multiple times"""
        self.log_clean(f"Starting {num_runs} runs of processing")
        
        results = []
        successful_runs = 0
        
        for run_number in range(1, num_runs + 1):
            print(f"\n{'='*60}")
            self.log_clean(f"Starting run {run_number} of {num_runs}")
            print(f"{'='*60}")
            
            success, script_path, output_path = self.process_csv_single_run(
                csv_file, prompt_file, run_number, algorithm_params, output_suffix, 
                model_name, prompt_id, temperature
            )
            
            results.append({
                'run_number': run_number,
                'success': success,
                'script_path': script_path,
                'output_path': output_path
            })
            
            if success:
                successful_runs += 1
        
        # Summary
        print(f"\n{'='*60}")
        self.log_clean("PROCESSING SUMMARY")
        print(f"{'='*60}")
        self.log_clean(f"Total runs: {num_runs}")
        self.log_clean(f"Successful runs: {successful_runs}")
        self.log_clean(f"Failed runs: {num_runs - successful_runs}")
        
        return results, successful_runs == num_runs

def main():
    """Main function to run the processor with command line arguments"""
    parser = argparse.ArgumentParser(description="Parameterized Grok CSV Processor (Native xAI SDK)")
    
    # Required arguments
    parser.add_argument("csv_file", help="Path to the CSV file to process")
    parser.add_argument("prompt_file", help="Path to the prompt file (.txt)")
    
    # Optional arguments
    parser.add_argument("--runs", type=int, default=3, help="Number of runs to execute (default: 3)")
    parser.add_argument("--api-key", help="xAI API token")
    parser.add_argument("--output-suffix", default="_cleaned", help="Suffix for output filename")
    parser.add_argument("--prompt-id", type=int, default=1, help="Prompt ID for tracking")
    parser.add_argument("--temperature", type=float, default=0.3, help="Temperature for generation")
    parser.add_argument("--model", type=str, default="grok-4-fast", help="Grok model to use (e.g., grok-4-fast)")
    
    # Algorithm parameters
    parser.add_argument("--time-threshold", type=float, default=2.0)
    parser.add_argument("--require-same-resource", action="store_true", default=False)
    parser.add_argument("--min-matching-events", type=int, default=2)
    parser.add_argument("--max-mismatches", type=int, default=1)
    parser.add_argument("--activity-suffix-pattern", type=str, default=r"(_signed\d*|_\d+)$")
    parser.add_argument("--similarity-threshold", type=float, default=0.8)
    parser.add_argument("--case-sensitive", action="store_true", default=False)
    parser.add_argument("--use-fuzzy-matching", action="store_true", default=False)
    
    args = parser.parse_args()
    
    # Create algorithm parameters dictionary
    algorithm_params = {
        'time_threshold': args.time_threshold,
        'require_same_resource': args.require_same_resource,
        'min_matching_events': args.min_matching_events,
        'max_mismatches': args.max_mismatches,
        'activity_suffix_pattern': args.activity_suffix_pattern,
        'similarity_threshold': args.similarity_threshold,
        'case_sensitive': args.case_sensitive,
        'use_fuzzy_matching': args.use_fuzzy_matching,
    }
    
    model_name = args.model
    
    print("[INFO] Starting Parameterized Grok CSV Processing Pipeline")
    print(f"[INFO] CSV File: {args.csv_file}")
    print(f"[INFO] Prompt File: {args.prompt_file}")
    print(f"[INFO] Model: {model_name}")
    print(f"[INFO] Number of runs: {args.runs}")
    
    # Initialize processor
    try:
        processor = ParameterizedGrokProcessor(api_key=args.api_key, model_name=model_name)
        
    except ValueError as e:
        print(f"[ERROR] {e}")
        print("[INFO] Please set your xAI API token as an environment variable:")
        print("[INFO] export XAI_API_KEY='your-xai-token-here'")
        return
    except Exception as e:
        print(f"[ERROR] Initialization error: {e}")
        return
    
    # Process CSV file with prompt multiple times
    try:
        results, all_successful = processor.process_csv_multiple_runs(
            args.csv_file, args.prompt_file, algorithm_params, args.output_suffix, 
            model_name, args.runs, args.prompt_id, args.temperature
        )
        
        if all_successful:
            print(f"\n[INFO] All {args.runs} runs completed successfully!")
        else:
            print(f"\n[WARNING] Some runs failed. Check results/ directory for details.")
            
    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    if len(os.sys.argv) == 1:
        print("[INFO] Parameterized Grok CSV Processor (Native xAI SDK)")
        print("[INFO] Usage: python grok_processor.py <csv_file> <prompt_file> [options]")
    else:
        main()