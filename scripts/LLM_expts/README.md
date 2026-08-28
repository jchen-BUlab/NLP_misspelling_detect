# JMI Code

This folder contains scripts used for GPT API experiments and related analyses.

Run the commands below from the `code_release/scripts/` folder. Relative input and output paths are resolved from that folder.

## Python Scripts

### `gpt_api_expts.py`

This script runs OpenAI chat-completion experiments for medication-name spelling classification. It reads a CSV dataset with medication terms, sends the terms to a selected GPT model under one or more prompt/temperature/top-p settings, and writes prediction files plus a run-summary CSV.

The task label convention is:

- `0`: correctly spelled medication name
- `1`: misspelled, nonstandard, noisy, non-medication, or uncertain term

#### Required Environment Variables

Before running the script, set the OpenAI API key:

```bash
export OPENAI_API_KEY="your_api_key"
```

The script also reads these optional environment variables:

```bash
export OPENAI_ORG_ID="your_org_id"
export OPENAI_PROJECT_ID="your_project_id"
```

#### Input Arguments

- `--model`: OpenAI model name. Default: `gpt-4o`.
- `--input_folder`: folder containing the input dataset CSV files. Default: `analysis_gpt_api_expts`.
- `--test_filename`: filename used when `--mode testing`. Default: `gpt_api_test_set.csv`.
- `--tuning_filename`: filename used when `--mode tuning`. Default: `gpt_api_tuning_set.csv`.
- `--mode`: which dataset to run, either `tuning` or `testing`. Default: `tuning`.
- `--batch_size`: number of terms sent per API request. Default: `1`.
- `--max_retries`: number of retries for a failed API request. Default: `3`.
- `--output_folder`: folder for output files. Default: `analysis_gpt_api_results`.
- `--prompt_styles`: comma-separated prompt templates to run. Default: `strict,few_shot`.
- `--temperatures`: comma-separated temperature values. Default: `0.0,0.2,0.7`.
- `--top_ps`: comma-separated top-p values. Default: `1.0,0.9`.
- `--disable_logprobs`: if included, do not request token log probabilities from the API.

Supported prompt styles in this script are:

```text
baseline0, baseline, strict, few_shot
```

#### Command-Line Examples

Run testing with one prompt and deterministic decoding:

```bash
python LLM_expts/gpt_api_expts.py \
  --mode testing \
  --model gpt-4o \
  --input_folder analysis_gpt_api_expts \
  --output_folder analysis_gpt_api_results \
  --prompt_styles baseline \
  --temperatures 0.0 \
  --top_ps 1.0 \
  --batch_size 1
```

Run a tuning sweep over two prompt styles and multiple decoding settings:

```bash
python LLM_expts/gpt_api_expts.py \
  --mode tuning \
  --model gpt-4o \
  --input_folder analysis_gpt_api_expts \
  --output_folder analysis_gpt_api_results \
  --prompt_styles strict,few_shot \
  --temperatures 0.0,0.2,0.7 \
  --top_ps 1.0,0.9 \
  --batch_size 1
```

#### Input File Format

The input file must be a CSV file with at least these columns:

```csv
input,target
metformin,0
amoxcillin,1
headache,1
```

Additional columns are allowed and are preserved in the output file.

By default, the script expects one of these files inside `--input_folder`:

- `gpt_api_tuning_set.csv` when `--mode tuning`
- `gpt_api_test_set.csv` when `--mode testing`

#### Output Files

For each combination of model, prompt style, temperature, and top-p, the script writes one CSV output file.

Example output path for testing:

```text
analysis_gpt_api_results/testing/gptneg4o_baseline_0p0_1p0.csv
```

Example output path for tuning:

```text
analysis_gpt_api_results/tuning/gptneg4o/gptneg4o_strict_0p0_1p0.csv
```

The output file keeps the original input columns and appends model output columns. Example:

```csv
input,target,gptneg4o_baseline_0p0_1p0_predict,gptneg4o_baseline_0p0_1p0_prob,process_time,process_speed,running_env
metformin,0,0,0.03,12.4,6.2,server_name
amoxcillin,1,1,0.98,12.4,6.2,server_name
```

The prediction column stores the parsed binary GPT output. The probability column stores the estimated probability of label `1` using API token log probabilities when available. If log probabilities are disabled or cannot be parsed, the script falls back to the hard prediction value.

The script also writes or updates:

```text
openai_prompt_sweep_runs.csv
```

This run-summary file records the experiment name, mode, dataset path, model, prompt style, decoding settings, output columns, processing time, running environment, and output path.

### `evaluate_gpt_api_stability.py`

This script evaluates prediction stability across repeated GPT API runs. It compares output files with the same filename across multiple trial folders, calculates per-term agreement across trials, and summarizes model performance variability across runs.

#### Input Arguments

- `--config`: required JSON configuration file.

The config file must contain:

- `model`: model label used in the output summary.
- `trial_folders`: list of folders containing repeated GPT output CSV files.

The config file may also contain:

- `output_folder`: folder where summary files are written. If omitted, the first trial folder is used.
- `source_filter`: source subset to evaluate. Default: `RxNorm`.

#### Example Config

```json
{
  "model": "gpt-4o",
  "trial_folders": [
    "analysis_gpt_api_results/trial1/testing",
    "analysis_gpt_api_results/trial2/testing",
    "analysis_gpt_api_results/trial3/testing"
  ],
  "output_folder": "analysis_gpt_api_results/stability",
  "source_filter": "RxNorm"
}
```

#### Command-Line Example

```bash
python LLM_expts/evaluate_gpt_api_stability.py \
  --config LLM_expts/stability_config.json
```

#### Input File Format

Each trial folder should contain GPT output CSV files. Files are matched by filename across trial folders, and only filenames found in at least two trial folders are analyzed.

Example input files:

```text
analysis_gpt_api_results/trial1/testing/gptneg4o_baseline_0p0_1p0.csv
analysis_gpt_api_results/trial2/testing/gptneg4o_baseline_0p0_1p0.csv
analysis_gpt_api_results/trial3/testing/gptneg4o_baseline_0p0_1p0.csv
```

Each CSV should contain an `input` column, a `target` column, and one prediction column. A probability column is optional but used when available for Brier score, ROC AUC, and PR AUC.

Example CSV:

```csv
input,target,source,gptneg4o_baseline_0p0_1p0_predict,gptneg4o_baseline_0p0_1p0_prob
metformin,0,RxNorm,0,0.03
amoxcillin,1,LTC,1,0.98
```

#### Output Files

The script writes four output files to `output_folder`. The filenames include the sanitized model name and source filter.

Example outputs for model `gpt-4o` and source filter `RxNorm`:

```text
gptneg4o_stability_summary_RxNorm.csv
gptneg4o_stability_detail_RxNorm.csv
gptneg4o_stability_summary_RxNorm.tsv
gptneg4o_stability_summary_RxNorm.docx
```

The summary CSV contains one row per repeated experiment setting and reports:

- number of trials
- number of terms
- overall consistency score
- mean and standard deviation of precision, recall, F1, accuracy, specificity, Brier score, ROC AUC, and PR AUC

The detail CSV contains one row per term per experiment setting and includes trial-level predictions, the number of trials used, counts and percentages of label `0` and label `1`, and the term-level consistency score.

The TSV and DOCX files are simplified summary reports for easier reading. The DOCX report uses a landscape table and reports each metric as `mean +- sd`.
