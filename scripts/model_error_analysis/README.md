# JMI Code

This folder contains scripts used for JMI model-error analyses and related appendix outputs.

Run the commands below from the `code_release/scripts/` folder so `results/fullset_filtered.csv` resolves to the included result file.

## Python Scripts

### `analyze_model_errors.py`

This script analyzes term-level prediction errors from a single model-output CSV file. It adds per-model correctness and difficulty flags, identifies terms that are easy or hard for all discovered models, and writes summary tables and a human-readable report.

#### Input Arguments

- `--input_file`: input CSV file. Default: `results/fullset_filtered.csv`.
- `--output_file`: output CSV file with added flags. Default: `results/error_analysis_results/fullset_filtered_model_errors_cleaned.csv`.
- `--output_prefix`: prefix for summary output files. If omitted, the script uses `--output_file` without its file suffix.
- `--subset`: rows to analyze. Choices are `cleaned` and `all`. Default: `cleaned`.

#### Command-Line Examples

Run the default cleaned-term analysis using prediction correctness:

```bash
python model_error_analysis/analyze_model_errors.py \
  --input_file results/fullset_filtered.csv \
  --output_file results/error_analysis_results/fullset_filtered_model_errors_cleaned.csv
```

Analyze all terms instead of only cleaned categories:

```bash
python model_error_analysis/analyze_model_errors.py \
  --input_file results/fullset_filtered.csv \
  --output_file results/error_analysis_results/fullset_filtered_model_errors_all.csv \
  --subset all
```

#### Input File Format

The input file must be a CSV file with at least these columns:

```csv
medication_name,target
metformin,0
amoxcillin,1
```

The file must also contain at least one model prediction column whose name ends with:

```text
__pred_default
```

Example:

```csv
medication_name,target,category,final_type,BERTDrug__pred_default,CharBERTDrug__pred_default,SpellChecker__pred_default
metformin,0,correct drug name,BN,0,0,0
amoxcillin,1,misspelled drug name,BN,1,1,1
EC Aspirin,0,correct short drug name,IN,1,0,0
```

Optional but commonly used columns include:

- `category`: term category, such as `correct drug name`, `correct short drug name`, or `misspelled drug name`.
- `final_type` or `type_final`: term type, such as `BN`, `IN`, or `BN,IN`.

#### How Term Type Is Determined

The script does not look up term type from external files. It uses the type already present in the input CSV:

- If `final_type` exists, use `final_type`.
- Otherwise, use `type_final`.
- Empty values are reported as `EMPTY`.
- Multiple type labels are split on commas, semicolons, slashes, or pipes, uppercased, deduplicated, sorted, and joined with commas.

For example:

```text
IN,BN
BN; IN
BN/IN
```

all become:

```text
BN,IN
```

#### Subset Filtering

When `--subset cleaned`, the script keeps rows whose `category` is one of:

```text
correct drug name
correct short drug name
misspelled drug name
mispelled drug name
```

When `--subset all`, every row is analyzed.

#### Difficulty Definition

- A model marks a term as easy if the model prediction equals `target`.
- A model marks a term as hard if the model prediction differs from `target`.
- `easy_for_all` means all discovered models mark the term as easy.
- `hard_for_all` means all discovered models mark the term as hard.
- All remaining terms are labeled `other`.

#### Output Files

The main output CSV is written to `--output_file`. It preserves the input columns and adds columns such as:

```text
<model>__correct
<model>__easy
<model>__hard
<model>__difficulty
difficult_term
easy_term
difficulty_tag
final_type
```

The script also writes summary files using `--output_prefix`:

```text
<output_prefix>_model_error_flags.tsv
<output_prefix>_model_correctness_summary.tsv
<output_prefix>_model_difficulty_summary.tsv
<output_prefix>_difficulty_tag_summary.tsv
<output_prefix>_category_by_difficulty_tag_summary.tsv
<output_prefix>_category_type_by_difficulty_tag_summary.tsv
<output_prefix>_spellchecker_bert_contrast_category_summary.tsv
<output_prefix>_summary.txt
```

The human-readable summary text file reports:

- model correctness breakdown
- model difficulty breakdown
- easy/hard/other term counts
- category breakdown within each difficulty tag
- category and final-type breakdown within each difficulty tag
- SpellChecker vs. BERT/CharBERT contrast groups with example terms
