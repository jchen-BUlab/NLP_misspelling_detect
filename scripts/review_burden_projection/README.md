# JMI Code

This folder contains scripts used for operational projection analyses and related JMI outputs.

Run the command below from the `code_release/scripts/` folder.

## Python Scripts

### `simulate_operational_implications.py`

This script projects expected model outputs per a fixed number of medication entries under assumed misspelling prevalences. It uses each model's observed recall and specificity to estimate true positives, false positives, false negatives, true negatives, review burden, missed errors, and projected precision.

#### Input Arguments

The script takes only one command-line argument:

- `--config`: required JSON configuration file containing model metrics and output settings.

Run the script as:

```bash
python review_burden_projection/simulate_operational_implications.py \
  --config review_burden_projection/simulate_operational_implications_config.json
```

#### Configuration File

The configuration file must contain a `metrics` list. Each row in `metrics` should provide a model name, recall, and specificity.

Example:

```json
{
  "metrics": [
    {
      "model": "BERTDrug",
      "recall": 0.718,
      "specificity": 0.854
    },
    {
      "model": "CharBERTDrug",
      "recall": 0.780,
      "specificity": 0.842
    }
  ],
  "output_file": "./ltc/operational_projection.tsv",
  "entries_per_batch": 1000,
  "prevalences": [0.005, 0.01, 0.02, 0.05],
  "model_column": "model",
  "recall_column": "recall",
  "specificity_column": "specificity",
  "round_digits": 2,
  "make_figures": true,
  "no_figures": false,
  "figure_folder": null,
  "figure_mode": "per_metric",
  "figure_models": [
    "BERTDrug",
    "CharBERTDrug"
  ]
}
```

#### Configuration Fields

- `metrics`: required list of model performance rows.
- `output_file`: output TSV path. If omitted, the script writes `operational_projection.tsv`.
- `entries_per_batch`: number of medication entries to project. Default: `1000`.
- `prevalences`: assumed misspelling prevalences. Values can be proportions such as `0.005` or percentages such as `0.5`.
- `model_column`: column name in each metric row that stores the model name. Default: `model`.
- `recall_column`: column name in each metric row that stores recall. Default: `recall`.
- `specificity_column`: column name in each metric row that stores specificity. Default: `specificity`.
- `round_digits`: digits after the decimal point for projected counts. Default: `2`.
- `make_figures`: whether to create figure files. Default: `true`.
- `no_figures`: if `true`, disables figure output even when `make_figures` is true.
- `figure_folder`: folder for SVG and PNG figures. If null, the script creates a folder next to `output_file`.
- `figure_mode`: `per_model` or `per_metric`.
- `figure_models`: optional list of models to include in figures. If omitted or null, all models are included.

#### Input Metric Format

The model metrics are stored directly inside the JSON config file. The minimum required format is:

```json
{
  "metrics": [
    {"model": "model_a", "recall": 0.80, "specificity": 0.95},
    {"model": "model_b", "recall": 0.90, "specificity": 0.85}
  ]
}
```

Recall and specificity can be provided as proportions from `0` to `1` or as percentages from `0` to `100`.

#### Output TSV File

The main output is a tab-separated file with one row for each model and prevalence combination.

Example columns:

```text
model
entries
prevalence
prevalence_percent
observed_recall
observed_specificity
expected_true_errors
expected_non_errors
true_positives
false_positives
false_negatives
true_negatives
review_burden_tp_plus_fp
missed_errors_fn
projected_precision
```

Example output:

```text
model	entries	prevalence	prevalence_percent	observed_recall	observed_specificity	true_positives	false_positives	false_negatives	true_negatives	review_burden_tp_plus_fp	missed_errors_fn	projected_precision
BERTDrug	1000	0.005	0.5	0.718	0.854	3.59	145.27	1.41	849.73	148.86	1.41	0.024116
```

#### Figure Output

When figure output is enabled, the script writes SVG and PNG figures. The figure folder defaults to:

```text
<output_file_stem>_figures
```

The script supports two figure modes:

- `per_model`: one figure per model, showing both review burden and missed errors.
- `per_metric`: one figure per metric, showing all selected models in one figure.

The PNG output uses Matplotlib when it is available. If Matplotlib cannot be imported or initialized, the script prints a warning and falls back to a lower-quality built-in PNG renderer. SVG files are generated directly by the script.

Model names `fastTextML` and `BioWordVecML` are displayed with `ML` as a subscript in figures.

#### Output Interpretation

- `review_burden_tp_plus_fp`: expected number of terms flagged for review, equal to true positives plus false positives.
- `missed_errors_fn`: expected number of misspelled terms missed by the model, equal to false negatives.
- `projected_precision`: projected fraction of reviewed terms that are true misspellings.
