# Drug-name misspelling detection

This is the public code release for the manuscript on BERT-based detection of misspelled medication names.

## Contents

```text
code_release/
|-- data/                        public LTCDC workbooks and source notes
|-- examples/                    
|-- scripts/data/                RxNorm and LTCDC preparation
|-- scripts/training/            model training and inference
|-- scripts/evaluation/          prediction assembly and point estimates
|-- scripts/analysis/            bootstrap, figures, and secondary analyses
|-- scripts/LLM_expts/           LLM experiments
|-- scripts/model_error_analysis/ error analysis
|-- scripts/review_burden_projection/ operational projection
|-- scripts/results/             manuscript result CSVs
|-- scripts/analyze_factors_affecting_model_performance.Rmd
|-- src/drug_spelling/           shared implementation
```

New workflow outputs should be written under `artifacts/`. The original result CSVs requested for release are preserved under `scripts/results/`.

## Installation

Use Python 3.10 or later:

```bash
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
```

CharacterBERT resources come from the [CharacterBERT repository](https://github.com/helboukkouri/character-bert), and BioWordVec comes from the [NCBI BioSentVec/BioWordVec repository](https://github.com/ncbi-nlp/BioSentVec).

The model commands below assume downloaded resources are placed under `code_release/pretrained-models/`. Replace those example paths if the resources are stored elsewhere.

## Data preparation

Run the main workflow commands from the root folder.

### RxNorm

The study used the RxNorm 03/04/2024 Full Update Release. Provide branded and generic/ingredient names as one-name-per-line files:

```bash
python scripts/data/prepare_rxnorm.py \
  --brand-file path/to/rxnorm_brand_names.txt \
  --generic-file path/to/rxnorm_generic_names.txt \
  --output-dir artifacts/data/rxnorm \
  --multipliers 1,2,4,6,8,10
```

This creates the balanced training files, development/test files, metadata, and preparation manifest. See [data/RXNORM_SOURCE.md](data/RXNORM_SOURCE.md) for source details.

### LTCDC

The distributed LTCDC data are the two reviewed source workbooks under `data/`. Build the evaluation set with:

```bash
python scripts/data/prepare_ltcdc.py \
  --input data/LTC_set1_merged_reviewed_with_category.xlsx \
          data/LTC_set2_merged_reviewed_with_category.xlsx \
  --known-name-file artifacts/data/rxnorm/train1.txt \
                    artifacts/data/rxnorm/train6.txt \
                    artifacts/data/rxnorm/dev.txt \
  --output-prefix artifacts/data/ltcdc/annotated_test
```

The script merges set 1 and set 2 in that order, removes duplicate normalized terms, maps the reviewed categories to binary labels, and derives manuscript OOV membership from the generated RxNorm files. It writes `annotated_test.txt`, aligned metadata, and a manifest under `artifacts/data/ltcdc/`.

Labels are `positive` for misspelled terms and `negative` for correct terms.

## Model training

Transformer examples:

```bash
python scripts/training/train_transformer.py --model-type bert --pretrained-model pretrained-models/medical_bert --data-dir artifacts/data/rxnorm --train-file train6.txt --output-dir artifacts/models/BERTDrug

python scripts/training/train_transformer.py --model-type characterbert --pretrained-model pretrained-models/medical_character_bert --tokenizer pretrained-models/bert-base-uncased --data-dir artifacts/data/rxnorm --train-file train4.txt --output-dir artifacts/models/CharBERTDrug
```

Baseline examples:

```bash
python scripts/training/train_fasttext.py --task classification --do_train --data_dir artifacts/data/rxnorm --setnum 1 --output_dir artifacts/models/fastTextML --fasttext_training_mode supervised --fasttext_dim 200

python scripts/training/train_biowordvec.py --task classification --do_train --do_lower_case --data_dir artifacts/data/rxnorm --setnum 1 --output_dir artifacts/models/BioWordVecML --biowordvec_path pretrained-models/BioWordVec_PubMed_MIMICIII_d200.vec.bin --biowordvec_format keyedvectors --biowordvec_file_type binary

python scripts/training/run_spellchecker.py --data-dir artifacts/data/rxnorm --output-dir artifacts/models/SpellChecker
```

## LTCDC prediction and evaluation

Apply the trained models to `artifacts/data/ltcdc/annotated_test.txt`:

```bash
python scripts/training/predict_transformer.py --run-dir artifacts/models/BERTDrug --input artifacts/data/ltcdc/annotated_test.txt --output-dir artifacts/predictions/ltcdc/BERTDrug
python scripts/training/predict_transformer.py --run-dir artifacts/models/CharBERTDrug --input artifacts/data/ltcdc/annotated_test.txt --output-dir artifacts/predictions/ltcdc/CharBERTDrug

python scripts/training/train_fasttext.py --task classification --do_predict --data_dir artifacts/data/ltcdc --test_file annotated_test.txt --model_dir artifacts/models/fastTextML --prediction_output_dir artifacts/predictions/ltcdc/fastTextML
python scripts/training/train_biowordvec.py --task classification --do_predict --data_dir artifacts/data/ltcdc --test_file annotated_test.txt --model_dir artifacts/models/BioWordVecML --prediction_output_dir artifacts/predictions/ltcdc/BioWordVecML --biowordvec_path pretrained-models/BioWordVec_PubMed_MIMICIII_d200.vec.bin
python scripts/training/run_spellchecker.py --data-dir artifacts/data/ltcdc --test-file annotated_test.txt --dictionary-data-dir artifacts/data/rxnorm --output-dir artifacts/predictions/ltcdc/SpellChecker
```

Merge the five outputs:

```bash
python scripts/evaluation/assemble_predictions.py \
  --samples artifacts/data/ltcdc/annotated_test_metadata.csv \
  --key-column index --prediction-key-column index \
  --charbert artifacts/predictions/ltcdc/CharBERTDrug/predictions.csv \
  --bert artifacts/predictions/ltcdc/BERTDrug/predictions.csv \
  --spellchecker artifacts/predictions/ltcdc/SpellChecker/test_predictions.csv \
  --fasttext artifacts/predictions/ltcdc/fastTextML/test_predictions.csv \
  --biowordvec artifacts/predictions/ltcdc/BioWordVecML/test_predictions.csv \
  --output artifacts/predictions/ltcdc_predictions.csv
```

Generate the manuscript analyses:

```bash
# Point estimates: Tables 3, 4, and Appendix 4
python scripts/evaluation/evaluate_predictions.py --input artifacts/predictions/ltcdc_predictions.csv --output-dir artifacts/results/ltcdc_point --stratify term_type --stratify frequency

# LTCDC confidence intervals and Figures 2-3
python scripts/analysis/bootstrap_ltcdc.py --input artifacts/predictions/ltcdc_predictions.csv --output-dir artifacts/results/bootstrap_ltcdc --n-bootstrap 2000 --metric-set all
python scripts/analysis/plot_figure2.py --input artifacts/predictions/ltcdc_predictions.csv --output artifacts/figures/Figure_2.png --data-type oov --n-bootstrap 2000
python scripts/analysis/plot_figure3.py --input artifacts/predictions/ltcdc_predictions.csv --output artifacts/figures/Figure_3.png --data-type oov --n-bootstrap 2000

# Main-workflow error analysis
python scripts/analysis/error_analysis.py --input artifacts/predictions/ltcdc_predictions.csv --subset cleaned --output-dir artifacts/results/error_analysis
```

## Additional analysis files

- `scripts/LLM_expts/`: LLM experiment and stability scripts, README, and configuration.
- `scripts/model_error_analysis/`: detailed model-error analysis and its original README.
- `scripts/review_burden_projection/`: Appendix 8 projection script, configuration, and original README.
- `scripts/analyze_factors_affecting_model_performance.Rmd`: Appendix 7 regression analysis.
- `scripts/results/`: the LTCDC and RxNorm result CSVs used by these analyses.

Run these analyses from `scripts/` so their relative `results/` paths resolve to the included result files. See the original [LLM experiment README](scripts/LLM_expts/README.md), [model-error README](scripts/model_error_analysis/README.md), and [review-burden README](scripts/review_burden_projection/README.md) for their usage instructions.

## Script-to-result map

| Script | Manuscript use |
|---|---|
| `scripts/data/prepare_rxnorm.py` | Appendix 1; synthetic misspelling generation and RxNorm splits |
| `scripts/data/prepare_ltcdc.py` | Appendix 2; conversion of the raw reviewed LTCDC workbooks |
| `scripts/data/describe_datasets.py` | Tables 1 and 2 |
| `scripts/training/train_transformer.py`, `scripts/training/predict_transformer.py` | BERTDrug and CharBERTDrug training/inference |
| `scripts/training/train_fasttext.py`, `scripts/training/train_biowordvec.py`, `scripts/training/run_spellchecker.py` | Baseline training/inference |
| `scripts/analysis/summarize_training_sweep.py`, `scripts/analysis/plot_training_curve.py` | Appendix 3 training-size results |
| `scripts/training/make_bootstrap_samples.py`, `scripts/analysis/bootstrap_rxnorm.py` | Table 3 stability and Appendix 3 stability results |
| `scripts/evaluation/assemble_predictions.py`, `scripts/evaluation/evaluate_predictions.py` | Tables 3 and 4 point estimates |
| `scripts/analysis/bootstrap_ltcdc.py` | Table 4 and Appendix 4 confidence intervals |
| `scripts/analysis/plot_figure2.py`, `scripts/analysis/plot_figure3.py` | Figures 2 and 3 |
| `scripts/analysis/error_analysis.py` | Manuscript error analysis |
| `scripts/analyze_factors_affecting_model_performance.Rmd` | Appendix 7 performance-factor regressions |
| `scripts/model_error_analysis/analyze_model_errors.py` | Detailed manuscript error analysis |
| `scripts/review_burden_projection/simulate_operational_implications.py` | Appendix 8 operational projections |
| `scripts/LLM_expts/` | Original LLM experiments and stability analysis |

The main Python entry points support `--help`. 

The original code in this release is available under the [MIT License](LICENSE). See [NOTICE](NOTICE) for third-party attribution and [CITATION.cff](CITATION.cff) for citation metadata.
