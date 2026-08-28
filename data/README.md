# Data

`LTC_set1_merged_reviewed_with_category.xlsx` and `LTC_set2_merged_reviewed_with_category.xlsx` are the two public LTCDC workbooks used in the study.

This directory also records the RxNorm source used by the main workflow. It does not contain RxNorm terminology exports or generated working splits; the requested RxNorm test-prediction table is under `../scripts/results/`. See [RXNORM_SOURCE.md](RXNORM_SOURCE.md).

From the `code_release/` folder, use [`scripts/data/prepare_ltcdc.py`](../scripts/data/prepare_ltcdc.py) to generate model-ready LTCDC files under `artifacts/`; generated LTCDC files are not stored in this data directory.
