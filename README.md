# SSU Data Dash 2026: XGBROS Toronto home-price prediction 
By: Daniel Tan, Jerry Zhou, Quinton Gao, Vincent Li

## Setup

Python 3.12 or newer. Install the dependencies:

```
pip install -r requirements.txt xgboost catboost
```

or, with [uv](https://docs.astral.sh/uv/):

```
uv sync
```

The data downloads itself from [Hugging Face](https://huggingface.co/datasets/Baliyang/SSU_data_dash) the first time you run the code.

## Running the code

From the project folder:

```
python run.py validate   # walk-forward validation, Dec 2025 to May 2026
python run.py report     # scores, LaTeX table and figure from the validation run
python run.py predict    # train on all sales and write both submission files
python run.py check      # check the submission files
```

On Windows, use `py` if `python` is not found.

To regenerate only the submission files, run `predict` and then `check`.

## Where to find the output

| What | Where |
|---|---|
| **Submission files** | `submissions/submission_off_market.csv`, `submissions/submission_list_aware.csv` |
| Validation predictions | `outputs/validation_predictions.csv` |
| Validation scores | `reports/validation_report.md` (also `.json`, and `results_table.tex` for the report) |
| Error-by-price figure | `figures/error_by_price_range.png` |

Each submission file has one row per test home:

```
ml_num,predicted_price
SYN-0123456,815000
```