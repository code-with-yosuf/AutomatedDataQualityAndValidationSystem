# Automated Data Quality & Validation System

This project implements an automated data-quality pipeline. Modules 1 and 2 provide profiling and cleaning; Modules 3 and 4 validate and orchestrate the result.

1. Data Profiling
2. Data Cleaning
3. Data Validation
4. End-to-End Automation

## Project structure

- [run_pipeline.py](run_pipeline.py) – orchestrates the full pipeline
- [module_1_profiling/profiler.py](module_1_profiling/profiler.py) – profiles the dataset and detects metadata issues
- [module_2_cleaning/cleaner.py](module_2_cleaning/cleaner.py) – cleans and standardizes the data
- [module_3_validation/validator.py](module_3_validation/validator.py) – validates the cleaned dataset (rules, anomaly detection, column classifier, error detection, drift, scoring, evaluation)
- [module_4_automation/orchestrator.py](module_4_automation/orchestrator.py) – final pipeline summary
- [requirements.txt](requirements.txt) – Python dependencies

## Modules 1 and 2

Module 1 infers column and semantic types, flags mixed or inconsistent values, profiles missingness/cardinality/distributions/numeric outliers/correlations, checks suspicious formats and possible PII, and writes three PNG visualizations. Its JSON-safe report includes those findings and visualization paths.

Module 2 infers expected types, observed numeric ranges, and date formats; normalizes text and dates; removes exact duplicates; logs fuzzy name-match candidates without automatically deleting them; imputes numeric data using `mean`, `median`, `most_frequent`, `constant`, `knn`, `regression`, or `iterative`; imputes categorical values using the most frequent value; and records before/after quality scores in `cleaning_log.json`.

### Module 1 report structure

`profiling_report.json` contains dataset dimensions, `metadata` (inferred and semantic types, mixed-type columns, and potential PII), per-column profiles (missingness, cardinality, distributions, consistency counts, numeric statistics, and outliers), `profiling` (cardinality, distributions, and numeric correlation matrix), `quality_checks` (rule findings and suspicious columns), and `visualizations` (paths to the generated PNGs). Each column's expected type is the dominant observed type; minority values are counted as inconsistent rather than silently converted.

Public entry points:

```python
profile_dataset(file_path, output_dir=None) -> dict
clean_dataset(file_path, profile, output_dir, imputation_strategy="median",
			  fuzzy_threshold=0.92, transformations=None) -> Path
transform_features(df, scaling="standard", encode_categoricals=True,
				   extract_date_features=True) -> DataFrame
```

The cleaning report includes `schema_inference`, `actions`, `quality_score.before`, `quality_score.after`, `quality_score.delta`, and transformation output details. Fuzzy name matches are review candidates only; exact duplicate rows are removed automatically.

Feature transformations are optional and can be requested through the Python API:

```python
clean_dataset(
	input_file,
	profiling_report,
	output_dir,
	imputation_strategy="knn",
	transformations={
		"scaling": "standard",  # also "minmax" or "none"
		"encode_categoricals": True,
		"extract_date_features": True,
	},
)
```

The transformed feature matrix is written to `transformed_features.csv` only when transformations are requested.

## Module 3

`validate_dataset` reads the cleaned CSV (all values as text, so leading zeros survive) and writes `validation_report.json`:

- **Rules** (`rules.py`): email, phone and postcode formats, numeric ranges (age, percentages and amounts are inferred from column names), allowed values, and inconsistent category spelling or casing.
- **Anomaly detection** (`anomaly.py`): Isolation Forest, Local Outlier Factor and an MLP autoencoder over numeric columns. A row is anomalous when at least two detectors agree. Skipped below 10 rows.
- **Column classifier** (`nlp_classifier.py`): a character n-gram TF-IDF + logistic regression model trained at runtime on synthetic examples. It predicts each column's meaning, flags columns whose name disagrees with their values, and flags off-type values inside structured columns. Digit-only columns are ambiguous between numeric, postcode and identifier, so those pairs are never flagged.
- **Error detection** (`error_detection.py`): future or pre-1900 dates, end-before-start, duplicate identifiers and rows, empty columns, extreme values, placeholder or fake-looking values, and format outliers.
- **Drift** (`drift.py`): population stability index against a `reference` dataset (moderate at 0.1, high at 0.25). Without a reference the two row halves are compared for information only and raise no findings.
- **Scoring** (`scoring.py`): `error_severity` is the severity-weighted share of rows with an error, `anomaly_severity` is the average detector agreement, and column-level findings without row masks (drift, mislabeled or empty columns) subtract up to 30 points. `health_score = 100 - 0.6*error - 0.25*anomaly - structural`; the status is `pass` at or above `pass_threshold` (default 80).
- **Evaluation** (`evaluation.py`): precision, recall, F1, confusion matrix and ROC curves (JSON plus `roc_curves.png`). With `labels` the detectors and the combined validator are scored against real bad rows. Without labels, known anomalies are injected into the numeric features and the detectors are scored on finding them, which measures detector quality rather than data quality. Needs at least 30 rows.

```python
validate_dataset(file_path, profile=None, output_dir=None, reference=None, rules=None,
				 labels=None, contamination="auto", pass_threshold=80.0, seed=42) -> dict
```

`rules` accepts `numeric_ranges` (`{column: (min, max)}`), `allowed_values` (`{column: [...]}`) and `postcode_countries` (`us`, `uk`, `ca`, `in`). The report keeps `status`, `issues` and `passed_checks` for Module 4 and adds `scores`, `findings`, `column_classification`, `anomaly_detection`, `drift` and `evaluation`. The bundled 7-row sample is too small for anomaly detection, drift and evaluation, so those sections report `skipped` there.

## Run locally

```bash
python -m pip install -r requirements.txt
python run_pipeline.py
```

This creates `profiling_report.json`, `cleaned_data.csv`, `cleaning_log.json`, `validation_report.json`, and the pipeline summary in `outputs/`. Module 1 also writes its heatmap and distribution PNG files under `outputs/profiling_visualizations/`, and Module 3 writes `roc_curves.png` under `outputs/validation_visualizations/` when the dataset is large enough to evaluate.
