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
- [module_3_validation/validator.py](module_3_validation/validator.py) – validates the cleaned dataset
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

## Run locally

```bash
python -m pip install -r requirements.txt
python run_pipeline.py
```

This creates `profiling_report.json`, `cleaned_data.csv`, `cleaning_log.json`, `validation_report.json`, and the pipeline summary in `outputs/`. Module 1 also writes its heatmap and distribution PNG files under `outputs/profiling_visualizations/`.
