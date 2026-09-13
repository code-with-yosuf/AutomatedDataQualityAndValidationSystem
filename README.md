# Automated Data Quality & Validation System

This project implements a working data-quality pipeline in four modules:

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

## Run locally

```bash
python run_pipeline.py
```

This creates the output files in the `outputs/` folder.
