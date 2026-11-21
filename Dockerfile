FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pipeline.py ./
COPY config ./config
COPY module_1_profiling ./module_1_profiling
COPY module_2_cleaning ./module_2_cleaning
COPY module_3_validation ./module_3_validation
COPY module_4_automation ./module_4_automation

RUN useradd --create-home --uid 1000 pipeline \
    && mkdir /data /results \
    && chown pipeline /results
USER pipeline

# Mount the dataset at /data (read-only is fine) and a writable directory at /results.
ENTRYPOINT ["python", "pipeline.py"]
CMD ["--input", "/data/input.csv", "--output", "/results"]
