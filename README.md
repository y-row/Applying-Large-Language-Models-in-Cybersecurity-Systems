# HW6 Email Pipeline

This project contains the code and outputs for an HW6 email security workflow. It covers:

1. Building a simplified phishing email knowledge base from a Hugging Face dataset
2. Running a small retrieval + generation demo for email analysis
3. Generating synthetic email records in JSON format for phishing detection experiments

## Project Structure

```text
hw6_email_pipeline/
├── README.md
├── flowchart.md
├── hw6.py
├── hw6_task1.py
├── hw6_task1_output/
│   ├── task1_knowledge_base.md
│   └── task1_simplified_dataset.csv
├── hw6_task3.py
├── hw6_task3_output/
│   └── ...
└── run_hw6_task3_batch.sh
```

## Files

- `hw6_task1.py`
  Downloads phishing email parquet files from Hugging Face, normalizes the schema, samples records, and exports:
  - `hw6_task1_output/task1_simplified_dataset.csv`
  - `hw6_task1_output/task1_knowledge_base.md`

- `hw6.py`
  A simple retrieval-augmented generation demo using:
  - `langchain`
  - `FAISS`
  - `HuggingFaceEmbeddings`
  - `Qwen`

- `hw6_task3.py`
  Generates one synthetic email record at a time in JSON format. The prompt includes:
  - a strict JSON schema
  - field definitions
  - one phishing example and one safe example loaded from `task1_knowledge_base.md`

- `run_hw6_task3_batch.sh`
  Runs `hw6_task3.py` multiple times in separate processes to reduce the chance of runtime crashes during repeated generation.

## Task 1: Build the Knowledge Base

Run:

```bash
python hw6_task1.py
```

Outputs:

- `hw6_task1_output/task1_simplified_dataset.csv`
- `hw6_task1_output/task1_knowledge_base.md`

## Task 2: Retrieval + Generation Demo

Run:

```bash
python hw6.py
```

This script builds a small vector store from sample records, retrieves the most relevant records for a query, and asks a Qwen model to produce a short structured answer.

## Task 3: Synthetic Email Generation

### Generate One Record

Run:

```bash
python hw6_task3.py --label phishing --incident-id SYN-001 --output-file record_001.json
```

Or:

```bash
python hw6_task3.py --label safe --incident-id SYN-002 --output-file record_002.json
```

Output:

- `hw6_task3_output/<your_output_file>.json`

### Batch Generation

Run:

```bash
bash run_hw6_task3_batch.sh
```

Default behavior:

- runs 20 rounds
- odd-numbered rounds generate `phishing`
- even-numbered rounds generate `safe`
- failed rounds are skipped instead of stopping the whole batch

Batch outputs:

- merged result:
  - `hw6_task3_output/batch_run/synthetic_emails.json`
- per-record JSON files:
  - `hw6_task3_output/batch_run/records/record_001.json`
- per-run logs:
  - `hw6_task3_output/batch_run/logs/run_001.log`
- failed run summary:
  - `hw6_task3_output/batch_run/failed_runs.txt`

## Environment Notes

This project depends on your local Python environment. In particular:

- `hw6_task1.py` needs packages such as `pandas`, `pyarrow`, and `huggingface_hub`
- `hw6.py` needs `langchain`, `faiss`, `transformers`, and embedding dependencies
- `hw6_task3.py` needs `torch` and `transformers`

If you run `hw6_task3.py` repeatedly in one long Python process, some model/runtime combinations may become unstable. The batch shell script avoids that by launching a fresh Python process for each generated sample.

## Notes

- Paths in the scripts are now based on the script location, so the project can be moved as one folder.
- The synthetic generation prompt is schema-driven and includes few-shot examples from the task 1 knowledge base.
- The generator validates output structure and checks whether the declared `urls` count matches the URLs found in the generated email body.
