from huggingface_hub import list_repo_files, hf_hub_download
from pathlib import Path
import pandas as pd
import csv
import ast

# =========================
# Config
# =========================
REPO_ID = "puyang2025/phish-email-datasets"
REPO_TYPE = "dataset"

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = "/mnt/NAS/home/joe/hf_cache"
OUTPUT_DIR = BASE_DIR / "hw6_task1_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_SAFE = 10
NUM_PHISHING = 20
RANDOM_SEED = 42

LABEL_MAP = {
    0: "safe",
    1: "phishing",
}

# =========================
# Helpers
# =========================
def normalize_urls(x):
    if x is None:
        return 0

    try:
        if pd.isna(x):
            return 0
    except Exception:
        pass

    if isinstance(x, (int, float)):
        return int(x)

    if isinstance(x, list):
        return len(x)

    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return 0

        if s.isdigit():
            return int(s)

        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return len(parsed)
            if isinstance(parsed, (int, float)):
                return int(parsed)
        except Exception:
            pass

        # 一般字串但不是純數字，視為至少有 URL 資訊
        return 1

    return 0


def pick_existing_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_dataframe(df):
    text_col = pick_existing_column(df, ["body", "text", "email_body"])
    subject_col = pick_existing_column(df, ["subject", "email_subject"])
    label_col = pick_existing_column(df, ["label"])
    sender_col = pick_existing_column(df, ["sender", "from"])
    urls_col = pick_existing_column(df, ["urls", "url_count", "url"])

    if text_col is None or label_col is None:
        return pd.DataFrame(columns=["body", "subject", "label", "sender", "urls"])

    work = pd.DataFrame()
    work["body"] = df[text_col].fillna("").astype(str).str.strip()
    work["subject"] = df[subject_col].fillna("").astype(str).str.strip() if subject_col else ""
    work["sender"] = df[sender_col].fillna("").astype(str).str.strip() if sender_col else ""
    work["urls"] = df[urls_col].apply(normalize_urls) if urls_col else 0
    work["label_raw"] = df[label_col]

    # 只保留 0 / 1
    work = work[work["label_raw"].isin([0, 1])].copy()
    work["label"] = work["label_raw"].map(LABEL_MAP)

    # 去掉太短 / 空白 body
    work = work[work["body"] != ""].copy()
    work = work[work["body"].str.len() > 30].copy()

    return work[["body", "subject", "label", "sender", "urls"]]


# =========================
# 1. List parquet files
# =========================
print("Listing parquet files from Hugging Face...")
repo_files = list_repo_files(REPO_ID, repo_type=REPO_TYPE)
parquet_files = [f for f in repo_files if f.endswith(".parquet")]

print(f"Found {len(parquet_files)} parquet files")

# =========================
# 2. Read and normalize all parquet files
# =========================
all_parts = []

for file_name in parquet_files:
    print(f"Downloading and reading: {file_name}")
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename=file_name,
        cache_dir=CACHE_DIR,
    )

    try:
        df = pd.read_parquet(local_path)
        clean_df = normalize_dataframe(df)
        if len(clean_df) > 0:
            all_parts.append(clean_df)
            print(f"  -> usable rows: {len(clean_df)}")
        else:
            print("  -> no usable rows")
    except Exception as e:
        print(f"  -> skipped due to error: {e}")

if not all_parts:
    raise RuntimeError("No usable rows were loaded from any parquet files.")

full_df = pd.concat(all_parts, ignore_index=True)
print(f"Total usable rows: {len(full_df)}")

# =========================
# 3. Sample rows
# =========================
safe_df = full_df[full_df["label"] == "safe"].copy()
phish_df = full_df[full_df["label"] == "phishing"].copy()

print(f"Safe rows available: {len(safe_df)}")
print(f"Phishing rows available: {len(phish_df)}")

safe_sample = safe_df.sample(n=min(NUM_SAFE, len(safe_df)), random_state=RANDOM_SEED)
phish_sample = phish_df.sample(n=min(NUM_PHISHING, len(phish_df)), random_state=RANDOM_SEED)

sample_df = (
    pd.concat([safe_sample, phish_sample], axis=0)
    .sample(frac=1, random_state=RANDOM_SEED)
    .reset_index(drop=True)
)

sample_df.insert(0, "incident_id", [f"PH-{i+1:03d}" for i in range(len(sample_df))])

# =========================
# 4. Save CSV
# =========================
csv_output = sample_df[["incident_id", "body", "subject", "label", "sender", "urls"]].copy()

csv_path = OUTPUT_DIR / "task1_simplified_dataset.csv"
csv_output.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"Saved CSV to: {csv_path}")

# =========================
# 5. Save markdown knowledge base
# =========================
md_path = OUTPUT_DIR / "task1_knowledge_base.md"

with open(md_path, "w", encoding="utf-8") as f:
    for _, row in sample_df.iterrows():
        f.write(f"## Incident {row['incident_id']}\n")
        f.write(f"- Subject: {row['subject']}\n")
        f.write(f"- Sender: {row['sender']}\n")
        f.write(f"- Label: {row['label']}\n")
        f.write(f"- URLs: {row['urls']}\n")
        f.write(f"- Body: {row['body']}\n\n")
        f.write("---\n\n")

print(f"Saved markdown KB to: {md_path}")

# =========================
# 6. Preview
# =========================
print("\n=== Preview ===")
print(csv_output.head(10).to_string(index=False))
