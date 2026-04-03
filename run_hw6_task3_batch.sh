#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

num_gen="${1:-20}"
output_dir="${2:-${SCRIPT_DIR}/hw6_task3_output/batch_run}"
tmp_dir="${output_dir}/records"
log_dir="${output_dir}/logs"
final_json="${output_dir}/synthetic_emails.json"
failed_runs_log="${output_dir}/failed_runs.txt"
success_count=0
fail_count=0

mkdir -p "${tmp_dir}"
mkdir -p "${log_dir}"
rm -f "${tmp_dir}"/record_*.json
rm -f "${log_dir}"/run_*.log
rm -f "${failed_runs_log}"

for ((i=1; i<=num_gen; i++)); do
  label="phishing"
  if (( i % 2 == 0 )); then
    label="safe"
  fi

  incident_id=$(printf "SYN-%03d" "${i}")
  output_file=$(printf "record_%03d.json" "${i}")
  run_log=$(printf "%s/run_%03d.log" "${log_dir}" "${i}")

  echo "[Run ${i}/${num_gen}] label=${label} incident_id=${incident_id}"
  if python "${SCRIPT_DIR}/hw6_task3.py" \
    --label "${label}" \
    --incident-id "${incident_id}" \
    --output-dir "${tmp_dir}" \
    --output-file "${output_file}" > "${run_log}" 2>&1; then
    echo "  -> success"
    success_count=$((success_count + 1))
  else
    echo "  -> failed, skipped"
    echo "run=${i} label=${label} incident_id=${incident_id} log=${run_log}" >> "${failed_runs_log}"
    fail_count=$((fail_count + 1))
    continue
  fi
done

python - "${tmp_dir}" "${final_json}" <<'PY'
import json
import sys
from pathlib import Path

tmp_dir = Path(sys.argv[1])
final_json = Path(sys.argv[2])
records = []

for path in sorted(tmp_dir.glob("record_*.json")):
    with path.open("r", encoding="utf-8") as f:
        records.append(json.load(f))

final_json.parent.mkdir(parents=True, exist_ok=True)
with final_json.open("w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"Saved merged JSON to: {final_json}")
print(f"Total merged records: {len(records)}")
PY

echo "Batch complete: success=${success_count}, failed=${fail_count}"
if [[ -f "${failed_runs_log}" ]]; then
  echo "Failed run summary: ${failed_runs_log}"
fi
