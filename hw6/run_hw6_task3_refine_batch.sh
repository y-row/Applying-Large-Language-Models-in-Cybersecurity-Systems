#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

input_json="${1:-${SCRIPT_DIR}/hw6_task3_output/batch_run/synthetic_emails.json}"
output_dir="${2:-${SCRIPT_DIR}/hw6_task3_output/refine_batch_run}"
gpu_devices="${3:-0}"

records_dir="${output_dir}/records"
reports_dir="${output_dir}/reports"
logs_dir="${output_dir}/logs"
failed_runs_log="${output_dir}/failed_runs.txt"
final_json="${output_dir}/synthetic_emails_refine.json"
final_report_json="${output_dir}/synthetic_emails_refine_report.json"

mkdir -p "${records_dir}" "${reports_dir}" "${logs_dir}"
rm -f "${records_dir}"/record_*.json
rm -f "${reports_dir}"/report_*.json
rm -f "${logs_dir}"/run_*.log
rm -f "${failed_runs_log}"

num_records=$(python - "${input_json}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)
print(len(data))
PY
)

success_count=0
fail_count=0

for ((i=0; i<num_records; i++)); do
  run_no=$((i + 1))
  output_file=$(printf "record_%03d.json" "${run_no}")
  report_file=$(printf "report_%03d.json" "${run_no}")
  run_log=$(printf "%s/run_%03d.log" "${logs_dir}" "${run_no}")

  echo "[Refine ${run_no}/${num_records}] record_index=${i}"
  if python "${SCRIPT_DIR}/hw6_task3_refine.py" \
    --input-json "${input_json}" \
    --output-json "${records_dir}/${output_file}" \
    --gpu-devices "${gpu_devices}" \
    --record-index "${i}" > "${run_log}" 2>&1; then
    report_path="${records_dir}/${output_file%.*}_report.json"
    if [[ -f "${report_path}" ]]; then
      mv "${report_path}" "${reports_dir}/${report_file}"
    fi
    echo "  -> success"
    success_count=$((success_count + 1))
  else
    echo "  -> failed, skipped"
    echo "record_index=${i} log=${run_log}" >> "${failed_runs_log}"
    fail_count=$((fail_count + 1))
    continue
  fi
done

python - "${records_dir}" "${reports_dir}" "${final_json}" "${final_report_json}" <<'PY'
import json
import sys
from pathlib import Path

records_dir = Path(sys.argv[1])
reports_dir = Path(sys.argv[2])
final_json = Path(sys.argv[3])
final_report_json = Path(sys.argv[4])

records = []
for path in sorted(records_dir.glob("record_*.json")):
    with path.open("r", encoding="utf-8") as f:
        records.extend(json.load(f))

reports = []
for path in sorted(reports_dir.glob("report_*.json")):
    with path.open("r", encoding="utf-8") as f:
        reports.extend(json.load(f))

final_json.parent.mkdir(parents=True, exist_ok=True)
with final_json.open("w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

with final_report_json.open("w", encoding="utf-8") as f:
    json.dump(reports, f, ensure_ascii=False, indent=2)

print(f"Saved merged refined JSON to: {final_json}")
print(f"Saved merged refine report to: {final_report_json}")
print(f"Total merged refined records: {len(records)}")
print(f"Total merged report entries: {len(reports)}")
PY

echo "Refine batch complete: success=${success_count}, failed=${fail_count}"
if [[ -f "${failed_runs_log}" ]]; then
  echo "Failed run summary: ${failed_runs_log}"
fi
