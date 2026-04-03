import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent
RUBRIC_KEYS = [
    "schema_compliance",
    "label_consistency",
    "sender_plausibility",
    "url_consistency",
    "realism_and_quality",
    "phishing_signal_clarity",
]


class RefineAgent:
    def __init__(
        self,
        model_name: str = "google/gemma-3-4b-it",
        input_json: str = str(BASE_DIR / "hw6_task3_output" / "batch_run" / "synthetic_emails.json"),
        output_json: Optional[str] = None,
        max_new_tokens: int = 200,
        max_retries: int = 3,
        refine_threshold: int = 3,
        max_input_length: int = 2048,
        gpu_devices: str = "0,1",
    ):
        self.model_name = model_name
        self.input_json = Path(input_json)
        self.output_json = Path(output_json) if output_json else self.input_json.with_name(
            f"{self.input_json.stem}_refine.json"
        )
        self.report_json = self.output_json.with_name(f"{self.output_json.stem}_report.json")
        self.max_new_tokens = max_new_tokens
        self.max_retries = max_retries
        self.refine_threshold = refine_threshold
        self.max_input_length = max_input_length
        self.gpu_devices = gpu_devices

        self.tokenizer = None
        self.model = None

    def load_model(self):
        print(f"Loading refine model: {self.model_name}", flush=True)

        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", self.gpu_devices)
        print(f"Using CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
        try:
            import torch._dynamo as dynamo
            dynamo.disable()
        except Exception:
            pass

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        self.model.eval()

    def clean_model_output(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
        text = text.replace("```", "")
        return text.strip()

    def extract_json_object(self, text: str) -> Optional[str]:
        text = self.clean_model_output(text)
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape = False

        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start:i + 1]
        return None

    def safe_json_load(self, text: str) -> Dict[str, Any]:
        json_block = self.extract_json_object(text)
        if json_block is None:
            raise ValueError("Could not find JSON object in model output.")
        return json.loads(json_block)

    def count_urls_in_body(self, body: str) -> int:
        url_pattern = re.compile(r"""(?ix)
            \b(
                https?://[^\s<>"']+
                |
                www\.[^\s<>"']+
            )
        """)
        return len(url_pattern.findall(body))

    def validate_record(self, record: Dict[str, Any]) -> bool:
        required_keys = ["body", "subject", "label", "sender", "urls"]
        for key in required_keys:
            if key not in record:
                return False

        if record["label"] not in ["safe", "phishing"]:
            return False

        if not isinstance(record["subject"], str):
            return False
        if not isinstance(record["sender"], str):
            return False
        if not isinstance(record["body"], str):
            return False

        if not isinstance(record["urls"], int):
            try:
                record["urls"] = int(record["urls"])
            except Exception:
                return False

        if record["urls"] < 0:
            return False

        if len(record["body"].strip()) < 20:
            return False

        if self.count_urls_in_body(record["body"]) != record["urls"]:
            return False

        return True

    def validate_rubric_scores(self, scores: Dict[str, Any]) -> bool:
        if not isinstance(scores, dict):
            return False

        for key in RUBRIC_KEYS:
            if key not in scores:
                return False
            if not isinstance(scores[key], int):
                return False
            if scores[key] < 1 or scores[key] > 5:
                return False

        return True

    def normalize_needs_refine(self, value: Any) -> Optional[bool]:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ["true", "yes", "1"]:
                return True
            if lowered in ["false", "no", "0"]:
                return False
        return None

    def should_refine(self, scores: Dict[str, Any], model_flag: bool) -> bool:
        if model_flag:
            return True
        return any(scores[key] <= self.refine_threshold for key in RUBRIC_KEYS)

    def build_judge_prompt(self, record: Dict[str, Any]) -> str:
        record_json = json.dumps(record, ensure_ascii=False, indent=2)
        rubric = """Rubric (score each item from 1 to 5):
1. schema_compliance: Are all required fields present with correct types?
2. label_consistency: Does the body match the label safe/phishing?
3. sender_plausibility: Does the sender look realistic and internally consistent?
4. url_consistency: Does the urls field match URLs mentioned in the body?
5. realism_and_quality: Does the email read like a realistic email rather than random or repetitive text?
6. phishing_signal_clarity: If phishing, are the suspicious cues clear? If safe, does it look naturally benign?

Scoring guide:
- 5 = excellent
- 4 = good
- 3 = acceptable but needs improvement
- 2 = weak
- 1 = poor"""

        return f"""You are a judge agent for a phishing email dataset.

Your job:
1. Evaluate the given record using the rubric below.
2. Decide whether the record should be refined.
3. Return exactly one valid JSON object.

{rubric}

Input record:
{record_json}

Output JSON schema:
{{
  "rubric_scores": {{
    "schema_compliance": 1,
    "label_consistency": 1,
    "sender_plausibility": 1,
    "url_consistency": 1,
    "realism_and_quality": 1,
    "phishing_signal_clarity": 1
  }},
  "overall_feedback": "short string",
  "needs_refine": true
}}

Return only the JSON object."""

    def build_refine_prompt(self, record: Dict[str, Any], judge_result: Dict[str, Any]) -> str:
        record_json = json.dumps(record, ensure_ascii=False, indent=2)
        judge_json = json.dumps(judge_result, ensure_ascii=False, indent=2)

        return f"""You are a refine agent for a phishing email dataset.

Your job:
1. Read the original record and the judge feedback.
2. Improve the record where needed.
3. Return exactly one valid JSON object.

Original record:
{record_json}

Judge result:
{judge_json}

Requirements for refined_record:
- Keep exactly these fields: body, subject, label, sender, urls
- label must be either "safe" or "phishing"
- urls must be an integer
- urls must match the number of URLs inside body
- body should be realistic, coherent, and not repetitive
- preserve the original intent when possible, but fix weak or inconsistent details

Output JSON schema:
{{
  "refined_record": {{
    "body": "string",
    "subject": "string",
    "label": "safe or phishing",
    "sender": "string",
    "urls": 0
  }}
}}

Return only the JSON object."""

    def generate_text(self, prompt: str) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
        ).to(self.model.device)

        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][input_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return text

    def judge_one_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        last_error = "Unknown judge error."

        for attempt in range(1, self.max_retries + 1):
            try:
                prompt = self.build_judge_prompt(record)
                raw_output = self.generate_text(prompt)
                parsed = self.safe_json_load(raw_output)

                if "rubric_scores" not in parsed or "overall_feedback" not in parsed or "needs_refine" not in parsed:
                    raise ValueError("Judge output is missing required fields.")

                if not self.validate_rubric_scores(parsed["rubric_scores"]):
                    raise ValueError("Judge rubric scores are invalid.")

                needs_refine = self.normalize_needs_refine(parsed["needs_refine"])
                if needs_refine is None:
                    raise ValueError("Judge needs_refine is invalid.")

                return {
                    "rubric_scores": parsed["rubric_scores"],
                    "overall_feedback": parsed["overall_feedback"],
                    "needs_refine": needs_refine,
                }
            except Exception as e:
                last_error = f"Judge attempt {attempt} failed: {e}"

        raise ValueError(last_error)

    def refine_one_record(self, record: Dict[str, Any], judge_result: Dict[str, Any]) -> Dict[str, Any]:
        last_error = "Unknown refine error."

        for attempt in range(1, self.max_retries + 1):
            try:
                prompt = self.build_refine_prompt(record, judge_result)
                raw_output = self.generate_text(prompt)
                parsed = self.safe_json_load(raw_output)

                if "refined_record" not in parsed:
                    raise ValueError("Refine output is missing refined_record.")

                refined_record = parsed["refined_record"]
                if not self.validate_record(refined_record):
                    raise ValueError("Refined record failed validation.")

                return refined_record
            except Exception as e:
                last_error = f"Refine attempt {attempt} failed: {e}"

        raise ValueError(last_error)

    def build_report_entry(
        self,
        incident_id: str,
        original_record: Dict[str, Any],
        judge_result: Dict[str, Any],
        was_refined: bool,
        final_record: Dict[str, Any],
        status: str,
        error: str,
    ) -> Dict[str, Any]:
        return {
            "incident_id": incident_id,
            "original_record": original_record,
            "judge_result": judge_result,
            "was_refined": was_refined,
            "final_record": final_record,
            "status": status,
            "error": error,
        }

    def load_records(self) -> List[Dict[str, Any]]:
        with self.input_json.open("r", encoding="utf-8") as f:
            records = json.load(f)

        if not isinstance(records, list):
            raise ValueError("Input JSON must be a list of task3 records.")
        return records

    def process_record(self, record: Dict[str, Any], idx: int, total: int) -> Dict[str, Any]:
        incident_id = record.get("incident_id", f"record_{idx:03d}")
        print(f"[Refine {idx}/{total}] {incident_id}", flush=True)

        original_record = dict(record)
        judge_result: Dict[str, Any] = {}
        was_refined = False
        status = "accepted"
        error = ""
        final_record = dict(original_record)

        try:
            judge_result = self.judge_one_record(original_record)
            needs_refine = self.should_refine(
                judge_result["rubric_scores"],
                judge_result["needs_refine"],
            )

            if needs_refine:
                final_record = self.refine_one_record(original_record, judge_result)
                was_refined = True
                status = "refined"
            else:
                final_record = dict(original_record)
                status = "accepted"

        except Exception as e:
            final_record = dict(original_record)
            was_refined = False
            status = "fallback_original"
            error = str(e)

        if "incident_id" in original_record:
            final_record["incident_id"] = original_record["incident_id"]

        if not self.validate_record(final_record):
            if self.validate_record(dict(original_record)):
                final_record = dict(original_record)
                if "incident_id" in original_record:
                    final_record["incident_id"] = original_record["incident_id"]
                was_refined = False
                status = "fallback_original"
                message = "Final record failed strict validation; original record was kept."
                error = f"{error} | {message}" if error else message
            else:
                status = "skipped_invalid"
                message = "Both final record and original record failed strict validation."
                error = f"{error} | {message}" if error else message
                return {
                    "save_record": False,
                    "record": {},
                    "report": self.build_report_entry(
                        incident_id=incident_id,
                        original_record=original_record,
                        judge_result=judge_result,
                        was_refined=was_refined,
                        final_record={},
                        status=status,
                        error=error,
                    ),
                }

        return {
            "save_record": True,
            "record": final_record,
            "report": self.build_report_entry(
                incident_id=incident_id,
                original_record=original_record,
                judge_result=judge_result,
                was_refined=was_refined,
                final_record=final_record,
                status=status,
                error=error,
            ),
        }

    def run(self, record_index: Optional[int] = None):
        self.load_model()
        records = self.load_records()

        if record_index is not None:
            if record_index < 0 or record_index >= len(records):
                raise IndexError(f"record_index {record_index} is out of range for {len(records)} records.")
            result = self.process_record(records[record_index], record_index + 1, len(records))
            single_output = [result["record"]] if result["save_record"] else []
            single_report = [result["report"]]

            self.output_json.parent.mkdir(parents=True, exist_ok=True)
            with self.output_json.open("w", encoding="utf-8") as f:
                json.dump(single_output, f, ensure_ascii=False, indent=2)

            with self.report_json.open("w", encoding="utf-8") as f:
                json.dump(single_report, f, ensure_ascii=False, indent=2)

            print(f"Saved refined JSON to: {self.output_json}", flush=True)
            print(f"Saved refine report to: {self.report_json}", flush=True)
            return

        refined_records: List[Dict[str, Any]] = []
        refine_report: List[Dict[str, Any]] = []

        for idx, record in enumerate(records, start=1):
            result = self.process_record(record, idx, len(records))
            if result["save_record"]:
                refined_records.append(result["record"])
            refine_report.append(result["report"])

        self.output_json.parent.mkdir(parents=True, exist_ok=True)
        with self.output_json.open("w", encoding="utf-8") as f:
            json.dump(refined_records, f, ensure_ascii=False, indent=2)

        with self.report_json.open("w", encoding="utf-8") as f:
            json.dump(refine_report, f, ensure_ascii=False, indent=2)

        print(f"Saved refined JSON to: {self.output_json}", flush=True)
        print(f"Saved refine report to: {self.report_json}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-json",
        default=str(BASE_DIR / "hw6_task3_output" / "batch_run" / "synthetic_emails.json"),
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--model-name", default="google/gemma-3-4b-it")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--refine-threshold", type=int, default=3)
    parser.add_argument("--max-input-length", type=int, default=2048)
    parser.add_argument("--gpu-devices", default="0,1")
    parser.add_argument("--record-index", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    agent = RefineAgent(
        model_name=args.model_name,
        input_json=args.input_json,
        output_json=args.output_json,
        max_new_tokens=args.max_new_tokens,
        max_retries=args.max_retries,
        refine_threshold=args.refine_threshold,
        max_input_length=args.max_input_length,
        gpu_devices=args.gpu_devices,
    )
    agent.run(record_index=args.record_index)
