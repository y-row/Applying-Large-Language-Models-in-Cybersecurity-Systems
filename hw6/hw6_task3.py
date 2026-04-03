import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_DIR = Path(__file__).resolve().parent


class SyntheticEmailGenerator:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-2B",
        output_dir: str = str(BASE_DIR / "hw6_task3_output"),
        knowledge_base_path: str = str(BASE_DIR / "hw6_task1_output" / "task1_knowledge_base.md"),
        num_records: int = 20,
        max_attempts: int = 60,
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.knowledge_base_path = Path(knowledge_base_path)

        self.num_records = num_records
        self.max_attempts = max_attempts

        self.tokenizer = None
        self.model = None
        self.example_cache = None

    # =========================
    # Model
    # =========================
    def load_model(self):
        print(f"Loading model: {self.model_name}", flush=True)

        # Avoid known instability in some PyTorch/Transformers stacks where
        # torch._dynamo participates in generation and crashes the interpreter.
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
        try:
            import torch._dynamo as dynamo
            dynamo.disable()
        except Exception:
            pass

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        self.model.eval()

    # =========================
    # Prompt
    # =========================
    def load_examples_from_knowledge_base(self) -> Dict[str, Dict[str, Any]]:
        if self.example_cache is not None:
            return self.example_cache

        text = self.knowledge_base_path.read_text(encoding="utf-8")
        blocks = [block.strip() for block in text.split("\n---\n") if block.strip()]
        incidents = []

        for block in blocks:
            record = {}

            subject_match = re.search(r"^- Subject:\s*(.*)$", block, flags=re.MULTILINE)
            sender_match = re.search(r"^- Sender:\s*(.*)$", block, flags=re.MULTILINE)
            label_match = re.search(r"^- Label:\s*(.*)$", block, flags=re.MULTILINE)
            urls_match = re.search(r"^- URLs:\s*(\d+)\s*$", block, flags=re.MULTILINE)
            body_match = re.search(r"^- Body:\s*(.*)$", block, flags=re.MULTILINE | re.DOTALL)

            if not all([subject_match, sender_match, label_match, urls_match, body_match]):
                continue

            body = body_match.group(1).strip()
            record = {
                "subject": subject_match.group(1).strip(),
                "sender": sender_match.group(1).strip(),
                "label": label_match.group(1).strip(),
                "urls": int(urls_match.group(1)),
                "body": body,
            }
            incidents.append(record)

        def pick_example(target_label: str) -> Dict[str, Any]:
            for incident in incidents:
                if incident["label"] != target_label:
                    continue
                if not incident["subject"] or not incident["sender"] or not incident["body"]:
                    continue
                if incident["urls"] != self.count_urls_in_body(incident["body"]):
                    continue
                return incident
            raise ValueError(f"Could not find a valid {target_label} example in {self.knowledge_base_path}.")

        self.example_cache = {
            "phishing": pick_example("phishing"),
            "safe": pick_example("safe"),
        }
        return self.example_cache

    def format_example_json(self, record: Dict[str, Any], max_body_chars: int = 600) -> str:
        body = record["body"].strip()
        if len(body) > max_body_chars:
            body = body[:max_body_chars].rsplit(" ", 1)[0].rstrip() + " ..."

        example = {
            "body": body,
            "subject": record["subject"],
            "label": record["label"],
            "sender": record["sender"],
            "urls": record["urls"],
        }
        return json.dumps(example, ensure_ascii=False, indent=2)

    def build_prompt(self, target_label: str) -> str:
        examples = self.load_examples_from_knowledge_base()
        phishing_example = self.format_example_json(examples["phishing"])
        safe_example = self.format_example_json(examples["safe"])

        return f"""You are generating one record for a phishing email detection dataset.

Task:
Generate exactly ONE synthetic email in valid JSON format.

Output schema:
{{
  "body": "string",
  "subject": "string",
  "label": "{target_label}",
  "sender": "string",
  "urls": 0
}}

Field definitions:
- "body": the full email message content. This is the main evidence for phishing detection.
- "subject": the email subject line. It may contain urgency, invoice, password reset, account alert, meeting, shipment, or routine business language depending on the label.
- "label": must be exactly "{target_label}". Use only "phishing" or "safe".
- "sender": a realistic sender name or email address, such as "IT Support <it-support@company.com>" or "billing@vendor-example.com".
- "urls": integer count of how many URLs appear inside the email body.

Reference examples from the knowledge base:
Phishing example:
{phishing_example}

Safe example:
{safe_example}

Generation rules:
1. Output exactly one JSON object and nothing else.
2. Do not output markdown, code fences, explanations, notes, or thinking text.
3. The JSON must be parseable by Python json.loads.
4. Keep all five fields exactly as written: body, subject, label, sender, urls.
5. "urls" must be a non-negative integer.
6. The email must be realistic and internally consistent:
   - if label is "phishing": use believable social engineering signals such as urgency, credential reset, payment pressure, account verification, delivery failure, payroll, tax, or invoice issues.
   - if label is "safe": use normal benign communication such as meetings, internal notices, project updates, HR reminders, routine billing, or customer support follow-up.
7. The number in "urls" must match the number of URLs mentioned in "body".
8. "body" should be 80 to 220 words and should read like a real email.
9. Do not invent extra fields.
10. Follow the style and field structure of the reference examples, but generate a new record rather than copying them.

Return only the JSON object in one response."""

    def build_retry_prompt(self, target_label: str) -> str:
        return f"""Previous output was invalid.

Fix the format and try again.

Return exactly ONE valid JSON object with this schema:
{{
  "body": "string",
  "subject": "string",
  "label": "{target_label}",
  "sender": "string",
  "urls": 0
}}

Hard constraints:
1. Output only JSON.
2. No markdown.
3. No <think> tags.
4. No explanations.
5. No extra fields.
6. "label" must be exactly "{target_label}".
7. "urls" must equal the number of URLs in "body".
8. Keep the email concise and realistic.
"""

    # =========================
    # Generation
    # =========================
    def generate_text(self, prompt: str, max_new_tokens: int = 300) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.model.device)

        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][input_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return text
    # =========================
    # JSON parsing
    # =========================
    def clean_model_output(self, text: str) -> str:
        text = text.strip()

        # Remove common reasoning tags that some models leak.
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)

        # Remove fenced code blocks while preserving inner content.
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

    # =========================
    # Validation
    # =========================
    def validate_record(self, record: Dict[str, Any]) -> bool:
        required_keys = ["subject", "sender", "label", "urls", "body"]
        for key in required_keys:
            if key not in record:
                return False

        if record["label"] not in ["phishing", "safe"]:
            return False

        if not isinstance(record["urls"], int):
            try:
                record["urls"] = int(record["urls"])
            except Exception:
                return False
        if record["urls"] < 0:
            return False

        if not isinstance(record["subject"], str):
            return False
        if not isinstance(record["sender"], str):
            return False
        if not isinstance(record["body"], str):
            return False

        if len(record["body"].strip()) < 20:
            return False

        actual_url_count = self.count_urls_in_body(record["body"])
        if record["urls"] != actual_url_count:
            return False

        return True

    # =========================
    # Dedup heuristic
    # =========================
    def is_duplicate(self, record: Dict[str, Any], existing: list[Dict[str, Any]]) -> bool:
        subj = record["subject"].strip().lower()
        body = re.sub(r"\s+", " ", record["body"].strip().lower())

        for ex in existing:
            ex_subj = ex["subject"].strip().lower()
            ex_body = re.sub(r"\s+", " ", ex["body"].strip().lower())

            if subj == ex_subj:
                return True
            if body == ex_body:
                return True

        return False

    # =========================
    # Main loop
    # =========================
    def generate_one_record(self, target_label: str, incident_id: Optional[str] = None) -> Dict[str, Any]:
        raw_log_path = self.output_dir / f"raw_generation_log_{target_label}.txt"

        with open(raw_log_path, "w", encoding="utf-8") as raw_log:
            attempt = 0

            while attempt < self.max_attempts:
                attempt += 1
                prompt = self.build_prompt(target_label)
                print(f"[Attempt {attempt}] generating 1 record for label={target_label}", flush=True)

                output = ""
                try:
                    output = self.generate_text(prompt)
                    raw_log.write(f"=== Attempt {attempt} ===\n")
                    raw_log.write(output + "\n\n")

                    record = self.safe_json_load(output)

                    if not self.validate_record(record):
                        retry_prompt = self.build_retry_prompt(target_label)
                        retry_output = self.generate_text(retry_prompt)
                        raw_log.write(f"--- Retry {attempt} ---\n")
                        raw_log.write(retry_output + "\n\n")
                        record = self.safe_json_load(retry_output)

                        if not self.validate_record(record):
                            print("  -> invalid schema, skipped", flush=True)
                            continue

                    if record["label"] != target_label:
                        print("  -> wrong label, skipped", flush=True)
                        continue

                    if incident_id is not None:
                        record["incident_id"] = incident_id
                    print(f"  -> accepted ({record['label']})", flush=True)
                    print(f"Saved raw generation log to: {raw_log_path}", flush=True)
                    return record

                except Exception as e:
                    print(f"  -> failed: {e}", flush=True)
                    if output:
                        print("  -> raw output preview:", flush=True)
                        print(output[:1000], flush=True)
                    continue

        raise RuntimeError(f"No valid synthetic record was generated for label={target_label}.")

    # =========================
    # Save
    # =========================
    def save_output(self, record: Dict[str, Any], output_filename: str = "synthetic_email.json"):
        json_path = self.output_dir / output_filename

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        print(f"Saved JSON to: {json_path}", flush=True)

    def run(self, target_label: str, incident_id: Optional[str] = None, output_filename: str = "synthetic_email.json"):
        self.load_model()
        record = self.generate_one_record(target_label=target_label, incident_id=incident_id)
        self.save_output(record, output_filename=output_filename)

        print("\n=== Preview ===", flush=True)
        print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", choices=["phishing", "safe"], required=True)
    parser.add_argument("--incident-id", default=None)
    parser.add_argument("--output-dir", default=str(BASE_DIR / "hw6_task3_output"))
    parser.add_argument("--output-file", default="synthetic_email.json")
    parser.add_argument("--knowledge-base-path", default=str(BASE_DIR / "hw6_task1_output" / "task1_knowledge_base.md"))
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--max-attempts", type=int, default=60)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generator = SyntheticEmailGenerator(
        model_name=args.model_name,
        output_dir=args.output_dir,
        knowledge_base_path=args.knowledge_base_path,
        num_records=1,
        max_attempts=args.max_attempts,
    )
    generator.run(
        target_label=args.label,
        incident_id=args.incident_id,
        output_filename=args.output_file,
    )
