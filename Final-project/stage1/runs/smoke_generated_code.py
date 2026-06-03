import argparse
import json
import ntpath


def get_path(obj, dotted):
    cur = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as src, open(args.output, "w", encoding="utf-8") as out:
        for line_no, line in enumerate(src, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            event_id = str(get_path(event, "event.code") or get_path(event, "winlog.event_id") or "")
            event_data = get_path(event, "winlog.event_data") or {}
            command_line = event_data.get("CommandLine") or get_path(event, "message")
            image = event_data.get("Image") or event_data.get("NewProcessName")
            if event_id not in {"1", "4688"}:
                continue
            if not image and not command_line:
                continue

            parent_image = event_data.get("ParentImage") or event_data.get("ParentProcessName")
            record = {
                "timestamp": event.get("@timestamp") or event_data.get("UtcTime") or event.get("timestamp"),
                "host": get_path(event, "host.name") or get_path(event, "winlog.computer_name") or event.get("computer_name"),
                "user": get_path(event, "user.name") or event_data.get("User") or event_data.get("SubjectUserName"),
                "event_id": event_id,
                "event_type": "process_creation",
                "process_name": ntpath.basename(image) if image else None,
                "process_path": image,
                "command_line": command_line,
                "parent_process_name": ntpath.basename(parent_image) if parent_image else None,
                "parent_command_line": event_data.get("ParentCommandLine"),
                "process_id": event_data.get("ProcessId") or event_data.get("NewProcessId"),
                "parent_process_id": event_data.get("ParentProcessId"),
                "source_ip": None,
                "dest_ip": None,
                "dest_port": None,
                "raw_message": event.get("message") or line.strip(),
                "source_line": line_no,
            }
            out.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
