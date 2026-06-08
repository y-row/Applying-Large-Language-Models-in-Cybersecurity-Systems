import json
import os


def load_logs(file_path: str) -> list:
    """
    讀取 JSON Lines (.jsonl) 格式檔案，逐行解析並回傳原始 log list。

    v2 升級重點：
    - 支援 .jsonl 格式（每行一個獨立 JSON 物件）
    - 空行自動跳過
    - 單行 JSONDecodeError 容錯，不中斷整體批次流程

    Returns:
        list: 原始 log list（每個元素為 dict）
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到輸入的日誌檔案: {file_path}")

    raw_logs = []
    parse_error_count = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            # 忽略空行
            stripped = line.strip()
            if not stripped:
                continue

            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as e:
                # 單行損毀不中斷批次；記錄警告後繼續
                print(f"[WARNING] 第 {line_number} 行 JSON 解析失敗，已跳過: {e}")
                parse_error_count += 1
                continue

            if not isinstance(record, dict):
                print(f"[WARNING] 第 {line_number} 行非 JSON object，已跳過。")
                parse_error_count += 1
                continue

            raw_logs.append(record)

    if parse_error_count > 0:
        print(f"[INFO] 載入完畢，共跳過 {parse_error_count} 筆損毀/格式異常行。")

    return raw_logs
