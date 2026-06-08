def build_feature_texts(valid_logs: list, template_version: str = "v1") -> tuple:
    """
    將結構化日誌欄位轉換為語意特徵文本。

    v2 升級重點（template_version="v2"）：
    - 優先直接使用 prebuilt_embedding_text（系統預組字串）
    - 若缺失，以 process_action_details + 網路上下文動態組裝
    - 若 process_action_details 也缺失，退回 raw_message
    - 保留 256 Token 截斷限制

    Returns:
        (feature_texts, summary)
    """
    if not isinstance(valid_logs, list):
        raise ValueError("valid_logs 必須為 list")

    feature_texts = []
    truncated_count = 0
    max_token_threshold = 256

    for idx, log in enumerate(valid_logs):

        if not isinstance(log, dict):
            raise ValueError(f"第 {idx} 筆 log 非 dict")

        if template_version == "v1":
            # ── v1 樣板：維持原始扁平欄位組裝邏輯（向後相容）──────
            text = (
                f"At {log['timestamp']}, user {log['actor_user']} "
                f"triggered event {log['event_id']}: "
                f"{log['process_action_details']}. "
                f"Source IP: {log['source_ip']}. "
                f"Destination IP: {log['dest_ip']}."
            )

        elif template_version == "v2":
            # ── v2 樣板：三層優先序 ────────────────────────────────

            prebuilt = _nonempty(log.get("prebuilt_embedding_text"))

            if prebuilt:
                # 層級 1：直接使用系統預組高維度特徵字串
                text = prebuilt

            else:
                pad = _nonempty(log.get("process_action_details"))

                if pad:
                    # 層級 2：process_action_details + 網路上下文動態組裝
                    conn_state  = log.get("conn_state")  or "unknown"
                    protocol    = log.get("protocol")    or "unknown"
                    dest_port   = log.get("dest_port")   or "unknown"
                    source_port = log.get("source_port") or "unknown"
                    service     = log.get("service")     or "unknown"

                    text = (
                        f"At {log['timestamp']}, user {log['actor_user']} "
                        f"triggered event {log['event_id']}: {pad}. "
                        f"Source: {log['source_ip']}:{source_port} -> "
                        f"Dest: {log['dest_ip']}:{dest_port}. "
                        f"Protocol: {protocol}. "
                        f"Conn state: {conn_state}. "
                        f"Service: {service}."
                    )

                else:
                    # 層級 3：退回 raw_message
                    raw = _nonempty(log.get("raw_message"))
                    if raw:
                        text = raw
                    else:
                        # 防護：validator 應已剔除此類日誌，此處為最後安全網
                        raise ValueError(
                            f"第 {idx} 筆 log 無任何可用語意文本 "
                            f"(log_id={log.get('log_id')})"
                        )

        else:
            # 未知版本：退回 process_action_details（最低限度相容）
            text = log["process_action_details"]

        # ── 256 Token 截斷控制（whitespace 估算）──────────────────
        tokens = text.split()
        if len(tokens) > max_token_threshold:
            text = " ".join(tokens[:max_token_threshold])
            truncated_count += 1

        feature_texts.append(text)

    # Invariant 檢查
    if len(feature_texts) != len(valid_logs):
        raise RuntimeError("Invariant violated: feature_texts length mismatch")

    summary = {
        "feature_template_version":  template_version,
        "truncated_count_estimate":  truncated_count,
    }

    return feature_texts, summary


# ── 內部工具 ───────────────────────────────────────────────────────────────

def _nonempty(value) -> str | None:
    """回傳非空字串值；None / 空字串一律回傳 None。"""
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None
