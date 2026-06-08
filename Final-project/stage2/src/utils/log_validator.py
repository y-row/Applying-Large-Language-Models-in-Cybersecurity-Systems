def validate_logs(raw_logs: list) -> tuple:
    """
    執行 v2 日誌結構解構、欄位映射、Schema 驗證與補值。

    v2 升級重點：
    - 安全提取 metadata / normalized_event 巢狀節點
    - log_id 多重來源提取（嚴格優先序）
    - process_action_details 三層 fallback 閉環
    - prebuilt_embedding_text / raw_message 保留映射

    Returns:
        tuple:
            valid_logs (list)   — 通過驗證並整平的日誌列表
            summary_dict (dict) — 統計摘要
    """
    if not isinstance(raw_logs, list):
        raise ValueError("raw_logs 必須為 list")

    valid_logs = []
    invalid_log_count = 0
    fallback_count = 0

    replacement_counts = {
        "source_ip": 0,
        "dest_ip": 0,
        "actor_user": 0,
        "event_id": 0,
    }

    for idx, log in enumerate(raw_logs):

        if not isinstance(log, dict):
            invalid_log_count += 1
            continue

        # ── 安全取得子節點 ─────────────────────────────────────────
        metadata         = log.get("metadata") or {}
        normalized_event = log.get("normalized_event") or {}

        if not isinstance(metadata, dict):
            metadata = {}
        if not isinstance(normalized_event, dict):
            normalized_event = {}

        # ── 1. log_id 多重來源提取（嚴格優先序）────────────────────
        #   metadata.event_uid -> normalized_event.event_uid
        #   -> 全域 event_uid  -> normalized_event.source_event_id
        log_id = (
            _nonempty(metadata.get("event_uid"))
            or _nonempty(normalized_event.get("event_uid"))
            or _nonempty(log.get("event_uid"))
            or _nonempty(normalized_event.get("source_event_id"))
        )
        if not log_id:
            invalid_log_count += 1
            continue

        # ── 2. timestamp（全域節點，缺失則剔除）────────────────────
        timestamp = _nonempty(log.get("timestamp"))
        if not timestamp:
            invalid_log_count += 1
            continue

        # ── 3. text_for_embedding 保留（全域優先，fallback normalized）
        prebuilt_embedding_text = (
            _nonempty(log.get("text_for_embedding"))
            or _nonempty(normalized_event.get("text_for_embedding"))
        )

        # ── 4. raw_message 保留（僅作 fallback 來源）───────────────
        raw_message = _nonempty(normalized_event.get("raw_message"))

        # ── 5. process_action_details 三層 fallback 閉環 ───────────
        #   primary  : normalized_event.selection_reason
        #   fallback1: normalized_event.raw_message
        #   fallback2: text_for_embedding（全域）
        #   全部缺失  : invalid
        selection_reason = _nonempty(normalized_event.get("selection_reason"))

        if selection_reason:
            process_action_details = selection_reason
        elif raw_message:
            process_action_details = raw_message
            fallback_count += 1
        elif prebuilt_embedding_text:
            process_action_details = prebuilt_embedding_text
            fallback_count += 1
        else:
            # 三者皆缺失 → invalid
            invalid_log_count += 1
            continue

        # ── 6. 可缺失欄位映射（缺失補 "unknown"）──────────────────
        #   event_id   ← normalized_event.event_type（v2 語意替代）
        #   source_ip  ← normalized_event.source_ip
        #   dest_ip    ← normalized_event.dest_ip
        #   actor_user ← metadata.user

        event_id = _nonempty(normalized_event.get("event_type"))
        if not event_id:
            event_id = "unknown"
            replacement_counts["event_id"] += 1

        source_ip = _nonempty(normalized_event.get("source_ip"))
        if not source_ip:
            source_ip = "unknown"
            replacement_counts["source_ip"] += 1

        dest_ip = _nonempty(normalized_event.get("dest_ip"))
        if not dest_ip:
            dest_ip = "unknown"
            replacement_counts["dest_ip"] += 1

        actor_user = _nonempty(metadata.get("user"))
        if not actor_user:
            actor_user = "unknown"
            replacement_counts["actor_user"] += 1

        # ── 7. 組裝整平後的 validated_log ─────────────────────────
        validated_log = {
            # v1 相容欄位
            "log_id":                  log_id,
            "timestamp":               timestamp,
            "event_id":                event_id,
            "source_ip":               source_ip,
            "dest_ip":                 dest_ip,
            "actor_user":              actor_user,
            "process_action_details":  process_action_details,
            # v2 擴充欄位（供 text_builder v2 template 使用）
            "prebuilt_embedding_text": prebuilt_embedding_text,
            "raw_message":             raw_message,
            # 網路上下文（供 text_builder v2 動態組裝）
            "conn_state":              _nonempty(normalized_event.get("conn_state")),
            "protocol":                _nonempty(normalized_event.get("protocol")),
            "dest_port":               _nonempty(normalized_event.get("dest_port")),
            "source_port":             _nonempty(normalized_event.get("source_port")),
            "service":                 _nonempty(normalized_event.get("service")),
        }

        valid_logs.append(validated_log)

    input_log_count = len(raw_logs)
    valid_log_count = len(valid_logs)

    # Invariant 檢查
    if input_log_count != (valid_log_count + invalid_log_count):
        raise RuntimeError(
            "Invariant violated: input_log_count != valid_log_count + invalid_log_count"
        )

    summary = {
        "input_summary": {
            "input_log_count":   input_log_count,
            "valid_log_count":   valid_log_count,
            "invalid_log_count": invalid_log_count,
        },
        "validation_details": {
            "fallback_to_raw_message_count": fallback_count,
            "field_replacements":            replacement_counts,
        },
    }

    return valid_logs, summary


# ── 內部工具 ───────────────────────────────────────────────────────────────

def _nonempty(value) -> str | None:
    """回傳非空字串值；None / 空字串 / 非字串型別一律回傳 None。"""
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() if value.strip() else None
    # 非字串（如 int/bool）轉字串後使用
    return str(value).strip() or None
