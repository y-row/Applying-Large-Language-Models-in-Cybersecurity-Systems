from collections import defaultdict
import numpy as np

def format_clustered_events(
    valid_logs: list,
    labels: np.ndarray,
    feature_texts: list,
    embedding_summary: dict,
    config: dict
) -> list:
    """
    將 clustering 結果轉換為 clustered_events.json 規格。

    前置條件：
        1. labels 必須為 numpy.ndarray
        2. valid_logs 已由 validate_logs() 保證欄位完整性，不允許再次補值
        3. feature_texts 與 valid_logs 順序一一對齊
    """
    # 1. 型態與硬性不變量檢查 (落實 SDD 4.1 節規範)
    if not isinstance(valid_logs, list):
        raise ValueError("valid_logs 必須為 list")

    if not isinstance(feature_texts, list):
        raise ValueError("feature_texts 必須為 list")

    if not isinstance(labels, np.ndarray):
        raise ValueError("labels 必須為 numpy.ndarray")

    if len(valid_logs) != len(labels):
        raise RuntimeError("Invariant violated: logs and labels size mismatch")

    if len(feature_texts) != len(valid_logs):
        raise RuntimeError("Invariant violated: feature_texts and logs size mismatch")

    output_config = config.get("output", {})
    representative_log_count = output_config.get("representative_log_count", 5)
    include_noise = output_config.get("include_noise", False)

    clustering_config = config.get("clustering", {})
    embedding_model = embedding_summary.get("model_name", "unknown")
    clustering_algorithm = clustering_config.get("algorithm", "DBSCAN")

    # 2. 高效分組機制 (時間複雜度 O(N))
    grouped = defaultdict(list)
    for log, label, text in zip(valid_logs, labels, feature_texts):
        grouped[label].append({
            "log": log,
            "text": text
        })

    clustered_events = []

    # 3. 依標籤排序歷遍，確保輸出 JSON 穩定遞增
    for label in sorted(grouped.keys()):
        if label == -1 and not include_noise:
            continue

        cluster_id = "noise" if label == -1 else f"cluster_{label:03d}"
        combined_items = grouped[label]

        # 4. 確定性排序：嚴格合約存取 timestamp 進行遞增排序
        combined_sorted = sorted(
            combined_items,
            key=lambda item: item["log"]["timestamp"]
        )

        total_count = len(combined_sorted)

        time_range = {
            "start": combined_sorted[0]["log"]["timestamp"],
            "end": combined_sorted[-1]["log"]["timestamp"]
        }

        # 5. 提取前 N 筆作為代表性日誌
        representative_logs = []
        for item in combined_sorted[:representative_log_count]:
            log = item["log"]
            representative_logs.append({
                "log_id": log["log_id"],
                "timestamp": log["timestamp"],
                "text": item["text"]
            })

        # 6. 彙整涉及實體且自動去重 (嚴格合約檢查 "unknown")
        ips = set()
        users = set()
        event_ids = set()

        for item in combined_sorted:
            log = item["log"]
            
            if log["source_ip"] != "unknown":
                ips.add(log["source_ip"])
            if log["dest_ip"] != "unknown":
                ips.add(log["dest_ip"])
            if log["actor_user"] != "unknown":
                users.add(log["actor_user"])
            if log["event_id"] != "unknown":
                event_ids.add(log["event_id"])

        cluster_obj = {
            "cluster_id": cluster_id,
            "total_count": total_count,
            "time_range": time_range,
            "representative_logs": representative_logs,
            "involved_entities": {
                "ips": sorted(list(ips)),
                "users": sorted(list(users)),
                "event_ids": sorted(list(event_ids))
            },
            "metadata": {
                "embedding_model": embedding_model,
                "clustering_algorithm": clustering_algorithm
            }
        }
        clustered_events.append(cluster_obj)

    return clustered_events