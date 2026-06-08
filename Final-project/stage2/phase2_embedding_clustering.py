import json
import os
import sys
import numpy as np

# 匯入自訂工具與轉接器模組
from src.utils.log_loader import load_logs
from src.utils.log_validator import validate_logs
from src.utils.text_builder import build_feature_texts
from src.utils.evaluator import evaluate_clusters
from src.utils.output_formatter import format_clustered_events
from src.adapters.embedding_adapter import EmbeddingAdapter
from src.adapters.clustering_adapter import ClusteringAdapter

def main():
    # 1. 載入設定檔
    config_path = "config/phase2_config.json"
    if not os.path.exists(config_path):
        print(f"[ERROR] 找不到設定檔: {config_path}")
        sys.exit(1)
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # 提取輸入輸出路徑與樣板版本
    input_path = config.get("input_path", "data/normalized_logs_sample.json")
    feature_template_version = config.get("feature_template_version", "v1")
    output_config = config.get("output", {})
    clustered_events_path = output_config.get("clustered_events_path", "output/clustered_events.json")
    cluster_evaluation_path = output_config.get("cluster_evaluation_path", "output/cluster_evaluation.json")

    print("[INFO] Pipeline 開始執行...")

    # 2. 資料載入 (Log Loader)
    try:
        raw_logs = load_logs(input_path)
        # raw_logs = load_logs(input_path)[:9000]  # [暫時性修改] 截斷前 9000 筆
        print(f"[INFO] 成功載入原始日誌，共計 {len(raw_logs)} 筆。")
    except Exception as e:
        print(f"[ERROR] 載入日誌失敗: {str(e)}")
        sys.exit(1)

    # 3. Schema 驗證與清洗 (Schema Validator)
    valid_logs, validation_summary = validate_logs(raw_logs)
    input_summary = validation_summary["input_summary"]
    print(f"[INFO] 驗證完成。有效日誌: {input_summary['valid_log_count']} 筆, 無效日誌: {input_summary['invalid_log_count']} 筆。")

    # 4. 文字特徵轉換 (Feature Text Builder)
    feature_texts, text_builder_summary = build_feature_texts(valid_logs, template_version=feature_template_version)
    print(f"[INFO] 特徵文本生成完畢。截斷次數統計: {text_builder_summary['truncated_count_estimate']} 筆。")

    # 5. 向量化編碼 (Embedding Encoder Adapter) 並引入快取機制
    print("[INFO] 檢查 Embedding 快取狀態...")
    embedding_checkpoint_path = config.get("embedding", {}).get("checkpoint_path")
    
    if embedding_checkpoint_path and os.path.exists(embedding_checkpoint_path):
        print(f"[INFO] 發現快取，載入既有向量矩陣: {embedding_checkpoint_path}")
        embeddings = np.load(embedding_checkpoint_path)
        
        # 重建 summary 以符合後續 Invariant 驗證與 Formatter 合約
        embedding_summary = {
            "model_name": config.get("embedding", {}).get("model_name"),
            "embedding_dimension": embeddings.shape[1],
            "normalized": config.get("embedding", {}).get("normalize_embeddings", True)
        }
        print(f"[INFO] 快取載入完成。矩陣形狀: {embeddings.shape}，維度: {embedding_summary['embedding_dimension']}。")
    else:
        print("[INFO] 開始初始化 Embedding 模型並計算向量...")
        embedding_adapter = EmbeddingAdapter(config)
        embeddings, embedding_summary = embedding_adapter.encode(feature_texts)
        print(f"[INFO] 向量化完成。矩陣形狀: {embeddings.shape}，維度: {embedding_summary['embedding_dimension']}。")
        
        # 執行持久化寫入
        if embedding_checkpoint_path:
            os.makedirs(os.path.dirname(embedding_checkpoint_path), exist_ok=True)
            np.save(embedding_checkpoint_path, embeddings)
            print(f"[INFO] 向量矩陣已持久化儲存至快取: {embedding_checkpoint_path}")

    # 6. 執行分群演算法 (Clustering Engine Adapter)
    # 優化：安全讀取配置，防範 KeyError
    clustering_algorithm = config.get("clustering", {}).get("algorithm", "DBSCAN")
    print(f"[INFO] 開始執行分群演算法 ({clustering_algorithm})...")
    clustering_adapter = ClusteringAdapter(config)
    cluster_labels = clustering_adapter.fit_predict(embeddings)

    # 7. 硬性不變量約束驗證 (Rigid Invariants Verification)
    # Invariant 1: input_log_count = valid_log_count + invalid_log_count
    if input_summary["input_log_count"] != (input_summary["valid_log_count"] + input_summary["invalid_log_count"]):
        print("[FATAL] Invariant 1 失敗: 輸入日誌總數不等於有效與無效日誌數之和。")
        sys.exit(1)

    # Invariant 2: len(feature_texts) = valid_log_count
    if len(feature_texts) != input_summary["valid_log_count"]:
        print("[FATAL] Invariant 2 失敗: 特徵文本數量與有效日誌數量不對齊。")
        sys.exit(1)

    # Invariant 3: embeddings.shape[0] = len(cluster_labels) = valid_log_count
    if embeddings.shape[0] != len(cluster_labels) or len(cluster_labels) != input_summary["valid_log_count"]:
        print("[FATAL] Invariant 3 失敗: 向量矩陣筆數、分群標籤數量與有效日誌數量不對齊。")
        sys.exit(1)

    print("[INFO] 硬性不變量約束全數通過，開始生成評估報告與格式化輸出...")

    # 8. 品質評估 (Cluster Quality Evaluator)
    evaluation_report = evaluate_clusters(embeddings, cluster_labels, validation_summary, config)

    # 優化：落實 SDD 4.2 節，將文本截斷警示寫入評估報告
    if text_builder_summary.get("truncated_count_estimate", 0) > 0:
        evaluation_report["evaluation_notes"].append(
            f"[WARNING] truncated_count_estimate = {text_builder_summary['truncated_count_estimate']} > 0 (feature text may exceed model token limit)"
        )

    # 9. 格式化事件輸出 (Cluster Output Formatter)
    clustered_events = format_clustered_events(valid_logs, cluster_labels, feature_texts, embedding_summary, config)

    # 10. 寫入外部 JSON 檔案
    os.makedirs(os.path.dirname(clustered_events_path), exist_ok=True)
    os.makedirs(os.path.dirname(cluster_evaluation_path), exist_ok=True)

    with open(clustered_events_path, "w", encoding="utf-8") as f:
        json.dump(clustered_events, f, indent=2, ensure_ascii=False)

    with open(cluster_evaluation_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, indent=2, ensure_ascii=False)

    print("[SUCCESS] Pipeline 執行成功。")
    print(f" -> 分群結果已寫入: {clustered_events_path}")
    print(f" -> 評估報告已寫入: {cluster_evaluation_path}")

if __name__ == "__main__":
    main()