import numpy as np
# 修正：匯入 sklearn 官方正式函式
from sklearn.metrics import silhouette_score, davies_bouldin_score

def evaluate_clusters(embeddings: np.ndarray, labels: np.ndarray, validation_summary: dict, config: dict) -> dict:
    """
    執行幾何分群品質評估與統計摘要建構。
    
    Args:
        embeddings: np.ndarray, 向量矩陣
        labels: np.ndarray, 分群標籤陣列
        validation_summary: dict, 由 log_validator 產出的統計摘要
        config: dict, 全域設定字典
        
    Returns:
        dict: 符合 cluster_evaluation.json 規範的完整字典結構
    """
    clu_config = config.get("clustering", {})
    algorithm = clu_config.get("algorithm", "DBSCAN")
    metric = clu_config.get("metric", "cosine")
    eps = clu_config.get("eps", 0.25)
    min_samples = clu_config.get("min_samples", 3)

    # 1. 計算基本分群統計
    unique_labels = set(labels)
    valid_labels = [label for label in unique_labels if label != -1]
    number_of_clusters = len(valid_labels)
    
    noise_count = int(np.sum(labels == -1))
    valid_log_count = len(labels)
    noise_ratio = round(noise_count / valid_log_count, 4) if valid_log_count > 0 else 0.0

    # 2. 計算群集大小分佈與標籤標準化格式映射
    cluster_size_distribution = {}
    for label in unique_labels:
        count = int(np.sum(labels == label))
        if label == -1:
            cluster_size_distribution["noise"] = count
        else:
            cluster_size_distribution[f"cluster_{label:03d}"] = count

    # 3. 核心防護機制：判斷有效群集數量是否小於 2
    sil_score = None
    dbi_score = None
    evaluation_notes = []

    # 排除 noise 後的樣本與標籤，用於幾何指標計算
    non_noise_mask = (labels != -1)
    filtered_embeddings = embeddings[non_noise_mask]
    filtered_labels = labels[non_noise_mask]
    filtered_unique_labels = set(filtered_labels)

    if number_of_clusters < 2 or len(filtered_unique_labels) < 2:
        # 當有效群集數小於 2，幾何指標失去數學意義，強制輸出 null 且不崩潰
        evaluation_notes.append("Less than two valid clusters after excluding noise label -1.")
    else:
        try:
            # 正常情境下計算幾何指標 (DBSCAN 預設使用 cosine 距離)
            sil_score = float(silhouette_score(filtered_embeddings, filtered_labels, metric=metric))
            dbi_score = float(davies_bouldin_score(filtered_embeddings, filtered_labels))
        except Exception as e:
            evaluation_notes.append(f"Geometry metric calculation failed: {str(e)}")

    # 4. 警示邊界檢查 (SDD 4.2 節 僅產生 warning 不中斷流程)
    if noise_ratio > 0.70:
        evaluation_notes.append("[WARNING] noise_ratio > 0.70 (eps may be too small or text lacks separability)")
    if noise_ratio < 0.05 and noise_count > 0:
        evaluation_notes.append("[WARNING] noise_ratio < 0.05 (eps may be too large, mixing benign and malicious logs)")

    # 5. 組裝最終 JSON 輸出結構
    evaluation_report = {
        "input_summary": validation_summary["input_summary"],
        "clustering_summary": {
            "algorithm": algorithm,
            "metric": metric,
            "eps": eps,
            "min_samples": min_samples,
            "number_of_clusters": number_of_clusters,
            "noise_count": noise_count,
            "noise_ratio": noise_ratio,
            "cluster_size_distribution": cluster_size_distribution
        },
        "quality_metrics": {
            "silhouette_score": sil_score,
            "davies_bouldin_index": dbi_score
        },
        "evaluation_notes": evaluation_notes
    }

    return evaluation_report