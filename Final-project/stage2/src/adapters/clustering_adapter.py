import numpy as np

try:
    from cuml.cluster import DBSCAN as cuDBSCAN
    HAS_CUML = True
except ImportError:
    from sklearn.cluster import DBSCAN
    HAS_CUML = False


class ClusteringAdapter:
    def __init__(self, config: dict):
        self.clu_config = config.get("clustering", {})

        self.algorithm = self.clu_config.get("algorithm", "DBSCAN")
        self.metric = self.clu_config.get("metric", "cosine")
        self.eps = self.clu_config.get("eps", 0.25)
        self.min_samples = self.clu_config.get("min_samples", 3)

        self.model = None

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        前置條件：
            embeddings shape = (n_samples, embedding_dim)
        """

        if not isinstance(embeddings, np.ndarray):
            raise ValueError("embeddings 必須為 numpy.ndarray")

        n_samples = embeddings.shape[0]

        # 空輸入處理
        if n_samples == 0:
            return np.array([], dtype=int)

        if self.model is None:
            if self.algorithm == "DBSCAN":
                if HAS_CUML:
                    # 引入顯存分塊控制參數，降低 OOM 風險
                    self.model = cuDBSCAN(
                        metric=self.metric,
                        eps=self.eps,
                        min_samples=self.min_samples,
                        max_mbytes_per_batch=4096,
                        output_type="numpy"
                    )
                else:
                    # 高資料量級軌道：未配置 cuML 時實施 Fail-Fast 安全熔斷
                    if n_samples > 10000:
                        raise RuntimeError(
                            f"Execution halted: Dataset size {n_samples} exceeds safety threshold (10,000) "
                            "and cuML accelerator is missing. Terminating to prevent system OOM."
                        )
                    # 低資料量級軌道：允許 Fallback 至 sklearn 並啟用全核心多執行緒
                    self.model = DBSCAN(
                        metric=self.metric,
                        eps=self.eps,
                        min_samples=self.min_samples,
                        n_jobs=-1
                    )
            else:
                raise ValueError(f"未支援的分群演算法: {self.algorithm}")

        labels = self.model.fit_predict(embeddings)

        # Invariant Check
        if len(labels) != n_samples:
            raise RuntimeError("Invariant violated: clustering result size mismatch")

        return labels