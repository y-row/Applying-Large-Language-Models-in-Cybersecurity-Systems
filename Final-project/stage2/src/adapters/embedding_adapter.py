import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingAdapter:
    def __init__(self, config: dict):
        """
        初始化 Embedding 轉接器。

        config 範例:
        {
            "embedding": {
                "provider": "sentence_transformers",
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "normalize_embeddings": true,
                "batch_size": 256
            }
        }
        """
        self.emb_config = config.get("embedding", {})
        self.provider = self.emb_config.get("provider", "sentence_transformers")
        self.model_name = self.emb_config.get(
            "model_name",
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.normalize = self.emb_config.get("normalize_embeddings", True)
        self.batch_size = self.emb_config.get("batch_size", 256)

        # 延遲載入
        self.model = None

    def encode(self, texts: list) -> tuple:
        """
        將文本列表編碼為向量。

        前置條件：
            texts 為 feature_texts（已完成清洗與語意建構）

        Returns:
            (embeddings, embedding_summary)
        """

        if not isinstance(texts, list):
            raise ValueError("texts 必須為 list")

        # 空輸入處理
        if len(texts) == 0:
            return np.empty((0, 0)), {
                "provider": self.provider,
                "model_name": self.model_name,
                "embedding_dimension": 0,
                "normalized": self.normalize
            }

        # lazy load（動態分配執行裝置）
        if self.model is None:
            if self.provider == "sentence_transformers":
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model = SentenceTransformer(self.model_name, device=device)
                print(f"[INFO] EmbeddingAdapter 偵測硬體裝置，指定執行裝置為: {device}")
            else:
                raise ValueError(f"未支援的 provider: {self.provider}")

        # 啟用進度條以提升巨量資料編碼期之觀測性
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=True
        )

        # Invariant Check
        if embeddings.shape[0] != len(texts):
            raise RuntimeError("Invariant violated: embedding size mismatch")

        summary = {
            "provider": self.provider,
            "model_name": self.model_name,
            "embedding_dimension": embeddings.shape[1],
            "normalized": self.normalize
        }

        return embeddings, summary