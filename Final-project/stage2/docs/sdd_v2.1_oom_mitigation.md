# 系統變更設計文件 (SDD Extension)：feat_v2.1_OOM_solution.md

## 1. 變更背景與問題量化 (Change Background & Problem Quantization)

本文件定義 Phase 2 Pipeline 在處理大規模資安日誌（數據量達 $1,533,064$ 筆）時的架構變更規格，旨在消除現行設計中的記憶體溢出（Out of Memory, OOM）風險與效能卡死問題。現行組件之技術瓶頸量化分析如下：

* **運算裝置模糊性**：現行 `EmbeddingAdapter` 未明確指定底層模型載入之硬體裝置（`device` 參數），導致在特定生產環境下回退至 CPU 執行。此外，因關閉進度觀測指標（`show_progress_bar=False`），致使巨量資料編碼時主流程呈現定格狀態，缺乏監控度。
* **分群空間複雜度爆炸**：現行 `ClusteringAdapter` 採用 `scikit-learn` 原生 `DBSCAN`。當配置參數 `metric="cosine"` 時，該實作無法啟用空間索引樹（如 KD-Tree 或 Ball-Tree）進行剪枝加速，其空間複雜度呈 Worst-case $O(n^2)$。
* **極限記憶體開銷估算**：在樣本數 $n = 1,533,064$ 下，若建構完整成對距離矩陣（Pairwise Distance Matrix），理論開銷量化如下：
* `float64` 精度：$(1,533,064)^2 \times 8 \text{ Bytes} \approx 18.80 \text{ TB}$ (約 $17.10 \text{ TiB}$)
* `float32` 精度：$(1,533,064)^2 \times 4 \text{ Bytes} \approx 9.40 \text{ TB}$ (約 $8.55 \text{ TiB}$)


此幾何級數增長之空間需求必將觸發作業系統之強制終止機制（OOM Killer）。必須重構組件行為規格，嚴禁在百萬級規模下採用原生 CPU DBSCAN 計算路徑。

---

## 2. 向量化轉接器規格變更 (src/adapters/embedding_adapter.py)

### 2.1 介面與行為規範

`EmbeddingAdapter` 必須在維持既有外部調用介面不變的前提下，優化內部延遲載入（Lazy Load）與編碼執行期之行為：

1. **動態硬體執行裝置分流**：
* 在初始化或模型延遲載入階段，必須引入硬體加速偵測機制。
* 若環境支援 CUDA 驅動及硬體，模型載入裝置參數必須明確指定為 `"cuda"`。
* 若 CUDA 環境不可用，則安全降級（Fallback）至 `"cpu"` 執行。


2. **進度條觀測性啟用**：
* `encode()` 方法內部呼叫底層模型時，參數 `show_progress_bar` 必須變更為 `True`。
* 確保標準輸出（stdout）能即時接收串流編碼之百分比與時程預估，提供外部監控指標。



### 2.2 內部控制邏輯變更規格

| 執行階段 | 現行行為 | 變更後標準規格 |
| --- | --- | --- |
| **模型初始化 (Lazy Load)** | 未指定 `device` 參數，依賴庫預設行為。 | 檢驗 `torch.cuda.is_available()`，動態賦予 `"cuda"` 或 `"cpu"` 至模型載入參數。 |
| **編碼執行 (Encode)** | `show_progress_bar=False` | `show_progress_bar=True` |

---

## 3. 分群轉接器規格變更 (src/adapters/clustering_adapter.py)

### 3.1 核心加速技術與內存控制

為適配限制型硬體環境（目標環境：RTX 3060 12GB VRAM），分群核心必須引入 NVIDIA RAPIDS 體系之 `cuML` 加速庫，替換原生 CPU 運算組件。變更規格如下：

1. **記憶體分塊控制限制**：
* 引入顯存分塊控制參數 `max_mbytes_per_batch`，其數值必須固定配置為 `4096`（即 4GB）。
* 此參數用於限制單次 pairwise distance 批次計算之記憶體配額，降低 RTX 3060 12GB VRAM 環境下的 OOM 風險；但不構成整體演算法總顯存使用量上限，最終可行性仍需依實測基準確認。


2. **輸出數據類型相容性**：
* 配置 `output_type="numpy"`，確保加速組件回傳之標籤矩陣型態與現有下游數據流無縫相容。



### 3.1.1 顯存控制限制聲明

`max_mbytes_per_batch=4096` 僅限制 `cuML DBSCAN` 於 pairwise distance computation 階段的批次記憶體目標，不代表整體演算法總顯存使用量上限。由於 DBSCAN 仍需依據 `eps`、資料局部密度與鄰接圖規模產生額外中間資料結構，因此本參數僅定義為 OOM 風險降低機制，不得描述為 12GB VRAM 環境下的完成保證。

### 3.2 巨量資料 Fail-Fast 安全熔斷機制

嚴禁盲目容許高負載資料直接回退至 `sklearn.cluster.DBSCAN`。系統必須依據輸入數據規模（$n = \text{embeddings.shape}[0]$）實施雙軌分流控制與安全熔斷策略：

* **低資料量級軌道（$n \le 10,000$）**：
* 若系統未安裝 `cuML` 套件，允許 Fallback 至 `sklearn.cluster.DBSCAN`。
* 必須顯式配置 `n_jobs=-1` 以啟用全核心多執行緒加速，維持非 GPU 環境下的基礎交叉驗證能力。


* **高資料量級軌道（$n > 10,000$）**：
* 系統執行前置檢查，若環境未成功配置 `cuML` 加速套件，**必須立即拋出 `RuntimeError` 終止當前 Pipeline 流程**。
* 實施安全熔斷，嚴禁進入 CPU 巨量成對距離計算，防止系統發生永久性卡死。



$n\_samples \le 10,000$ 為本專案保守工程熔斷閾值，用於限制 `sklearn fallback` 的最大輸入規模；該數值非 `scikit-learn` 官方保證之記憶體安全邊界，亦非 DBSCAN 演算法之理論上限。

### 3.3 分群執行期控制流規格

```
[輸入: embeddings (n_samples)]
       │
       ├─► 若 n_samples == 0 ──► 回傳空陣列
       │
       └─► 檢查環境是否存在 cuML 模組 (HAS_CUML)
             │
             ├──► [TRUE] ──► 實例化 cuDBSCAN
             │               配置: max_mbytes_per_batch=4096, output_type="numpy"
             │               執行 fit_predict() ──► 輸出結果
             │
             └──► [FALSE] ──► 檢查資料量級
                                │
                                ├──► 若 n_samples <= 10000 ──► Fallback 至 sklearn DBSCAN
                                │                             (工程熔斷閾值，配置: n_jobs=-1)
                                │                             執行 fit_predict() ──► 輸出結果
                                │
                                └──► 若 n_samples > 10000 ──► 觸發安全熔斷
                                                              拋出 RuntimeError (拒絕執行並中斷)

```

---

## 4. 環境相依性與部署規格 (Environment & Deployment Specs)

### 4.1 部署環境依賴約束

部署命令應以 RAPIDS 官方 Release Selector 依實際 CUDA Driver、CUDA Runtime、Python 版本與 WSL2/Linux 發行版產生之結果為準。本文列出之 conda / pip 命令僅作為 CUDA 12.x + Python 3.10 基準環境之已知相容範例，不作為跨環境固定安裝命令，不應硬編碼為永久約束。環境配置腳本中嚴禁採用泛用型 `pip install cuml` 命令。

* **相容環境範例 A（Conda 渠道安裝）**：
```bash
conda install -c rapidsai -c conda-forge -c nvidia cuml=24.04 python=3.10 cuda-version=12.2

```


* **相容環境範例 B（NVIDIA PyPI 渠道安裝）**：
```bash
pip install --extra-index-url https://pypi.nvidia.com cuml-cu12==24.4.*

```



### 4.2 品質與效能矩陣判定

* **架構非邊緣性判定**：事件分群階段位於整個 Pipeline 資料流的核心環節（Embedding $\to$ Clustering $\to$ Evaluator），此優化規格直接關係到後續評估報告與格式化輸出的完整性，定義為系統核心變更。
* **執行時程不確定性標記**：[不確定] <由於未實際在 RTX 3060 12GB 上執行 153 萬筆資料的 cuML DBSCAN 完整測試，最終執行時間受 eps 與資料局部密度影響，無法給出確定的耗時結論，需依賴實測基準。>


## 5. v2.1 problem report 與解決方案規格

```log
python phase2_embedding_clustering.py
[INFO] Pipeline 開始執行...
[INFO] 成功載入原始日誌，共計 1533064 筆。
[INFO] 驗證完成。有效日誌: 1533064 筆, 無效日誌: 0 筆。
[INFO] 特徵文本生成完畢。截斷次數統計: 0 筆。
[INFO] 開始初始化 Embedding 模型並計算向量...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|██████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 2134.31it/s]
[INFO] EmbeddingAdapter 偵測硬體裝置，指定執行裝置為: cuda
Batches: 100%|██████████████████████████████████████████████████████████████████████| 5989/5989 [31:17<00:00,  3.19it/s]
Killed
```

### 5.1 事實與肇因推論

* **已知事實**：全量日誌（1,533,064 筆）完成 Embedding 階段（耗時 31 分鐘），於進入分群階段時觸發 Linux OOM Killer 程序終止。
* **系統依賴**：下游 Pipeline (Phase 3) 依賴 DBSCAN 產出之 `-1` (Noise Label) 進行孤立事件過濾與降噪。
* **專案約束**：要求最小改動、最大相容性，不可破壞 Pipeline 資料流與外部格式。
* **合理推論 (複雜度與分片影響)**：原生 DBSCAN 演算法空間複雜度呈二次方增長，為百萬級資料觸發 OOM 之高度疑似肇因。外部資料物理分片（Sharding）預期將切斷日誌之全局上下文關聯，增加攻擊時間線還原破碎化之風險。
* **[不確定] <原因與影響：多重記憶體開銷>**：分群階段的鄰域搜尋或圖狀中間結構為主要 OOM 來源的推論合理；但在未取得系統記憶體監控、GPU/CPU memory trace 與 `cuML` 內部配置紀錄前，仍不應排除其他後處理或資料轉換造成的記憶體峰值。

### 5.2 核心修復策略

* **修復策略 A (導入磁碟快取)**：[OK] 持久化 Embedding 中間產物，將向量化結果視為可復用產物，避免失敗重跑時重複消耗 31 分鐘級別之運算時間。
* **修復策略 B (高載演算法替換)**：當資料量超過配置門檻時，改採記憶體需求較可控的 MiniBatchKMeans。此為確保 Pipeline 資源穩定性之優先考量，但分群語意不視為與 DBSCAN 完全等價。

### 5.3 建議方案：Inductive DBSCAN (歸納式密度分群)

此理論設計目標為嘗試將空間複雜度自 $O(N^2)$ 降至 $O(M^2) + O(N \log M)$（其中 $M$ 為抽樣數），預期可保留 `-1` 降噪特性、維持全局關聯，並對外部組件介面透明。

1. **代表性核心抽樣**：自全量輸入矩陣中隨機抽取 $M$ 筆樣本（初始建議基準為 50,000 筆）作為密度分群之基準空間。
2. **基準密度分群**：僅對該 $M$ 筆核心樣本執行 `cuML DBSCAN`，產出包含有效群集與 `-1` 標籤之核心模型，將運算之記憶體峰值與全量 $N$ 脫鉤。
3. **全量空間映射 (KNN Projection)**：運用 K-Nearest Neighbors ($K=1$) 空間索引演算法，將剩餘之 $N-M$ 筆全量資料映射至距離最近之核心樣本點。
4. **邊界約束與合約對齊**：若全量樣本至最近核心點之距離 $\le \text{eps}$，則繼承該核心點之 `cluster_id`；若距離 $> \text{eps}$，則標記為 `-1`。最終重新組合長度 $N$ 的標籤陣列回傳。

* **[不確定] <記憶體峰值控制成效>**：雖然理論上可限縮運算規模，但在未於目標環境進行壓力測試前，無法保證絕對迴避 OOM。
* **[不確定] <微型群集遺失風險>**：隨機抽樣機制可能遺漏資料佔比極低之微型異常事件（規模小於抽樣覆蓋率），導致該類群集在映射階段被誤判為 `-1` 噪訊，影響程度無法於實測前量化。

### 5.4 實作修改清單與相容性評估

* **`config/phase2_config.json`**：新增 `embedding_checkpoint_path` 指定快取儲存路徑，並新增 `clustering_strategy` 節點配置 `dbscan_max_samples` 動態安全門檻與 fallback 參數。
* 在 Windows 的使用者目錄（C:\Users\<YourUsername>\）下，編輯 .wslconfig 檔案，手動提高 WSL2 的記憶體與 Swap 上限，藉此扛過矩陣合併的峰值。(依硬體規格調整)

```
[wsl2]
memory=24GB
swap=32GB
processors=24
```

* **`src/adapters/clustering_adapter.py`**：實施 $n \le \text{dbscan\_max\_samples}$ 與 $n > \text{dbscan\_max\_samples}$ 之雙軌控制邏輯，強制配置回傳格式為 `numpy.ndarray` 以確保資料介面一致性。
* **`phase2_embedding_clustering.py`**：於主流程加入快取命中（略過編碼）與未命中（執行編碼並持久化）之判定邏輯。
* **相容性預期**：`ClusteringAdapter` 之 I/O 合約維持不變，預期後端 `evaluator.py` 與 `output_formatter.py` 無需修改。

### 5.5 替代方向與恢復機制

前置假設條件：若實測後發現抽樣導致關鍵攻擊事件被判定為噪訊（覆蓋率過低），或 KNN 映射階段遭遇資源瓶頸，執行以下兩級替代方案。

1. **一級替代 (調整 Inductive DBSCAN 內部策略)**：提高抽樣數 $M$、固定隨機種子 (Random Seed)、改用時間窗覆蓋抽樣或分層抽樣 (Stratified Sampling)，以改善語意覆蓋率。
2. **二級替代 (退回外部物理分片 Sharding)**：若上述調整仍無法通過目標環境之資源限制，則採用物理分片策略，並明確接受跨分片事件無法聚合、攻擊時間線被硬性切斷之業務折損。