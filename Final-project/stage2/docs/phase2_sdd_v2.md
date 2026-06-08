# 軟體設計文件 (SDD) v2: Phase 2 系統升級規格

## 1. 核心升級目標與 MVP 定義

本階段任務旨在將系統由 Phase 1 之輸出與測試環境，全面升級並對接至 Phase 2 之真實網路遙測日誌（Real Data）。

* **MVP 目標 (Minimum Viable Product)**：維持與 v1 完全相容之下游輸出結構（Match the v1），確保無縫對接下一階段模組，並精確提取所需之核心數據。
* **資料格式與來源轉換**：v1 階段使用 `.json` 格式與模擬數據（Mock Data）。v2 階段需全面遷移至支援大數據循序讀取之 `.jsonl`（JSON Lines）格式，並導入包含高密度資訊之真實日誌（Real Data including much more info）。
* **結構適配與特徵強化**：真實日誌具備多層次嵌套結構（Nested JSON），必須於資料輸入層執行結構整平。同時，必須更新 `text_builder.py` 之特徵萃取邏輯，以獲取並整合更佳的上下文資訊（Gain better information）。

## 2. 系統輸入規格變更與真實數據樣本

系統輸入資料變更為高維度之真實網路日誌。資料集提取與驗證基準如下：

**真實資料提取範例 (PowerShell)**：

```bash
powershell -Command "Get-Content -Path "phase2_project\data\stage2_input.jsonl" -TotalCount 5"

```

提取結果作為 `example.jsonl`，供先期管線驗證使用。必須先於 `example.jsonl` 上運行以確認解析與邏輯正確無誤。

**單筆 JSONL 結構樣本**：

```json
{"dataset": "uwf_zeekdata22_csv_all_no_mitre", "event_id": null, "event_type": "suspicious_network_connection", "event_uid": "E000001", "host": null, "metadata": {"dataset": "uwf_zeekdata22_csv_all_no_mitre", "event_id": null, "event_type": "suspicious_network_connection", "event_uid": "E000001", "host": null, "scenario": null, "source_file": "data/uwf-zeekdata22/uwf_zeekdata22_csv_all_no_mitre.jsonl", "source_line": 257, "timestamp": "2022-02-10T03:58:29.982Z", "user": null}, "normalized_event": {"command_line": null, "conn_state": "S0", "dataset": "uwf_zeekdata22_csv_all_no_mitre", "dest_ip": "143.88.5.1", "dest_port": "53", "duration": "5.004887819290161", "event_id": null, "event_type": "suspicious_network_connection", "event_uid": "E000001", "history": "D", "host": null, "orig_bytes": "78", "orig_pkts": "2", "parent_command_line": null, "parent_process_id": null, "parent_process_name": null, "process_id": null, "process_name": null, "process_path": null, "protocol": "udp", "raw_message": "{\"resp_pkts\":\"0\",\"service\":\"dns\",\"orig_ip_bytes\":\"134\",\"local_resp\":\"false\",\"missed_bytes\":\"0\",\"protocol\":\"udp\",\"duration\":\"5.004887819290161\",\"conn_state\":\"S0\",\"dest_ip\":\"143.88.5.1\",\"orig_pkts\":\"2\",\"community_id\":\"1:z6/qFXDqWjH8nl4GVJt26qkOLkw=\",\"resp_ip_bytes\":\"0\",\"dest_port\":\"53\",\"orig_bytes\":\"78\",\"local_orig\":\"false\",\"datetime\":\"2022-02-10T03:58:29.982Z\",\"history\":\"D\",\"resp_bytes\":\"0\",\"uid\":\"CNHONp4bmwTvWFmFh\",\"src_port\":\"44428\",\"ts\":\"1.644465509982958E9\",\"src_ip\":\"143.88.5.12\",\"source_csv_file\":\"part-00000-0af89d10-df53-44fd-b124-a8a496fd5023-c000.csv\",\"source_csv_line\":258}", "resp_bytes": "0", "resp_pkts": "0", "scenario": null, "selection_reason": "Suspicious network pattern identified: conn_state_S0", "service": "dns", "source_event_id": "CNHONp4bmwTvWFmFh", "source_file": "data/uwf-zeekdata22/uwf_zeekdata22_csv_all_no_mitre.jsonl", "source_ip": "143.88.5.12", "source_line": 257, "source_port": "44428", "suspicion_category": "conn_state_S0", "text_for_embedding": "At 2022-02-10T03:58:29.982Z | dataset=uwf_zeekdata22_csv_all_no_mitre | scenario=None | host=None | user=None | event_type=suspicious_network_connection | event_id=None | source_event_id=CNHONp4bmwTvWFmFh | process=None | command=None | parent=None | source=143.88.5.12:44428 | dest=143.88.5.1:53 | protocol=udp | conn_state=S0 | tactic=None | evidence=E000001@data/uwf-zeekdata22/uwf_zeekdata22_csv_all_no_mitre.jsonl:257", "timestamp": "2022-02-10T03:58:29.982Z", "user": null}, "scenario": null, "source_file": "data/uwf-zeekdata22/uwf_zeekdata22_csv_all_no_mitre.jsonl", "source_line": 257, "text_for_embedding": "At 2022-02-10T03:58:29.982Z | dataset=uwf_zeekdata22_csv_all_no_mitre | scenario=None | host=None | user=None | event_type=suspicious_network_connection | event_id=None | source_event_id=CNHONp4bmwTvWFmFh | process=None | command=None | parent=None | source=143.88.5.12:44428 | dest=143.88.5.1:53 | protocol=udp | conn_state=S0 | tactic=None | evidence=E000001@data/uwf-zeekdata22/uwf_zeekdata22_csv_all_no_mitre.jsonl:257", "timestamp": "2022-02-10T03:58:29.982Z", "user": null}

```

依據上述樣本，核心節點定義如下：

* **全域節點**：`dataset`, `timestamp`, `event_type`。
* **`metadata` 節點**：包含事件全域背景屬性（如 `event_uid`, `user`, `scenario`）。
* **`normalized_event` 節點**：包含網路行為特徵（如 `source_ip`, `dest_ip`, `dest_port`, `conn_state`, `protocol`, `selection_reason`）與底層設備原始日誌（`raw_message`）。
* **`text_for_embedding`**：系統端預先組合之高維度上下文特徵字串。

## 3. 核心組件架構與配置更新

延續 v1 之**配置驅動（Config-driven）**與**轉接器模式（Adapter Pattern）**架構，資料流順序保持不變。為支援 v2 升級，全域設定檔 `phase2_config.json` 必須進行更新，且**必須嚴格保留 v1 既有之 `embedding`, `clustering`, `output` 配置區塊**：

* `input_path`：更改為指向 JSONL 檔案（如 `stage2_input.jsonl` 或 `example.jsonl`）。
* `feature_template_version`：指定為 `"v2"`，以觸發新版特徵萃取邏輯。
* 保留約束：不得移除或修改 `embedding`, `clustering`, `output` 區塊內的鍵值結構，確保下游行為與 v1 一致。

## 4. 資料映射與正規化約束

為維持資料流管線之相容性，輸入之 v2 多層次節點必須透過 `log_validator.py` 映射至系統內部之扁平欄位。完整映射規則與容錯優先序如下：

| 內部欄位 (v1 相容目標) | 來源節點 (v2 JSONL) | 提取與處理策略 |
| --- | --- | --- |
| `log_id` | 多重來源 | 優先序：1. `metadata.event_uid` 2. `normalized_event.event_uid` 3. `event_uid` 4. `normalized_event.source_event_id`。若皆缺失則剔除日誌。 |
| `timestamp` | 全域 `timestamp` | 提取全域時間戳；缺失則剔除日誌。 |
| `event_id` | `normalized_event.event_type` | 映射事件類型（註：此為 v1 `event_id` 之 phase 2 real-data 語意替代方案）；缺失則補 `"unknown"`。 |
| `source_ip` / `dest_ip` | `normalized_event.source_ip` / `dest_ip` | 提取實體 IP；缺失則補 `"unknown"`。 |
| `actor_user` | `metadata.user` | 提取使用者名稱；缺失則補 `"unknown"`。 |
| `process_action_details` | `normalized_event.selection_reason` | 優先取觸發原因。若缺失，進入嚴格 fallback 流程（詳見 5.2 節）。 |
| `raw_message` | `normalized_event.raw_message` | 選填；僅作為 `process_action_details` 與 `text_for_embedding` 缺失時之降級 fallback 來源。 |
| `prebuilt_embedding_text` | `text_for_embedding` | **[新增擴充]** 保留原始內建字串，供特徵工程層直接取用。 |

## 5. 模組重構範圍與規格 (Impact Scope)

為達成上述映射並維持資料管線之穩定性，以下核心模組必須依循新規格重構：

### 5.1 `log_loader.py` (資料讀取層)

* **重構目標**：支援 JSON Lines 格式。
* **規格要求**：
* 實作逐行讀取（Line-by-line parsing）與單行 JSON 序列化。
* 導入空行忽略與 JSONDecodeError 之例外容錯機制，確保部分損毀資料不中斷整體讀取流程。



### 5.2 `log_validator.py` (資料清洗與檢驗層)

* **重構目標**：實作多層次結構解構與扁平化映射。
* **規格要求**：
* 導入對 `metadata` 與 `normalized_event` 節點之安全提取（Safe Get）邏輯。
* 嚴格執行「第 4 節」定義之 `log_id` 多重來源提取與 `"unknown"` 補值。
* **[關鍵閉環] `process_action_details` 缺失與無效判定**：若 `selection_reason`、`raw_message` 與 `text_for_embedding` 三者皆缺失或為空字串，該筆日誌強制標記為 invalid 並剔除。



### 5.3 `text_builder.py` (特徵工程層)

* **重構目標**：擴充特徵組裝邏輯，新增 `template_version="v2"`。
* **規格要求**：當執行模式為 `"v2"` 時，特徵字串生成必須遵循以下優先序：
1. 若 `prebuilt_embedding_text` 存在且非空，直接使用。
2. 若缺失，使用 `process_action_details` 搭配網路上下文（如 `conn_state`, `protocol`）動態組裝。
3. 若 `process_action_details` 缺失，回退使用 `raw_message`。
4. 若仍無任何語意文本，將拋出例外或標記異常（依賴 5.2 節的 invalid 剔除機制防護）。


* 保留既有之長度估算與 256 Token 截斷控制邏輯。



### 5.4 `output_formatter.py` (不可修改，僅宣告相容限制)

* **規範重申**：Formatter 行為必須完全沿用 v1 規格，嚴禁因升級 v2 而變更：
* `include_noise` 預設為 `false`，不輸出標籤為 `-1` 的群集。
* `representative_logs` 必須依 `timestamp` 遞增排序。
* 擷取數量固定為前 N 筆，N 由 `config.output.representative_log_count` 控制。
* `representative_logs` 輸出結構（`log_id`, `timestamp`, `text`）不得變更。



## 6. 輸出規格約束 (Output Specification)

確保 Phase 3 可無縫接收資料，輸出之 JSON 結構必須與 v1 保持絕對一致：

* **`clustered_events.json`**：保留 `cluster_id`、`total_count`、`time_range`、`representative_logs`、`involved_entities` 與 `metadata`。
* **`cluster_evaluation.json`**：保留 `input_summary`、`clustering_summary`、`quality_metrics` 與 `evaluation_notes`。有效群集數小於 2 時輸出 `null` 的防護機制維持運作。

## 7. 開發驗證方法與硬性約束 (Verification & Invariants)

升級至 v2 後，系統必須持續滿足以下硬性不變量與邊界條件，任一條件不滿足視為 Pipeline 執行失敗：

### 7.1 硬性不變量約束 (Rigid Invariants)

1. `input_log_count == valid_log_count + invalid_log_count`
2. `len(feature_texts) == valid_log_count`
3. `embeddings.shape[0] == len(cluster_labels) == valid_log_count`

### 7.2 警示與邊界機制 (Warnings)

下列條件僅觸發警告並寫入 `evaluation_notes`，不中斷主流程：

* `noise_ratio > 0.70`
* `noise_ratio < 0.05`
* `number_of_clusters < 2`
* `truncated_count_estimate > 0`

---

# Appendix: Prompt for Engineer (can ignore)

## 任務指派：Phase 2 資料管線升級實作 (v1 to v2)

**【任務目標】**
依據附加之 `phase2_sdd_v2.md` 規格文件，執行 Phase 2 系統升級。將系統資料輸入源從扁平結構之 JSON Array 模擬數據，重構為支援解析高維度、多層次嵌套結構之 JSON Lines (JSONL) 真實網路遙測日誌。

**【實作範圍與限制】**

1. **修改目標**：僅限重構以下三個模組：
* `src/utils/log_loader.py`
* `src/utils/log_validator.py`
* `src/utils/text_builder.py`


2. **鎖定模組**：嚴禁修改 `evaluator.py` 與 `output_formatter.py`，必須依賴前置模組之資料整平（Flattening）確保向下相容。
3. **配置更新**：`phase2_config.json` 僅允許修改 `input_path` 與 `feature_template_version`；`embedding`、`clustering`、`output` 區塊的鍵值結構與既有值均不得變更。

**【核心實作約束】**

* **I/O 容錯機制 (Loader)**：
必須實作逐行讀取（Line-by-line parsing），並加入空行忽略與單行 `JSONDecodeError` 之 `try-except` 容錯，確保單筆資料毀損不中斷整體批次處理。
* **資料映射與缺失防護 (Validator)**：
* **多重 ID 提取**：`log_id` 提取順序必須嚴格遵循：
`metadata.event_uid` -> `normalized_event.event_uid` -> `event_uid` -> `normalized_event.source_event_id`。
若皆缺失或為空字串，該筆日誌標記為 invalid 並剔除。
* **Fallback 閉環**：
`process_action_details` 以 `normalized_event.selection_reason` 為主要來源。
若 `selection_reason` 缺失，不得立即剔除；需保留 `normalized_event.raw_message` 與 `text_for_embedding` 供後續 fallback。
當 `selection_reason`、`normalized_event.raw_message`、`text_for_embedding` 三者皆缺失或為空字串時，該筆日誌強制標記為 invalid 並從有效清單中剔除。
* **預留特徵**：
必須將原始日誌的 `text_for_embedding` 保留並映射為內部欄位 `prebuilt_embedding_text`。
* **raw_message 保留**：
必須將 `normalized_event.raw_message` 映射為內部欄位 `raw_message`，僅作為 fallback 來源，不得預設覆蓋主要語意欄位。


* **特徵組裝邏輯 (Text Builder)**：
必須實作 `v2` 樣板邏輯，執行順序為：
1. 優先使用 `prebuilt_embedding_text`。
2. 若缺失，使用 `process_action_details` 搭配網路上下文（`conn_state`, `protocol` 等）動態組裝。
3. 若 `process_action_details` 也缺失，退回 `raw_message`。

* 保留 256 Token 截斷限制。


**【硬性不變量 (Rigid Invariants)】**
實作過程中，下列防護邊界不得被破壞：

* `input_log_count == valid_log_count + invalid_log_count`
* `len(feature_texts) == valid_log_count`
* `embeddings.shape[0] == len(cluster_labels) == valid_log_count`

**【交付與驗證標準】**

1. 程式碼需先於小樣本測試集（`example.jsonl`）成功執行。
2. 過程無拋出 `KeyError` 或結構解構失敗例外。
3. 輸出之 `clustered_events.json` 結構格式（包含 `representative_logs` 陣列結構與 timestamp 排序規則）與 v1 基準保持絕對一致。
4. 交付修改完成之 `log_loader.py`、`log_validator.py`、`text_builder.py` 原始碼進行 Code Review。

## 提供之檔案

Allowed to modify:
- src/utils/log_loader.py
- src/utils/log_validator.py
- src/utils/text_builder.py

Allowed config changes:
- config/phase2_config.json:
  - input_path -> data/example.jsonl
  - feature_template_version -> v2

Do not modify:
- src/utils/evaluator.py
- src/utils/output_formatter.py
- embedding / clustering / output config blocks

Validation input:
- data/example.jsonl

Reference spec:
- sdd_v2.md