# Email Security Triage Agent with n8n Workflow

## 專案簡介

本專案實現了一個以 **Open WebUI + n8n + Ollama** 架構為基礎的電子郵件安全分類代理（Email Security Triage Agent）。該系統旨在對接收的電子郵件進行智能風險評估，判定其是否存在以下風險要素：

- **Phishing 釣魚攻擊**：冒充合法實體以竊取認證資訊
- **Credential Request 認證資訊請求**：主動要求使用者提供登入憑據
- **Suspicious URL 可疑連結**：包含異常或潛在惡意的 URL
- **Urgency 緊迫性提示**：過度強調時間限制或立即行動的必要性
- **Payment Pressure 支付壓力**：要求立即進行金銀交易或轉帳

系統採用**規則型檢測 + LLM 輔助決策 + 保守型合併策略**的設計，在單一 LLM 輔助情境下提供額外的風險管控層。

---

## Workflow 架構

### 整體流程圖

```
Open WebUI (user input)
    ↓
n8n Webhook (receive chatInput)
    ↓
Prepare Cases (parse email / load test cases)
    ↓
Rule-based Detector (heuristic analysis)
    ↓
HTTP Request → Ollama API (qwen2.5:7b LLM analysis)
    ↓
Decision Merger (merge rule_risk & llm_risk)
    ↓
Format Output (prepare response)
    ↓
Respond to Webhook (send result)
    ↓
Open WebUI (display result to user)
```

### 各 Node 職責說明

| Node | 功能 |
|------|------|
| **Prepare Cases** | 從 Webhook 接收使用者輸入；支援測試模式、JSON 格式郵件、原始文本郵件；進行初步資料解析與正規化 |
| **Rule-based Detector** | 基於啟發式規則進行郵件分析；偵測 URL、認證請求、緊迫語氣、支付壓力等信號；產生規則型風險評分 |
| **HTTP Request (Ollama)** | 呼叫本機 Ollama API；使用 qwen2.5:7b 進行 LLM 分析；產生 LLM 型風險評分、推薦行動與分析理由 |
| **Decision Merger** | 解析 LLM 輸出；採用保守策略合併規則型與 LLM 型風險評分；決定最終風險等級與對應行動 |
| **Format Output** | 整理分析結果為 Open WebUI pipe 可讀的格式；支援單封郵件與批量評估模式 |

---

## 檔案說明

### 1. Prepare Cases.js

**功能概述**：n8n JavaScript node，負責接收與解析使用者輸入。

**核心邏輯**：

- **測試模式觸發**：若輸入為 `test` 或 `測試`，加載內置的 6 筆代表性測試案例
  - 每筆案例包含 email text、預期風險標籤（expected_label）、incident_id
  
- **JSON 郵件格式**：若輸入為有效 JSON，則解析以下欄位
  ```json
  {
    "subject": "string",
    "sender": "string",
    "body": "string",
    "urls": ["url1", "url2"],
    "incident_id": "string (optional)"
  }
  ```
  
- **原始文本模式**：若輸入為一般文字，將其視為完整郵件文本進行分析
  - 自動嘗試萃取 subject 與 sender（若格式允許）

**輸出結構**：

```javascript
{
  mode: "test" | "structured" | "raw",
  cases: [
    {
      email_id: string,
      subject: string,
      sender: string,
      body: string,
      urls: string[],
      incident_id: string,
      expected_label: string  // 測試模式時存在
    }
  ],
  batch_mode: boolean
}
```

---

### 2. Rule-based Detector.js

**功能概述**：n8n JavaScript node，基於啟發式規則偵測郵件風險信號。

**檢測機制**：

1. **URL 偵測**（URL Detection）
   - 提取郵件文本中的所有 URL
   - 檢查 URL 是否使用 HTTP（非 HTTPS）
   - 檢查 URL 是否包含可疑字元或短連結服務

2. **認證請求偵測**（Credential Request Detection）
   - 掃描郵件中的關鍵詞：`password`, `verify`, `confirm identity`, `re-enter`, `update account`, 等
   - 計算關鍵詞出現頻率

3. **緊迫語氣偵測**（Urgency Detection）
   - 掃描郵件中的時間限制詞彙：`urgent`, `immediately`, `within 24 hours`, `expire`, 等
   - 檢查全大寫詞彙與感嘆號密度

4. **支付壓力偵測**（Payment Pressure Detection）
   - 掃描郵件中的支付相關詞彙：`payment`, `transfer`, `wire`, `invoice`, `billing`, 等
   - 結合緊迫語氣判斷支付壓力強度

**風險評分策略**：

- 統計各類信號的檢測結果
- 基於信號組合產生 `rule_risk` 評分：`LOW`, `MEDIUM`, `HIGH`
- 輸出 `detected_signals` 清單供後續決策參考

**輸出結構**：

```javascript
{
  rule_risk: "LOW" | "MEDIUM" | "HIGH",
  detected_signals: {
    urls: string[],
    credential_request: boolean,
    urgent_tone: boolean,
    payment_pressure: boolean,
    signal_count: number
  }
}
```

---

### 3. HTTP Request.json

**功能概述**：n8n HTTP Request node 的設定檔，定義對 Ollama API 的呼叫參數。

**API 端點**：`http://localhost:11434/api/chat`

**模型選擇**：`qwen2.5:7b`

**請求結構**：

```json
{
  "method": "POST",
  "url": "http://localhost:11434/api/chat",
  "headers": {
    "Content-Type": "application/json"
  },
  "body": {
    "model": "qwen2.5:7b",
    "messages": [
      {
        "role": "system",
        "content": "You are a cybersecurity analyst specializing in email security. Analyze the given email and assess its security risk."
      },
      {
        "role": "user",
        "content": "[email text with detected signals]"
      }
    ],
    "stream": false,
    "temperature": 0.3
  }
}
```

**LLM 分析指示**：
- 模型被指示針對下列要素進行評分：phishing risk、credential harvesting risk、malicious URL risk、urgency exploitation risk、financial pressure risk
- 輸出應包含：`llm_risk` (LOW/MEDIUM/HIGH)、`llm_action` (allow/review/block)、`reason` (文字說明)

**輸出結構**（由 Ollama 返回）：

```json
{
  "llm_risk": "LOW" | "MEDIUM" | "HIGH",
  "llm_action": "allow" | "review" | "block",
  "reason": "string"
}
```

---

### 4. Decision Merger.js

**功能概述**：n8n JavaScript node，合併規則型與 LLM 型風險評分，生成最終決策。

**合併策略**：

採用**保守型合併策略**（conservative merge strategy）確保高風險情況不被忽視：

$$\text{final\_risk} = \max(\text{rule\_risk}, \text{llm\_risk})$$

其中風險等級排序為：`LOW < MEDIUM < HIGH`

**決策映射**：

| final_risk | final_action | 含義 |
|-----------|-------------|------|
| LOW | allow | 郵件允許通過 |
| MEDIUM | review | 郵件標記待人工審核 |
| HIGH | block | 郵件建議攔截 |

**詳細處理流程**：

1. 驗證 rule-based detector 輸出
2. 驗證與解析 LLM 輸出（若存在格式錯誤，降級為 MEDIUM）
3. 將風險字串轉換為數值等級（LOW=1, MEDIUM=2, HIGH=3）
4. 取最大值決定 final_risk
5. 映射至對應的 final_action

**輸出結構**：

```javascript
{
  final_risk: "LOW" | "MEDIUM" | "HIGH",
  final_action: "allow" | "review" | "block",
  rule_risk: "LOW" | "MEDIUM" | "HIGH",
  llm_risk: "LOW" | "MEDIUM" | "HIGH",
  detected_signals: object,
  llm_reason: string,
  decision_trace: {
    rule_score: number,
    llm_score: number,
    final_score: number
  }
}
```

---

### 5. Format Output.js

**功能概述**：n8n JavaScript node，將分析結果格式化為 Open WebUI pipe 可讀的輸出。

**單封郵件模式**（Single Email Mode）：

輸出包含以下欄位：

```javascript
{
  email_id: string,
  subject: string,
  sender: string,
  final_risk: "LOW" | "MEDIUM" | "HIGH",
  final_action: "allow" | "review" | "block",
  rule_risk: "LOW" | "MEDIUM" | "HIGH",
  llm_risk: "LOW" | "MEDIUM" | "HIGH",
  detected_signals: object,
  decision_summary: string,
  llm_reason: string,
  timestamp: ISO8601 string
}
```

**批量評估模式**（Batch Evaluation Mode）：

當輸入為測試模式時，評估每筆案例的預測是否符合預期標籤：

```javascript
{
  batch_mode: true,
  batch_size: number,
  results: [
    {
      email_id: string,
      expected_label: string,
      final_action: string,
      pass: boolean,  // final_action 對應 expected_label
      decision_trace: object
    }
  ],
  accuracy: number,  // pass count / batch size
  summary: string
}
```

**自動格式轉換**：
- 將決策理由與信號清單轉換為易讀的文字摘要
- 確保輸出為 UTF-8 編碼的可讀文本

---

## Harness Engineering 說明

本專案實現的是**評估 Harness（Evaluation Harness）與執行時護欄（Runtime Guardrail）**的結合，而非完整的代理微調流程。

### 評估 Harness（Evaluation Harness）

目的：透過固定的測試案例組與預期標籤驗證代理行為的正確性。

**實現方式**：
- 內置 6 筆代表性電子郵件案例，涵蓋 phishing、credential request、malicious URL、urgency、payment pressure 等風險類型
- 每筆案例附帶 `expected_label`（expected_risk 與 expected_action）
- 批量評估模式計算 accuracy（正確分類的案例比例）
- 提供每筆案例的 pass/fail 判定

**評估指標**：
- 整體 accuracy
- 各風險類型的檢測率
- 假陽性與假陰性率（若測試案例覆蓋多類）

### 執行時護欄（Runtime Guardrail）

目的：透過規則型檢測與保守型決策合併策略限制 LLM 單獨決策的風險。

**實現方式**：
- **Rule-based Detector**：獨立於 LLM，基於啟發式規則提供第一層風險評分
- **Decision Merger**：採用最大值策略合併規則型與 LLM 型風險分數，確保高風險情況不被 LLM 誤判為低風險
- **可解釋性**：每筆決策輸出詳細的信號清單與決策追蹤，便於理解系統決策邏輯

**防護效果**：
- 即使 LLM 表現不佳，規則型檢測仍提供基礎風險評估
- 保守策略確保過度預測（false negatives）被優先避免

### 當前範圍與未來方向

**本版本未實現**：
- **代理微調（Agent Refinement）**：未針對郵件安全分類任務微調 qwen2.5:7b 模型
- **自我修正（Self-Correction）**：LLM 輸出不經過自我檢查或迭代改進
- **人工介入迴圈（Human-in-the-Loop）**：無標籤系統或人工反饋機制供持續改進
- **提示工程擴展（Advanced Prompting）**：使用基礎系統 prompt 與使用者 prompt，未應用 chain-of-thought 或其他進階提示技術

**設計考量**：本次實作重點在於 **n8n workflow 的可靠實現與評估 harness 的建立**，為未來的模型改進與代理優化提供基礎架構。

---

## 測試方式

### 前置條件

- n8n workflow 已部署
- Open WebUI 已啟動
- Ollama 已安裝且 `qwen2.5:7b` 模型已 pull

### 測試場景

#### 1. 批量評估測試（Batch Evaluation Test）

**操作步驟**：

1. 在 Open WebUI 聊天欄輸入：
   ```
   test
   ```
   或
   ```
   測試
   ```

2. 系統加載 6 筆預定義案例，逐一進行分析

3. 最終輸出包含：
   - 批量評估結果（所有 6 筆案例的 pass/fail）
   - 整體 accuracy

**預期輸出示例**：
```
批量評估結果：
案例1（Phishing 釣魚）: PASS - 正確判定為 HIGH 風險
案例2（Credential Request）: PASS - 正確判定為 MEDIUM 風險
...
整體準確度：83% (5/6)
```

#### 2. 結構化郵件分析（Structured Email Analysis）

**操作步驟**：

1. 在 Open WebUI 聊天欄輸入 JSON 格式郵件：
   ```json
   {
     "subject": "Urgent: Verify Your Account",
     "sender": "noreply@bank-security.com",
     "body": "Click here to verify your account within 24 hours: http://verify-now.example.com",
     "urls": ["http://verify-now.example.com"],
     "incident_id": "manual_001"
   }
   ```

2. 系統解析並進行完整分析

3. 輸出單封郵件的決策結果

**預期輸出示例**：
```
郵件分析結果：
主旨: Urgent: Verify Your Account
寄件者: noreply@bank-security.com
風險等級: HIGH
推薦行動: BLOCK
檢測信號: [credential_request, urgent_tone, suspicious_url]
決策理由: 郵件包含認證請求、緊迫語氣與 HTTP 不安全連結，風險等級為 HIGH
```

#### 3. 原始文本郵件分析（Raw Text Analysis）

**操作步驟**：

1. 在 Open WebUI 聊天欄輸入一般電子郵件文本：
   ```
   From: boss@company.com
   Subject: Payment Urgently Required
   
   Dear John,
   
   We need to process an urgent wire transfer of $50,000 to the vendor account. 
   This is time-sensitive and must be completed today. 
   Reply with your banking information immediately.
   
   Best regards,
   Boss
   ```

2. 系統自動解析為原始文本模式，進行分析

3. 輸出分析結果

**預期輸出示例**：
```
郵件分析結果（原始文本模式）：
風險等級: MEDIUM
推薦行動: REVIEW
檢測信號: [payment_pressure, urgent_tone]
決策理由: 郵件要求緊急支付轉帳且強調時間限制，建議進行人工審核
```

---

## 已知限制

### 模型層面

1. **未微調的 LLM**
   - 使用的 qwen2.5:7b 為預訓練模型，未針對電子郵件安全分類任務進行微調
   - 在某些邊界案例或複雜社交工程攻擊上的性能可能不理想

2. **無檢索增強生成（RAG）**
   - 系統未整合已知惡意 URL 黑名單、已知釣魚域名資料庫等外部知識源
   - 依賴模型自身的預訓練知識進行判斷

### 規則層面

3. **規則型檢測器的覆蓋不完整**
   - 未實現 **域名信譽分析**（domain reputation analysis）
   - 未實現 **寄件者域名不匹配偵測**（sender-domain mismatch detection）
   - 未實現 **URL 重定向分析**（URL redirect analysis）
   - 特定郵件社交工程技術可能規避現有規則

### 評估層面

4. **測試案例有限**
   - 批量評估僅包含 6 筆代表性案例
   - 尚未針對廣泛的郵件集合進行驗證
   - 可能無法反映生產環境的郵件多樣性

### 流程層面

5. **無迭代改進機制**
   - 未實現代理微調（agent refinement）流程
   - 未實現自我修正（self-correction）
   - 未實現人工介入迴圈（human-in-the-loop），缺乏標籤機制與反饋收集

### 擴展性考量

6. **適用限制**
   - 系統主要設計用於英文郵件分析；對非英文郵件的支援可能有限
   - 對複雜多語言或代碼混用的郵件內容支援度未知

---

## 技術棧與部署環境

### 所需元件

- **n8n**：Workflow 編排與執行
- **Open WebUI**：使用者介面與聊天入口
- **Ollama**：本機 LLM 推理服務（需預先 pull `qwen2.5:7b` 模型）

### 網路與 API

- Ollama API 預設在 `localhost:11434` 監聽
- n8n Webhook 接收 Open WebUI 的請求

---

## 參考資料

- [Ollama 官方文檔](https://ollama.ai)
- [n8n 官方文檔](https://docs.n8n.io)
- [Open WebUI 官方文檔](https://openwebui.com)

---

**版本**：1.0  
**最後更新**：2026 年 5 月
