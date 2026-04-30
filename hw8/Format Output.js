const items = $input.all();
const originalItems = $('Rule-based Detector').all();
const llmItems = $('HTTP Request').all();

const summary = items.map((item, index) => {
  const original = originalItems[index].json;
  const llmResponse = llmItems[index].json;

  let aiReason = "N/A";
  try {
    const contentObj = JSON.parse(llmResponse.message.content);
    aiReason = contentObj.reason || item.json.reason || "未提供理由";
  } catch (e) {
    aiReason = item.json.reason || "理由解析失敗";
  }

  return {
    incident_id: original.incident_id || "未知",
    expected_risk: original.expected_risk || "N/A",
    predicted_risk: item.json.final_risk || "N/A",
    expected_action: original.expected_action || "N/A",
    predicted_action: item.json.final_action || "N/A",
    reason: aiReason,
    pass: item.json.final_risk === original.expected_risk && 
          item.json.final_action === original.expected_action
  };
});

// 👇 新增動態排版邏輯
let reportText = "";

// 判斷是否為「單筆分析模式」（只有一筆，且沒有標準答案）
const isSingleAnalysis = summary.length === 1 && summary[0].expected_risk === "N/A";

if (isSingleAnalysis) {
  const s = summary[0];
  const cleanReason = s.reason.replace(/\n/g, ' ');
  
  reportText = `### 🔍 單筆郵件安全分析 (${s.incident_id})\n\n`;
  reportText += `- **風險等級：** ${s.predicted_risk}\n`;
  reportText += `- **建議處置：** ${s.predicted_action}\n`;
  reportText += `- **分析理由：** ${cleanReason}\n`;
  
} else {
  // 原本的批量測試表格模式
  const total = summary.length;
  const passed = summary.filter(x => x.pass).length;

  reportText = `### 🛡️ 批量安全分析測試報告\n\n`;
  reportText += `**系統準確率 (Accuracy):** ${passed} / ${total}\n\n`;
  reportText += `| 案件編號 | 實際風險 | AI 預測 | 建議動作 | 判定結果 | 分析理由 |\n`;
  reportText += `|---|---|---|---|---|---|\n`;

  summary.forEach(s => {
    const passIcon = s.pass ? '✅ 成功' : '❌ 失敗';
    const cleanReason = s.reason.replace(/\n/g, ' '); 
    reportText += `| ${s.incident_id} | ${s.expected_risk} | ${s.predicted_risk} | ${s.predicted_action} | ${passIcon} | ${cleanReason} |\n`;
  });
}

return [
  {
    json: {
      output: reportText
    }
  }
];