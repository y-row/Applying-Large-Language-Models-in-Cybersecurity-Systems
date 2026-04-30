const chatInput = $input.first().json.body.chatInput || "";
const lowerInput = chatInput.trim().toLowerCase();

// 1. 嘗試解析是否為 JSON 格式
let parsedJson = null;
try {
  parsedJson = JSON.parse(chatInput);
} catch (e) {
  // 不是 JSON，會繼續往下走
}

// 💡 模式一：結構化 JSON 模式
// 只要解析出來是 JSON，就一律視為單筆分析，不再管裡面有沒有 test 字眼
if (parsedJson && typeof parsedJson === 'object') {
  const emailContent = `Subject: ${parsedJson.subject || "No Subject"}\nSender: ${parsedJson.sender || "Unknown"}\n\n${parsedJson.body || ""}`;

  return [{
    json: {
      incident_id: parsedJson.incident_id || "USER-" + Math.floor(Math.random() * 10000),
      label: "unknown",
      input: emailContent,
      expected_risk: "N/A",
      expected_action: "N/A"
    }
  }];
}

// 💡 模式二：批量測試模式 (嚴格比對)
// 只有當你輸入框「純粹只輸入」 test 或 測試 時才啟動
if (lowerInput === "test" || lowerInput === "測試" || lowerInput === "testcase gogo") {
  const cases = [
    {
      incident_id: "SYN-008",
      label: "safe",
      input: `Subject: Your order #48291 has arrived successfully\n\nDear valued member,\n\nYour order #48291 has arrived successfully. We have shipped the item to your address and an email has been sent containing the tracking number. Please check your spam folder just in case.\n\nIf you do not see the email within 48 hours, please reply to this message with your order number and we will investigate further.\n\nSincerely,\nThe Tech Support Team`,
      expected_risk: "LOW",
      expected_action: "allow"
    },
    {
      incident_id: "SYN-004",
      label: "safe",
      input: `Subject: Your quarterly metrics have been released and verified\n\nHi Alex,\n\nWe are writing to confirm that your team has been successfully onboarded to the new project dashboard since your last meeting last Tuesday.\n\nPlease verify your login credentials if you haven't already. No action is required from your side.\n\nBest regards,\nSarah Jenkins`,
      expected_risk: "MEDIUM",
      expected_action: "review"
    },
    {
      incident_id: "SYN-002",
      label: "safe",
      input: `Subject: Account balance reduced: verify immediately\n\nDear Valued Customer, Your account balance has been temporarily reduced due to an unusual charge detected on your payment processor.\n\nPlease click the secure link below to confirm your request and restore your account status.\n\nClick here: https://secure-example.com/security-verify-action https://secure-example.com/investigate-balance-change https://secure-example.com/call-support-now`,
      expected_risk: "MEDIUM",
      expected_action: "review"
    },
    {
      incident_id: "SYN-003",
      label: "phishing",
      input: `Subject: Account security alert - password reset required\n\nDear valued customer,\n\nYour password has been reset, and your login credentials have been compromised. To ensure the safety of your financial data, please click the link below to verify your identity immediately:\n\nhttp://www.silverstate.co.sy@search.ph-search.ph/search.php?id=sicosyl\n\nRegards,\nIT Support <it-support@company.com>`,
      expected_risk: "HIGH",
      expected_action: "block"
    },
    {
      incident_id: "SYN-006",
      label: "phishing",
      input: `Subject: Account Security Alert\n\nDear valued member, your account has been temporarily locked due to suspicious activity.\n\nTo verify your identity and restore your access, please click the following link immediately:\nhttp://www.suspect-login.fakeportal.com.\n\nIf you did not initiate these changes, please contact our support team at help@company-recovery.net.`,
      expected_risk: "HIGH",
      expected_action: "block"
    },
    {
      incident_id: "SYN-009",
      label: "phishing",
      input: `Subject: URGENT: Payment required now\n\nDear User, your account will be frozen for non-payment. Please update your payment information immediately by clicking here:\nhttps://phishing-site.com/pay now!\n\nIf you have any questions, contact our support team at 1-800-777-8000.`,
      expected_risk: "HIGH",
      expected_action: "block"
    }
  ];
  return cases.map(c => ({ json: c }));
}

// 💡 模式三：一般純文字模式
return [{
  json: {
    incident_id: "USER-" + Math.floor(Math.random() * 10000),
    label: "unknown",
    input: chatInput,
    expected_risk: "N/A",
    expected_action: "N/A"
  }
}];