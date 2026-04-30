return items.map(item => {
  const ruleRisk = item.json.rule_risk;

  // LLM 回傳是 string JSON，要 parse
  let llmRisk = "LOW";
  let llmAction = "allow";

  try {
    const content = item.json.message.content;
    const parsed = JSON.parse(content);

    llmRisk = parsed.llm_risk;
    llmAction = parsed.llm_action;
  } catch (e) {
    // fallback（LLM 爛掉時）
    llmRisk = "LOW";
    llmAction = "allow";
  }

  // risk mapping
  const riskScore = { LOW: 1, MEDIUM: 2, HIGH: 3 };

  const finalRisk =
    riskScore[ruleRisk] > riskScore[llmRisk]
      ? ruleRisk
      : llmRisk;

  let finalAction = "allow";
  if (finalRisk === "HIGH") finalAction = "block";
  else if (finalRisk === "MEDIUM") finalAction = "review";

  return {
    json: {
      ...item.json,
      llm_risk: llmRisk,
      llm_action: llmAction,
      final_risk: finalRisk,
      final_action: finalAction
    }
  };
});