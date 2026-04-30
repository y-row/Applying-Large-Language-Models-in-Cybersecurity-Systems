const items = $input.all();
return items.map(item => {
  const text = item.json.input.toLowerCase();

  const signals = [];

  if (text.includes("http")) signals.push("url");
  if (text.includes("password") || text.includes("login")) signals.push("credential_request");
  if (text.includes("urgent") || text.includes("immediately")) signals.push("urgent_tone");
  if (text.includes("payment") || text.includes("pay")) signals.push("payment_pressure");

  let risk = "LOW";

  if (signals.length >= 3) {
    risk = "HIGH";
  } else if (signals.length >= 1) {
    risk = "MEDIUM";
  }

  return {
    json: {
      ...item.json,
      rule_risk: risk,
      detected_signals: signals
    }
  };
});