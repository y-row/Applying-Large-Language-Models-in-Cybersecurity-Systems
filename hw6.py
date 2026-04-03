from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# -----------------------------
# 1. Sample phishing email records
# -----------------------------
records = [
    {
        "incident_id": "PH-001",
        "text": """Subject: Urgent: Password Expiration Notice
Sender: security@micr0soft-login.co
Receiver: user@company.com
Date: 2025-04-01
Label: phishing
Severity: high
Social Engineering Tactic: urgency
Recommended Action: Block sender and reset password if clicked
Body: Your Microsoft 365 password expires today. Verify now via the link below."""
    },
    {
        "incident_id": "PH-002",
        "text": """Subject: Invoice Attached
Sender: billing@vendor-payments.biz
Receiver: finance@company.com
Date: 2025-04-02
Label: phishing
Severity: high
Social Engineering Tactic: invoice fraud
Recommended Action: Quarantine attachment and inspect file
Body: Please review the attached invoice and confirm payment today."""
    },
    {
        "incident_id": "PH-003",
        "text": """Subject: Team Lunch Poll
Sender: hr@company.com
Receiver: staff@company.com
Date: 2025-04-03
Label: safe
Severity: low
Social Engineering Tactic: normal request
Recommended Action: No action required
Body: Please choose your lunch option for Friday."""
    },
    {
        "incident_id": "PH-004",
        "text": """Subject: Mailbox Storage Full
Sender: alert@mailbox-reset.net
Receiver: employee@company.com
Date: 2025-04-04
Label: phishing
Severity: high
Social Engineering Tactic: fear
Recommended Action: Block URL and inspect browser history if clicked
Body: Your mailbox is full. Click here to avoid losing incoming emails."""
    },
    {
        "incident_id": "PH-005",
        "text": """Subject: Updated VPN Policy
Sender: security@company.com
Receiver: all@company.com
Date: 2025-04-05
Label: safe
Severity: low
Social Engineering Tactic: normal request
Recommended Action: No action required
Body: Please review the updated VPN policy before Friday."""
    }
]

# -----------------------------
# 2. Convert to Documents
# 不額外切 chunk，避免 label / action / body 被切散
# -----------------------------
docs = [
    Document(page_content=r["text"], metadata={"incident_id": r["incident_id"]})
    for r in records
]

# -----------------------------
# 3. Embedding + Vector Store
# -----------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(docs, embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# -----------------------------
# 4. Generation model (Qwen)
# -----------------------------
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# -----------------------------
# 5. Retrieval helper
# 先把最相關的 records 找回來，再組成乾淨 context
# -----------------------------
def build_context(query: str, k: int = 2):
    retrieved_docs = retriever.invoke(query)
    context = "\n\n---\n\n".join([d.page_content for d in retrieved_docs])
    return retrieved_docs, context

# -----------------------------
# 6. Generation helper
# -----------------------------
def generate_answer(query: str, context: str):
    prompt_text = f"""You are a cybersecurity email analyst.
Use only the retrieved context to answer the question.
Do not answer with only one word.
Give a short but complete answer.

Your answer must include exactly these three sections:
Classification:
Reason:
Recommended Action:

If the context does not provide enough evidence, say:
I do not have enough evidence from the retrieved context.

Retrieved context:
{context}

Question:
{query}

Answer:"""

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(model.device)

    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=220,
            do_sample=False,
            repetition_penalty=1.05,
            eos_token_id=tokenizer.eos_token_id
        )

    new_tokens = outputs[0][input_len:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # 去掉常見回顯
    for marker in ["Assistant:", "Answer:", "Human:"]:
        if marker in decoded:
            decoded = decoded.split(marker, 1)[-1].strip()

    return decoded

# -----------------------------
# 7. Test query + debug retrieval
# -----------------------------
query = (
    "An email claims that a Microsoft 365 password will expire today and asks the user "
    "to verify immediately. Based on the retrieved records, explain whether this is likely "
    "phishing, why, and what action should be taken."
)

retrieved_docs, context = build_context(query)

print("=== Retrieved Docs ===")
for i, d in enumerate(retrieved_docs):
    print(f"[Doc {i+1}] (incident_id={d.metadata['incident_id']})")
    print(d.page_content)
    print("-" * 80)

answer = generate_answer(query, context)

print("=== Final Answer ===")
print(answer)