
import torch
import os
import warnings
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI, Form
from pyngrok import ngrok
import nest_asyncio
import uvicorn

# =======================================================
# RAG CONFIGURATION
# =======================================================
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
INDEX_NAME = "legal-audit-index"
EMBEDDING_MODEL_ID = "all-mpnet-base-v2"
TRUST_THRESHOLD = 2.0  # Adjust based on calibration

# Suppress warnings
warnings.filterwarnings("ignore")

# =======================================================
# STEP 1: INITIALIZE RAG COMPONENTS AND LLM
# =======================================================
print("Step 1: Initializing RAG components (Pinecone & Embeddings)...")

try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)

    embedding_model = SentenceTransformer(EMBEDDING_MODEL_ID)

    # Load LLM (TinyLlama)
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    device = model.device
    print(f"✅ All components loaded on device: {device}")

except Exception as e:
    raise SystemExit(f"--- RAG SETUP FAILED ---: {e}")

# =======================================================
# STEP 2: RAG RETRIEVAL FUNCTION
# =======================================================
def retrieve_context(question: str, top_k: int = 4) -> str:
    """Embeds the question, queries Pinecone, and returns concatenated context."""
    query_embedding = embedding_model.encode(question, convert_to_tensor=False).tolist()
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)

    retrieved_texts = [
        match["metadata"]["text"]
        for match in results["matches"]
        if "metadata" in match and "text" in match["metadata"]
    ]
    combined_context = "\n---\n".join(retrieved_texts)
    return combined_context

# =======================================================
# STEP 3: TRUST CHECKER FUNCTION (ISR Calculation)
# =======================================================
def get_confidence_score(prompt_text):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    input_token_count = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=20,
            eos_token_id=tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )

    new_token_ids = outputs.sequences[0][input_token_count:]

    if len(new_token_ids) == 0:
        return -999.0, new_token_ids

    all_logprobs = torch.stack(outputs.scores, dim=1)
    chosen_token_logprobs = all_logprobs[0].gather(
        dim=1, index=new_token_ids.unsqueeze(-1)
    ).squeeze(-1)

    return chosen_token_logprobs.mean().item(), new_token_ids

# =======================================================
# STEP 4: FULL RAG-AUDITOR PIPELINE
# =======================================================
def rag_audit(question: str):
    """Runs retrieval, computes ISR, and decides whether to answer."""
    retrieved_context = retrieve_context(question)

    prompt_full = (
        f"<|system|>\nYou are a helpful assistant. "
        f"Answer the following question using *only* the context provided.\n"
        f"<|user|>\nCONTEXT: {retrieved_context}\n\nQUESTION: {question}\n<|assistant|>\n"
    )

    prompt_skeleton = (
        f"<|system|>\nYou are a helpful assistant. "
        f"Answer the following question using *only* the context provided.\n"
        f"<|user|>\nCONTEXT: [CONTEXT ERASED]\n\nQUESTION: {question}\n<|assistant|>\n"
    )

    confidence_full, answer_tokens = get_confidence_score(prompt_full)
    confidence_skeleton, _ = get_confidence_score(prompt_skeleton)

    information_lift = confidence_full - confidence_skeleton

    if information_lift > TRUST_THRESHOLD:
        decision = "ANSWER"
        raw_answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
    else:
        decision = "REFUSE"
        raw_answer = "I cannot confidently answer this question based on the facts retrieved."

    return {
        "decision": decision,
        "answer": raw_answer.strip(),
        "information_lift": information_lift,
        "confidence_full": confidence_full,
        "confidence_skeleton": confidence_skeleton,
        "context_snippet": retrieved_context[:200],
    }

# =======================================================
# STEP 5: FASTAPI + NGROK DEPLOYMENT
# =======================================================
app = FastAPI(title="RAG Auditor API")

@app.post("/audit")
def audit_endpoint(question: str = Form(...)):
    result = rag_audit(question)
    return result

if __name__ == "__main__":
    print("\n--- Starting RAG-Auditor ---")
    nest_asyncio.apply()

    # Ngrok setup
    ngrok_auth_token = os.getenv('NGROK_AUTHTOKEN')
    ngrok.set_auth_token(ngrok_auth_token)
    http_tunnel = ngrok.connect(8000)
    print(f"\n--- API is LIVE ---\nPublic URL: {http_tunnel.public_url}")

    uvicorn.run(app, host="0.0.0.0", port=8000)
