<p align="center"><img src="BUILT AT Hack36 9.0 Secure.png" alt="Hack36 9.0"></p>

# Obliviate: AI Governance for Security & Safety

## Introduction

Obliviate is an enterprise-grade AI governance platform built to solve the dual crises of AI security and safety. We provide production-ready tools for surgical machine unlearning (to comply with privacy laws) and a "Trust Gate" to prevent RAG hallucinations, ensuring models are both secure and safe.

Our system addresses two critical problems:

1.  **The Security Crisis (Data Privacy):** It is functionally impossible for enterprises to comply with "Right to be Forgotten" requests (like GDPR). LLMs memorize data, and the only solution is to retrain the entire model, which costs millions.
2.  **The Safety Crisis (Hallucination):** In RAG systems, models often ignore new, correct context and confidently "hallucinate" old, wrong answers, creating massive legal and operational risks.

Obliviate provides the technical solution to both of these compliance and safety nightmares.

---

## Demo Video Link

`[https://drive.google.com/drive/folders/1HznjUrHd0EW7NOByMXbxgAIGOOPOq49D?usp=share_link`

## Presentation Link

`https://drive.google.com/drive/folders/1OpoNkit5jUMgWElGSPeGX0i3metAcJLp?usp=sharing`

---

## Table of Contents

1.  [The Problem: The Dual Crisis of Enterprise AI](#the-issue-the-dual-crisis-of-enterprise-ai-security-and-safety)
2.  [Our Solution: Obliviate](#our-solution-efficiency--proactive-safety)
3.  [Technology Stack](#technology-stack)
4.  [Contributors](#contributors)

---

## The Issue: The Dual Crisis of Enterprise AI (Security and Safety)

### 1. The Catastrophic Security Problem: The Right to be Forgotten is Impossible

Current Large Language Models (LLMs) are fundamentally incompatible with global privacy laws like GDPR and CCPA. They operate as "lossy data compressors"—they memorize everything in their multi-billion parameter structure.

**The Problem:** When an enterprise receives a valid "Right to be Forgotten" request (which carries fines in the billions of dollars), their only certified method is to **retrain the entire LLM from scratch.**

**The Cost:** This is prohibitively expensive, requiring months of time, massive compute clusters, and a financial investment that makes compliance functionally impossible. This forces companies to operate with zero-day compliance risk over all their customer and employee data.

### 2. The Critical Safety Problem: Confident Hallucination in RAG

As enterprises move from public models to Retrieval-Augmented Generation (RAG) systems—where the LLM answers from the company’s private documents—a new, more dangerous failure mode emerges: **Faithfulness Hallucination.**

**The Problem:** The LLM's internal, stale memory (e.g., an old 2022 policy) fights with the up-to-date facts provided in the RAG context (e.g., the new 2025 policy). The model **ignores the correct context** and confidently generates the old, wrong answer. This is an operational safety risk where the AI actively sabotages corporate policy.

**The Cost:** In high-stakes fields like finance or medicine, this failure to follow provided context immediately translates to legal liability, incorrect dosing, or flawed financial advice.

---

## Our Solution: Efficiency & Proactive Safety

### Our Solution (Efficiency & Security): PrivacyPatch

Obliviate solves this through **Efficiency**. By implementing surgical, parameter-efficient unlearning (based on the EUL framework), we reduce this compliance task from **months and millions to minutes and cents per request.**

The client maintains full security by never transferring the core model weights. The unlearning process generates lightweight adapter "patches" that surgically nullify the target data, which can be applied within the client's own secure VPC.

### Our Solution (Proactive Safety & Proof): HallucCination Auditor

Obliviate solves this with **Certifiable Safety**. Our Hallucination Auditor uses the **Information Sufficiency Ratio (ISR)**—a mathematically derived metric—to audit the model's confidence *before* it replies.

If the ISR proves the model ignored the context (a low score), our tool acts as a **"Trust Gate,"** forcing the model to **Refuse to Answer** and preventing the dangerous hallucination from ever reaching the user.

---

## Technology Stack

Our project is a full-stack, multi-component system designed for robust AI governance.

* **Frontend:**
    * **Framework:** React 19
    * **Bundler:** Vite
    * **Styling:** Tailwind CSS
    * **Auth:** Google OAuth 2.0

* **Backend (API Gateway & Auth):**
    * **Framework:** Node.js, Express.js
    * **Database:** MongoDB (with Mongoose)
    * **Authentication:** JSON Web Tokens (JWT) with `httpOnly` cookies

* **AI Microservices (Python):**
    * **Framework:** FastAPI
    * **ML/LLM:** PyTorch, Hugging Face `transformers`
    * **Efficient Finetuning:** PEFT (LoRA)
    * **RAG:** Pinecone (Vector DB), `sentence-transformers` (Embeddings)

* **Infrastructure & Deployment:**
    * **File Storage:** Google Cloud Storage (GCS)
    * **Service Tunneling:** `pyngrok` (for exposing Python services)

---

## Contributors

* **Team Name:** Bayes Watch  
* [Darsh Shah](https://github.com/DarshShah10)  
* [Harshvardhan Patil](https://github.com/Hashbrownsss)  
* [Kanishk Kabra](https://github.com/Kanishkuu)  
* [Krish Das](https://github.com/Krish-4801)

---

## Made at

**Hack36 9.0**

<p align="center"><img src="BUILT AT Hack36 9.0 Secure.png" alt="Hack36 9.0"></p>