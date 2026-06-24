# Zyro Dynamics HR Help Desk

**NIAT Masterclass RAG Challenge** — Intelligent HR Policy Chatbot

## 🚀 Live App
[Launch HR Help Desk](https://zyro-hr-rag.streamlit.app)

## 🏗️ Architecture
- **Document Loading**: PyPDF + LangChain
- **Embeddings**: `sentence-transformers/all-mpnet-base-v2`
- **Vector Store**: FAISS
- **LLM**: Groq (`llama-3.3-70b-versatile`)
- **Tracing**: LangSmith
- **UI**: Streamlit

## 📁 Corpus
10 HR policy documents for Acrux Dynamics:
- Company Profile, Employee Handbook, Leave Policy
- WFH Policy, Code of Conduct, Performance Review Policy
- Compensation & Benefits, IT & Data Security
- POSH Policy, Onboarding & Separation, Travel & Expense

## 🛡️ Guardrails
Out-of-scope questions (financials, competitor policies, product features) are automatically declined.
