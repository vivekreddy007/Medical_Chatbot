<div align="center">

# 🏥 Multi-Agent Medical Chatbot

### An AI-powered medical assistant that thinks, routes, and responds like a specialist team

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.3+-FF6B6B?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![Azure OpenAI](https://img.shields.io/badge/Azure_OpenAI-GPT--4o-0078D4?style=for-the-badge&logo=microsoft-azure&logoColor=white)](https://azure.microsoft.com/en-us/products/ai-services/openai-service)
[![License](https://img.shields.io/badge/License-Apache_2.0-green?style=for-the-badge)](LICENSE)

<br/>

> A production-ready multi-agent medical AI system built with LangGraph orchestration, RAG, real-time web search, computer vision diagnostics, speech I/O, and human-in-the-loop validation — all served through a clean FastAPI backend.

<br/>

[Features](#-features) • [Architecture](#-architecture) • [Agents](#-agents) • [Setup](#-getting-started) • [API Reference](#-api-reference) • [Tech Stack](#-tech-stack)

</div>

---

## ✨ Features

| Capability | Description |
|---|---|
| 🧠 **Intelligent Routing** | LangGraph-powered decision engine dynamically routes every query to the best-fit agent |
| 📚 **Medical RAG** | Retrieval-Augmented Generation over ingested medical literature using Qdrant vector DB |
| 🌐 **Live Web Search** | Pulls real-time data from Tavily + PubMed for current medical developments |
| 🩻 **Medical Imaging** | Analyzes Brain MRIs, Chest X-rays, and Skin Lesion images with deep learning models |
| 🔊 **Voice I/O** | Speech-to-text transcription and text-to-speech response via ElevenLabs |
| 🛡️ **Guardrails** | Input and output safety filters to prevent harmful or off-topic responses |
| 👨‍⚕️ **Human Validation** | Medical diagnoses require clinician confirmation before being finalized |
| 💬 **Conversation Memory** | Maintains context across multi-turn conversations with configurable history window |

---

## 🏗️ Architecture

```
User Query (text / image / voice)
        │
        ▼
┌───────────────────┐
│   Input Guardrails │  ← blocks unsafe or irrelevant inputs
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Agent Decision    │  ← GPT-4o routes to the right agent
│     (LangGraph)    │
└────────┬──────────┘
         │
    ┌────┴──────────────────────────────────┐
    │                                       │
    ▼                                       ▼
Text Agents                         Vision Agents
────────────                        ─────────────
Conversation Agent                  Brain Tumor Agent
RAG Agent ──► (low confidence)      Chest X-ray Agent
              Web Search Agent      Skin Lesion Agent
         │
         ▼
┌────────────────────┐
│  Human Validation   │  ← required for all medical CV outputs
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  Output Guardrails  │
└────────┬───────────┘
         │
         ▼
     Response
```

---

## 🤖 Agents

<details>
<summary><b>💬 Conversation Agent</b> — General medical chat</summary>

Handles greetings, general health questions, follow-up discussions, and non-diagnostic queries. Maintains conversation context and gracefully defers to specialists when needed.
</details>

<details>
<summary><b>📖 RAG Agent</b> — Medical knowledge base</summary>

Queries a Qdrant vector database built from ingested medical literature. Uses a cross-encoder reranker (`ms-marco-TinyBERT-L-6`) for precision. Automatically falls back to web search when retrieval confidence drops below threshold.

**Current knowledge base covers:**
- Brain tumor introduction and diagnosis techniques
- Deep learning methods for brain tumor detection
- COVID-19 detection from chest X-rays using deep learning
</details>

<details>
<summary><b>🌐 Web Search Agent</b> — Real-time medical intelligence</summary>

Combines Tavily web search with PubMed academic search to answer questions about recent outbreaks, new treatments, and time-sensitive medical topics. Results are processed and synthesized by an LLM before being returned.
</details>

<details>
<summary><b>🧠 Brain Tumor Agent</b> — MRI segmentation</summary>

Analyzes brain MRI scans to detect and segment tumors using a trained deep learning segmentation model. All outputs require human validation before being presented.
</details>

<details>
<summary><b>🫁 Chest X-ray Agent</b> — COVID-19 classification</summary>

Classifies chest X-ray images as COVID-19 positive or normal using a fine-tuned CNN model. Outputs are flagged for clinician review.
</details>

<details>
<summary><b>🔬 Skin Lesion Agent</b> — Lesion segmentation</summary>

Segments skin lesion images and generates a visual segmentation map to assist in distinguishing benign from potentially malignant regions. Requires human validation.
</details>

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Azure OpenAI API access (GPT-4o deployment + Embeddings deployment)
- Tavily API key
- ElevenLabs API key
- Qdrant (local or cloud)
- ffmpeg installed on system (for audio processing)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/Multi-Agent-Medical-Chatbot.git
cd Multi-Agent-Medical-Chatbot

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the root directory:

```env
# Azure OpenAI (LLM)
deployment_name=your-deployment-name
model_name=gpt-4o
azure_endpoint=https://your-resource.openai.azure.com/
openai_api_key=your-api-key
openai_api_version=2024-02-01

# Azure OpenAI (Embeddings)
embedding_deployment_name=your-embedding-deployment
embedding_model_name=text-embedding-3-large
embedding_azure_endpoint=https://your-resource.openai.azure.com/
embedding_openai_api_key=your-api-key
embedding_openai_api_version=2024-02-01

# Qdrant (leave blank to use local)
QDRANT_URL=
QDRANT_API_KEY=

# Search
TAVILY_API_KEY=your-tavily-key

# Speech
ELEVEN_LABS_API_KEY=your-elevenlabs-key

# HuggingFace (for reranker model)
HUGGINGFACE_TOKEN=your-hf-token
```

### Ingest Medical Documents

```bash
# Place your PDF/document files in the appropriate data folder, then:
python ingest_rag_data.py
```

### Run the Application

```bash
python app.py
```

The server will start at `http://localhost:8000`

---

## 📡 API Reference

### `POST /chat`
Send a text query to the multi-agent system.

```json
{
  "query": "What are the early signs of a brain tumor?",
  "conversation_history": []
}
```

**Response:**
```json
{
  "status": "success",
  "response": "Early signs of a brain tumor include...",
  "agent": "RAG_AGENT"
}
```

---

### `POST /upload`
Upload a medical image with an optional text query.

```
Content-Type: multipart/form-data
Fields: image (file), text (string, optional)
```

**Response:**
```json
{
  "status": "success",
  "response": "The analysis indicates...",
  "agent": "CHEST_XRAY_AGENT",
  "result_image": "/uploads/skin_lesion_output/segmentation_plot.png"
}
```

---

### `POST /transcribe`
Transcribe an audio file using ElevenLabs Scribe.

```
Content-Type: multipart/form-data
Fields: audio (file, .webm)
```

---

### `POST /generate-speech`
Convert text to speech via ElevenLabs.

```json
{
  "text": "The patient shows no signs of abnormality.",
  "voice_id": "21m00Tcm4TlvDq8ikWAM"
}
```

---

### `POST /validate`
Submit human validation for medical AI outputs.

```
Content-Type: multipart/form-data
Fields: validation_result ("yes"/"no"), comments (string, optional)
```

---

### `GET /health`
Health check endpoint for deployment monitoring.

```json
{ "status": "healthy" }
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **API Framework** | FastAPI + Uvicorn |
| **Agent Orchestration** | LangGraph (StateGraph with memory) |
| **LLM** | Azure OpenAI GPT-4o |
| **Embeddings** | Azure OpenAI text-embedding-3-large |
| **Vector DB** | Qdrant (local or cloud) |
| **Reranker** | cross-encoder/ms-marco-TinyBERT-L-6 |
| **Document Parsing** | Docling |
| **Web Search** | Tavily + PubMed API |
| **Speech** | ElevenLabs (Scribe STT + TTS) |
| **Computer Vision** | PyTorch + torchvision + OpenCV |
| **Frontend Templating** | Jinja2 |
| **Audio Processing** | pydub + ffmpeg |

---

## 📁 Project Structure

```
Multi-Agent-Medical-Chatbot/
├── app.py                          # FastAPI application & all endpoints
├── config.py                       # Centralized configuration for all agents
├── ingest_rag_data.py              # Script to ingest documents into Qdrant
├── requirements.txt
├── agents/
│   ├── agent_decision.py           # LangGraph workflow & routing logic
│   ├── guardrails/
│   │   └── local_guardrails.py     # Input/output safety filters
│   ├── rag_agent/                  # RAG pipeline (parse, embed, retrieve, rerank, respond)
│   ├── web_search_agent/           # Tavily + PubMed search & synthesis
│   └── image_analysis_agent/       # Brain tumor, chest X-ray, skin lesion models
├── templates/                      # Jinja2 HTML templates
├── data/
│   ├── qdrant_db/                  # Local Qdrant vector store
│   └── docs_db/                    # Parsed document store
└── uploads/                        # Temporary file storage (images, audio)
```

---

## ⚙️ Configuration

All system behaviour is controlled through `config.py`. Key parameters:

| Parameter | Location | Default | Description |
|---|---|---|---|
| `min_retrieval_confidence` | `RAGConfig` | `0.40` | Below this, RAG auto-routes to web search |
| `context_limit` | `RAGConfig` / `WebSearchConfig` | `20` | Messages included in conversation history |
| `max_image_upload_size` | `APIConfig` | `5 MB` | Maximum image upload size |
| `validation_timeout` | `ValidationConfig` | `300s` | Human validation timeout |
| `require_validation` | `ValidationConfig` | per agent | Toggle human validation per agent type |

---

## 🔒 Safety & Ethics

This system includes multiple layers of safety:

- **Input Guardrails** — Filters harmful, abusive, or off-topic inputs before any processing
- **Output Guardrails** — Sanitizes all LLM responses before delivery
- **Human-in-the-Loop** — All computer vision medical diagnoses (brain tumor, chest X-ray, skin lesion) require clinician confirmation
- **Disclaimer** — This tool is intended to assist healthcare professionals. It is **not** a substitute for clinical diagnosis.

---

<div align="center">

Built with FastAPI · LangGraph · Azure OpenAI · Qdrant · ElevenLabs · PyTorch

</div>
