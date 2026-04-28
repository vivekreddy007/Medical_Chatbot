import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

def _llm(temperature: float) -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=temperature,
        max_retries=3,
    )


class AgentDecisionConfig:
    def __init__(self):
        self.llm = _llm(0.1)


class ConversationConfig:
    def __init__(self):
        self.llm = _llm(0.7)
        self.context_limit = 20


class WebSearchConfig:
    def __init__(self):
        self.llm = _llm(0.3)
        self.context_limit = 20


class RAGConfig:
    def __init__(self):
        self.vector_db_type = "qdrant"
        self.embedding_dim = 1536
        self.distance_metric = "Cosine"
        self.use_local = True
        self.vector_local_path = "./data/qdrant_db"
        self.doc_local_path = "./data/docs_db"
        self.parsed_content_dir = "./data/parsed_docs"
        self.url = os.getenv("QDRANT_URL")
        self.api_key = os.getenv("QDRANT_API_KEY")
        self.collection_name = "medical_assistance_rag"
        self.chunk_size = 512
        self.chunk_overlap = 50
        self.embedding_model = OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.llm = _llm(0.3)
        self.summarizer_model = _llm(0.5)
        self.chunker_model = _llm(0.0)
        self.response_generator_model = _llm(0.3)
        self.top_k = 5
        self.vector_search_type = "similarity"
        self.huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
        self.reranker_model = "cross-encoder/ms-marco-TinyBERT-L-6"
        self.reranker_top_k = 3
        self.max_context_length = 8192
        self.include_sources = True
        self.min_retrieval_confidence = 0.40
        self.context_limit = 20


class MedicalCVConfig:
    def __init__(self):
        self.brain_tumor_model_path = "./agents/image_analysis_agent/brain_tumor_agent/models/brain_tumor_segmentation.pth"
        self.brain_tumor_segmentation_output_path = "./uploads/brain_tumor_output/segmentation_plot.png"
        self.chest_xray_model_path = "./agents/image_analysis_agent/chest_xray_agent/models/covid_chest_xray_model.pth"
        self.skin_lesion_model_path = "./agents/image_analysis_agent/skin_lesion_agent/models/skin_lesion_segmentation.pth"
        self.skin_lesion_segmentation_output_path = "./uploads/skin_lesion_output/segmentation_plot.png"
        self.llm = _llm(0.1)


class SpeechConfig:
    def __init__(self):
        self.tts_model = "tts-1"
        self.tts_voice = "alloy"  # options: alloy, echo, fable, onyx, nova, shimmer
        self.stt_model = "whisper-1"


class ValidationConfig:
    def __init__(self):
        self.require_validation = {
            "CONVERSATION_AGENT": False,
            "RAG_AGENT": False,
            "WEB_SEARCH_AGENT": False,
            "BRAIN_TUMOR_AGENT": True,
            "CHEST_XRAY_AGENT": True,
            "SKIN_LESION_AGENT": True,
        }
        self.validation_timeout = 300
        self.default_action = "reject"


class APIConfig:
    def __init__(self):
        self.host = "0.0.0.0"
        self.port = 8000
        self.rate_limit = 10  # requests per minute per IP
        self.max_image_upload_size = 5  # MB


class Config:
    def __init__(self):
        self.agent_decision = AgentDecisionConfig()
        self.conversation = ConversationConfig()
        self.rag = RAGConfig()
        self.medical_cv = MedicalCVConfig()
        self.web_search = WebSearchConfig()
        self.api = APIConfig()
        self.speech = SpeechConfig()
        self.validation = ValidationConfig()
        self.eleven_labs_api_key = os.getenv("ELEVEN_LABS_API_KEY")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.max_conversation_history = 20
