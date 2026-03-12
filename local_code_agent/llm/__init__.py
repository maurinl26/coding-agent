"""Interface LangChain ↔ Ollama pour Mistral NeMo 12B."""
from langchain_ollama import OllamaLLM
from local_code_agent.config import config


def get_llm() -> OllamaLLM:
    """Retourne une instance configurée du LLM Ollama."""
    return OllamaLLM(
        model=config.model_name,
        base_url=config.ollama_base_url,
        temperature=config.temperature,
        top_p=config.top_p,
        num_predict=config.num_predict,
    )
