"""Interface LangChain ↔ Azure Mistral (via OpenAI-compatible MaaS endpoint)."""
import os
from langchain_openai import ChatOpenAI
from local_code_agent.config import config


def get_llm():
    """Retourne un client LLM vers l'endpoint Azure AI MaaS (Mistral-Large).

    Azure Cognitive Services expose une API OpenAI-compatible.
    On utilise ChatOpenAI (pas ChatMistralAI) car le SDK Mistral construit
    une URL incorrecte pour les endpoints cognitiveservices.azure.com.
    URL construite : {base_url}/chat/completions  ✓
    """
    api_key  = os.getenv("AZURE_MISTRAL_API_KEY")
    endpoint = os.getenv("AZURE_MISTRAL_ENDPOINT", "").rstrip("/")

    if not api_key or not endpoint:
        raise ValueError(
            "AZURE_MISTRAL_API_KEY et AZURE_MISTRAL_ENDPOINT doivent être définis (cf .env)"
        )

    target = os.getenv("AZURE_MISTRAL_MODEL", "mistral-large-latest")
    return ChatOpenAI(
        base_url=endpoint,
        api_key=api_key,
        model=target,
        temperature=config.temperature,
    )


# Backward-compatibility aliases (anciennement deux fonctions identiques)
get_translator_llm = get_llm
get_reasoning_llm  = get_llm
