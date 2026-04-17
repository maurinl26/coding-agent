"""Interface LangChain ↔ Azure Mistral (via OpenAI-compatible MaaS endpoint)."""
import os
from langchain_openai import ChatOpenAI
from local_code_agent.config import config


def _get_azure_mistral(target_model):
    """
    Retourne un client LLM pointant vers l'Endpoint Azure AI MaaS.

    Architecture : Azure Cognitive Services expose une API OpenAI-compatible.
    On utilise ChatOpenAI (et non ChatMistralAI) car le SDK Mistral construit
    une URL incorrecte pour les endpoints cognitiveservices.azure.com.

    URL construite : {base_url}/chat/completions  ✅
    """
    api_key = os.getenv("AZURE_MISTRAL_API_KEY")
    endpoint = os.getenv("AZURE_MISTRAL_ENDPOINT", "").rstrip("/")

    if not api_key or not endpoint:
        raise ValueError("AZURE_MISTRAL_API_KEY et AZURE_MISTRAL_ENDPOINT doivent être définis (cf .env)")

    # L'endpoint Azure AI (services.ai.azure.com/openai/v1/) est utilisé directement.
    # ChatOpenAI construit : {base_url}/chat/completions ✅
    return ChatOpenAI(
        base_url=endpoint,
        api_key=api_key,
        model=target_model,
        temperature=config.temperature
    )


def get_translator_llm():
    """LLM de traduction syntaxique (Mistral Large via Azure MaaS)."""
    target = os.getenv("AZURE_MISTRAL_MODEL", "mistral-large-latest")
    return _get_azure_mistral(target_model=target)


def get_reasoning_llm():
    """LLM d'architecture et raisonnement (Mistral Large via Azure MaaS)."""
    target = os.getenv("AZURE_MISTRAL_MODEL", "mistral-large-latest")
    return _get_azure_mistral(target_model=target)


# Backward compatibility
def get_llm():
    return get_translator_llm()
