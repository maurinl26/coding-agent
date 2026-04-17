"""Interface LangChain ↔ Azure Mistral Multi-Model."""
import os
from langchain_mistralai import ChatMistralAI
from local_code_agent.config import config


def _get_azure_mistral(target_model):
    """
    Retourne l'Endpoint Azure AI Studio configuré.
    Doit planter s'il n'y a pas d'API Key Azure (plus de fallback local).
    """
    api_key = os.getenv("AZURE_MISTRAL_API_KEY")
    endpoint = os.getenv("AZURE_MISTRAL_ENDPOINT")
    
    if not api_key or not endpoint:
        raise ValueError("AZURE_MISTRAL_API_KEY et AZURE_MISTRAL_ENDPOINT doivent être définis (cf .env)")
        
    return ChatMistralAI(
        endpoint=endpoint,
        api_key=api_key,
        model=target_model,
        temperature=config.temperature
    )

def get_translator_llm():
    """Le LLM "Code Monkey" pour la syntaxe (Codestral)."""
    return _get_azure_mistral(target_model="codestral-latest")

def get_reasoning_llm():
    """L'Architecte pour analyser les logs d'erreurs et DL (Mistral Large)."""
    return _get_azure_mistral(target_model=os.getenv("AZURE_MISTRAL_MODEL", "mistral-large-latest"))

# Backward compatibility
def get_llm():
    return get_translator_llm()
