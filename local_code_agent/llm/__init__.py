"""Interface LangChain ↔ Azure/Ollama pour Mistral Multi-Model."""
import os
from langchain_ollama import OllamaLLM
from langchain_openai import AzureChatOpenAI
from langchain_mistralai import ChatMistralAI
from local_code_agent.config import config


def _get_azure_or_fallback(target_model="mistral-large"):
    """
    Retourne un Chat Azure si la clé est présente, sinon tombe en mode Mac Mini
    vers Ollama Mistral NeMo.
    """
    api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("MISTRAL_API_KEY")
    if api_key:
        if os.getenv("AZURE_OPENAI_API_KEY"):
            # En production Total : Azure
            return AzureChatOpenAI(
                azure_deployment=target_model,
                api_version="2024-02-15-preview",
                temperature=config.temperature
            )
        else:
            # Mistral direct
            return ChatMistralAI(
                model=target_model,
                temperature=config.temperature
            )
    else:
        # Fallback de prototypage local Mac Mini (Ce qui tournait avant)
        print(f"[Warning] No Azure/Mistral API Keys found. Falling back to Ollama local ({config.model_name}) for {target_model}")
        return OllamaLLM(
            model=config.model_name,
            base_url=config.ollama_base_url,
            temperature=config.temperature,
            num_predict=config.num_predict,
        )

def get_translator_llm():
    """Le LLM "Code Monkey" pour la syntaxe (Codestral ou fallback Ollama)."""
    # En prod, on voudra "codestral-latest", en fallback on prend Nemo
    return _get_azure_or_fallback(target_model="codestral-latest")

def get_reasoning_llm():
    """L'Architecte pour analyser les logs d'erreurs (Mistral Large 2)."""
    return _get_azure_or_fallback(target_model="mistral-large-latest")

# Backward compatibility
def get_llm():
    return get_translator_llm()
