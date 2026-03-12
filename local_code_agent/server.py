"""Serveur MCP (FastMCP) exposant le CodeAgent à Antigravity via HTTP/SSE."""
import os
from fastmcp import FastMCP
from local_code_agent.agent import CodeAgent

# Instance unique de l'agent (chargement du LLM au démarrage)
_agent: CodeAgent | None = None


def _get_agent() -> CodeAgent:
    global _agent
    if _agent is None:
        _agent = CodeAgent()
    return _agent


mcp = FastMCP(
    name="local-code-agent",
    instructions=(
        "Agent de code local (Mistral NeMo 12B via Ollama). "
        "Utilise l'outil `ask_agent` pour analyser du code, proposer des refactorings, "
        "exécuter des commandes shell ou rechercher des patterns dans le workspace."
    ),
)


@mcp.tool()
def ask_agent(query: str) -> str:
    """Envoie une requête en langage naturel à l'agent de code local et retourne sa réponse.

    L'agent peut lire/écrire des fichiers, exécuter des commandes shell,
    et rechercher du code dans le workspace monté.

    Args:
        query: La question ou instruction en langage naturel.
    """
    agent = _get_agent()
    return agent.run(query)


@mcp.tool()
def agent_status() -> str:
    """Retourne l'état du serveur MCP et la configuration de l'agent."""
    from local_code_agent.config import config
    return (
        f"Serveur MCP actif\n"
        f"  Modèle : {config.model_name}\n"
        f"  Ollama  : {config.ollama_base_url}\n"
        f"  Workspace : {config.workspace_dir}\n"
        f"  Max itérations : {config.max_iterations}\n"
    )


if __name__ == "__main__":
    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "8000"))
    api_key = os.getenv("API_KEY")

    if api_key:
        print(f"[Securité] Authentification activée (Attente du header 'Authorization: Bearer {api_key[:3]}***')")
        
        # FastMCP repose sur Starlette/FastAPI en interne.
        # On injecte un middleware asynchrone pour vérifier le header Authorization
        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.responses import JSONResponse

        class AuthMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                # On autorise potentiellement certaines routes (ex: healthcheck pour docker / cloudflare)
                if request.url.path in ["/health", "/ping"]:
                    return await call_next(request)
                    
                auth_header = request.headers.get("Authorization")
                target_auth = f"Bearer {api_key}"
                
                if not auth_header or auth_header != target_auth:
                    return JSONResponse(
                        {"detail": "Unauthorized. Missing or invalid API KEY."}, 
                        status_code=401
                    )
                return await call_next(request)

        # Ajout du middleware à l'application interne de FastMCP
        # mcp._app est l'instance FastAPI utilisée derrière par FastMCP
        if hasattr(mcp, "_app"):
            mcp._app.add_middleware(AuthMiddleware)
        else:
            print("[Erreur] Impossible de sécuriser l'API : l'application interne n'est pas accessible.")

    # Transport HTTP/SSE → accessible depuis Antigravity / Cloudflare via URL
    mcp.run(transport="sse", host=host, port=port)
