#!/bin/bash
set -e

# Démarrer le serveur Ollama en arrière-plan
ollama serve &
OLLAMA_PID=$!

# Attendre que l'API soit disponible
echo "Waiting for Ollama to be ready..."
until curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; do
  sleep 1
done
echo "Ollama is ready."

# Pull du modèle si absent
MODEL="${OLLAMA_MODEL:-mistral-nemo:12b}"
if ollama list | grep -q "$MODEL"; then
  echo "Model $MODEL already present."
else
  echo "Pulling model $MODEL ..."
  ollama pull "$MODEL"
  echo "Model $MODEL pulled."
fi

# Garder le processus Ollama en premier plan
wait $OLLAMA_PID
