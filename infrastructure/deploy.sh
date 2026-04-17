#!/bin/bash
# ==============================================================================
# Script de déploiement d'Infrastructure Azure via Azure CLI
# Pour l'Agent de Refactoring JAX HPC 
# ==============================================================================

# Vérification de l'authentification
if ! az account show >/dev/null 2>&1; then
    echo "❌ Vous n'êtes pas connecté à Azure. Lancez 'az login' en premier."
    exit 1
fi

set -e # Arret du script en cas d'erreur

RG_NAME="rg-total-seismic-agent"
LOCATION="francecentral"
VNET_NAME="vnet-seismic"
SUBNET_NAME="snet-compute"

echo "✅ 1. Création du Groupe de Ressources ($RG_NAME) dans $LOCATION..."
az group create --name $RG_NAME --location $LOCATION -o table

echo "✅ 2. Création du Réseau Virtuel (VNET) et du Sous-réseau (Subnet)..."
az network vnet create \
  --resource-group $RG_NAME \
  --name $VNET_NAME \
  --address-prefix "10.0.0.0/16" \
  --subnet-name $SUBNET_NAME \
  --subnet-prefix "10.0.1.0/24" -o table

# On désactive les avertissements polluants et on force le mode erreurs seules
export AZURE_CORE_ONLY_SHOW_ERRORS=True
az config set core.display_region_identified=false

echo "✅ 3. Déploiement de l'Instance d'Orchestration (Serveur LangGraph)..."
# Standard_D8s_v5 (Usage général)
az vm create \
  --resource-group $RG_NAME \
  --name "vm-orchestrator-d8" \
  --image "Ubuntu2204" \
  --size "Standard_D8s_v5" \
  --admin-username "azureuser" \
  --admin-password "P@ssw0rdTotal2026!" \
  --vnet-name $VNET_NAME \
  --subnet $SUBNET_NAME \
  --public-ip-address "pip-orchestrator" \
  --nsg "nsg-orchestrator" \
  --output none

echo "✅ 4. Déploiement de l'Instance GPU Haute Performance (Nœud JAX/FNO)..."
# Standard_NC24ads_A100_v4
az vm create \
  --resource-group $RG_NAME \
  --name "vm-gpu-a100" \
  --image "microsoft-dsvm:ubuntu-2204:2204-gen2:latest" \
  --size "Standard_NC24ads_A100_v4" \
  --admin-username "azureuser" \
  --admin-password "P@ssw0rdTotal2026!" \
  --vnet-name $VNET_NAME \
  --subnet $SUBNET_NAME \
  --public-ip-address "pip-gpu" \
  --nsg "nsg-gpu" \
  --output none

echo "🚀 Déploiement réussi !"
echo "Les adresses IP ont été provisionnées. Utilisez 'az vm list-ip-addresses -g $RG_NAME -o table' pour vous connecter en SSH."
