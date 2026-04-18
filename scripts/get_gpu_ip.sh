#!/bin/bash
# get_gpu_ip.sh — Récupère l'IP de la VM GPU et met à jour .env
#
# Usage : bash scripts/get_gpu_ip.sh
#         bash scripts/get_gpu_ip.sh --set-env   (écrit AZURE_GPU_HOST dans .env)

RG="rg-total-seismic-agent"
VM="${AZURE_GPU_VM:-vm-gpu-t4}"

IP=$(az vm show -g "$RG" -n "$VM" --show-details --query publicIps -o tsv 2>/dev/null)

if [ -z "$IP" ]; then
    echo "VM '$VM' introuvable ou non démarrée dans le groupe '$RG'."
    echo ""
    echo "Vérifiez les VMs disponibles :"
    az vm list -g "$RG" -o table 2>/dev/null || echo "  (aucune VM trouvée)"
    exit 1
fi

echo "$IP"

if [ "$1" = "--set-env" ]; then
    ENV_FILE=".env"
    if grep -q "AZURE_GPU_HOST" "$ENV_FILE" 2>/dev/null; then
        sed -i '' "s/^AZURE_GPU_HOST=.*/AZURE_GPU_HOST=${IP}/" "$ENV_FILE"
    else
        echo "AZURE_GPU_HOST=${IP}" >> "$ENV_FILE"
    fi
    echo "AZURE_GPU_USER=azureuser" >> "$ENV_FILE" 2>/dev/null || true
    echo "[.env] AZURE_GPU_HOST=${IP}"
fi
