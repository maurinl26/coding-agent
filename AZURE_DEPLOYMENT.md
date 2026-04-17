# Guide de Déploiement : Azure pour TotalEnergies

Ce document détaille l'architecture et les prérequis pour déployer l'Agent LangGraph de traduction Fortran ➔ JAX dans un environnement sécurisé Microsoft Azure (orienté Enterprise).

## 1. Architecture Cible

Pour ce cas d'usage traitant de physiques lourdes (EDP, C-PML) et d'intelligence artificielle générative, l'architecture se divise en trois composants :

1. **Serveur Orchestrateur (Compute)** : Exécute le code LangGraph (`translation_graph.py`), parse les arbres de syntaxe avec `loki-ifs` et héberge l'interface FastMCP.
   - *Instance recommandée :* `Standard_D8s_v5` (Machine optimisée calcul à usage général).
2. **Nœud d'inférence Physique / JAX (GPU)** : Exécute la simulation JAX compilée (JIT) et entraîne le surrogate model de type Fourier Neural Operator (FNO).
   - *Instance recommandée :* `Standard_NCads_A100_v4` (qui dispose d'un GPU Nvidia A100 80GB) pour supporter de grandes dimensions spatiales.
3. **Le Cerveau IA (Le LLM Mistral)** : Assure la compréhension mathématique et la traduction via un Endpoint Serverless.
   - *Service :* **Azure AI Studio (Models as a Service)**. Modèle `Mistral-Large`.

---

## 2. Accès à Mistral : "Models-as-a-Service" (MaaS)

**Question clé pour le PoC : Ai-je besoin d'une clé Mistral en plus de mon compte Azure ?**
**Non.** Avec le partenariat Microsoft x Mistral, le modèle fait partie de l'écosystème Azure.
1. Depuis le portail Azure AI Studio, vous sélectionnez le modèle *Mistral-Large-Latest* et vous cliquez sur "Deploy".
2. Azure va créer un *Endpoint* (URL) dédié à votre souscription.
3. Azure vous fournira une **clé API Azure** spécifique pour cet Endpoint.
La facturation se fait au nombre de tokens directement sur votre facture Azure existante. Aucune donnée n'est partagée à Mistral AI en propre.

**Intégration Python (`.env`) :**
```env
AZURE_MISTRAL_ENDPOINT="https://Mistral-large-YOUR-tenant.services.ai.azure.com/v1"
AZURE_MISTRAL_API_KEY="votreclet-azure-ai-generative-23948..."
```

---

## 3. Provisionnement et CI/CD (azure-pipelines.yml)

### Différence fondamentale entre CI/CD et Provisionnement (Infrastructure as Code)
**`azure-pipelines.yml`** ne sert PAS à instancier/louer la carte graphique ou le serveur en soi.
- Son rôle est purement logiciel (CI/CD) : À chaque "Push" sur votre dépôt Git, `azure-pipelines.yml` va demander à Azure de lancer des tests unitaires, vérifier la syntaxe Python, compiler le code, et éventuellement créer une image Docker.

**Pour *Provisionner* (allumer les machines `NC-A100` et `D8s`), on utilise de l'Infrastructure-as-Code (IaC) :**
- L'idéal est de créer un fichier **Bicep** ou **Terraform** (`main.tf`).
- Ce fichier déclare formellement "TotalEnergies désire X machines virtuelles et Y bases de données dans la région France-Central".
- Il est possible d'automatiser le déploiement de cette infrastructure via `azure-pipelines.yml`, mais c'est bien le langage *Terraform* ou le portail *Azure Resource Manager (ARM)* qui loue le matériel.

---

## 4. Workflow de Déploiement Recommandé pour le PoC

1. **Validation Locale** : Testez le script sur un ordinateur local/VSCode pour vous assurer de la syntaxe JAX (ce que nous avons fait).
2. **Setup Azure AI** : Allez sur le portail Azure AI, déployez Mistral et récupérez les clés pour votre fichier `.env`.
3. **Déploiement Docker** : Poussez votre code dans un Docker Registry Azure (`ACR - Azure Container Registry`).
4. **Execution GPU** : Démarrez une instance `NC` (A100), connectez-vous y en SSH, tirez l'image Docker, et lancez la boucle FWI Surrogate.
