# 🚀 Tutoriel : Mode Interactif (Effet Whaou)

Ce tutoriel explique comment utiliser le mode manuel du pipeline Fortran-to-JAX pour modifier le code généré directement dans votre IDE avant la suite du traitement.

## 1. Activation du mode
Par défaut, le pipeline tourne en mode "Auto" (CI/CD). Pour activer le mode interactif, définissez la variable d'environnement suivante dans votre terminal :

```bash
export AGENT_INTERACTION_MODE=manual
```

## 2. Lancement du Pipeline
Lancez la traduction comme d'habitude :

```bash
uv run agent-pipeline translate "chemin/vers/votre/kernel.f90"
```

## 3. Déroulement de l'interaction
1.  **Génération** : L'agent traduit le kernel Fortran en JAX.
2.  **Ouverture IDE** : Une fois la traduction terminée, l'agent ouvre automatiquement le fichier `src/<kernel>/jax_kernel.py` dans votre éditeur par défaut (VSCode, PyCharm, etc.).
3.  **Modification** : Vous pouvez alors modifier le code JAX (ex: changer une boucle, ajouter des commentaires, optimiser une opération).
4.  **Validation** : Revenez dans votre terminal. L'agent affiche : `👉 [ACTION REQUIRED]`.
5.  **Continuité** : Appuyez sur **ENTREE**. L'agent re-lit votre fichier modifié et l'utilise pour les étapes suivantes :
    *   Analyse HPC (Halo Exchange)
    *   Validation numérique (Reproducibility)
    *   Benchmarks (Performance)

## 4. Mode CI/CD
Pour désactiver l'interaction et revenir au mode automatique :

```bash
export AGENT_INTERACTION_MODE=auto
```

> [!TIP]
> Ce mode est idéal pour les kernels complexes où l'expertise humaine est nécessaire pour affiner la vectorisation JAX avant de lancer les benchmarks de performance.
