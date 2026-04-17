# 🧪 Plan de Recette Manuelle (UAT) : Pipeline `seismic_cpml`

Ce document définit les étapes pas-à-pas pour la validation ("recette") du pipeline **Fortran ➔ JAX ➔ Surrogate** exécuté sur votre environnement local ou cloud.

## Prérequis de la Recette
- Environnement CLI avec `uv` et Docker (Mistral) ou `.env` avec Endpoint Azure.
- Cloner le dosier contenant les codes source cibles : `seismic_cpml`.
- **Fichier de test cible** : `seismic_CPML_2D_isotropic_second_order.f90`

---

## 🏃 Étape 1 : Déclenchement du Pipeline
Lancer le pipeline LangGraph complet en pointant vers le code source.
```bash
uv run agent-pipeline translate "/path/to/seismic_cpml/seismic_CPML_2D_isotropic_second_order.f90"
```

> [!IMPORTANT]
> **Ce qu'il faut vérifier (CLI Output) :** 
> - Assurez-vous que l'authentification MaaS Azure `ChatMistralAI` ne rejette pas l'appel (code HTTP 401). Le pipeline doit démarrer la séquence en affichant l'appel à *Loki*.

---

## 🔍 Étape 2 : Validation du Parser (Loki)
*(Nœud `parse_and_isolate_agent`)*

Le but de cette étape est de s'assurer que le compilateur dégage le "bruit" IT pour garder la physique pure.
- `[ ]` **Extraction Isolée** : Le fichier généré ne doit comporter que le "Kernel" numérique ou les boucles temporelles principales (`do it = 1, NSTEP`). 
- `[ ]` **AST Info & Hints** : En lisant les logs, vérifiez que l'agent Loki a correctement détecté les constantes de dimension comme `NX`, `NY` et `NSTEP`. Celles-ci seront cruciales pour l'opérateur neuronal plus tard.

---

## 🐍 Étape 3 : Validation de la Traduction JAX
*(Nœud `translate_kernel_agent`)*

Le passage du C++/Fortran au paradigme Tensoriel de JAX s'inscrit en rupture totale. Le code JAX produit doit respecter les normes de programmation fonctionnelle pure.
- `[ ]` **Fin des boucles spatiales `do i`, `do j`** : Le code Python généré DOIT utiliser soit le slicing de matrices NumPy `array[1:-1, ...]`, soit `jax.numpy.roll` pour calculer les différences finies spatiales des vitesses et contraintes.
- `[ ]` **Gestion Temporelle** : Les boucles temporelles strictes ne peuvent pas être vectorisées. Le code doit idéalement faire état d'un contexte de type `jax.lax.scan` pour faire évoluer l'état `(vx, vy, sigma_xx)` d'un timestep $t$ à $t+1$.

---

## 🔄 Étape 4 : Validation de l'Adjoint JAX (Autodiff)
*(Nœud `autodiff_adjoint_agent`)*

C'est l'essence du JIT paramétrique exploité par TotalEnergies.
- `[ ]` **Présence de `jax.vjp` ou `jax.grad`** : Le code retourné doit contenir, en fin de fichier, une fonction (souvent appelée `adjoint_kernel` ou `backward_step`), qui appelle explicitement le Reverse-Mode Autodiff pour extraire la sensibilité du sismogramme de sortie vis-à-vis en fonction des paramètres géologiques d'entrée ($V_p$ / vitesse de l'onde P).

---

## 🧠 Étape 5 : Validation FNO Surrogate
*(Nœud `surrogate_fno_agent`)*

Il s'agit de vérifier le script DL Deepmind.
- `[ ]` **Paramétrage ciblé** : Ouvrez le fichier de "Surrogate". Assurez-vous que la classe est définie avec le bon framework. Si votre config demande `flax`, vous devez voir `class FNO2d(nn.Module)`. Si `equinox`, vous devez voir `class FNO2d(eqx.Module)`.
- `[ ]` **Prise en compte des dimensions (AST)** : Le modèle possède-t-il les bonnes résolutions spatiales internes ? Si le `NX` du fichier Fortran était `101`, le constructeur du réseau doit idéalement indiquer ces projections pour valider l'information passée du premier au dernier agent de la chaîne !
- `[ ]` **Boucle d'Entraînement de Sobolev** : La `loss_function` doit additionner deux erreurs :
  1. `MSE(preds, vrais_sismogrammes)`
  2. `MSE(grad_preds, backward_step())`  *(L'apprentissage du gradient).*

---

## 📚 Étape 6 : Validation de la Documentation (Docstring)
*(Nœud `docstring_agent`)*

Vérification finale du packaging du code pour les équipes M&D et géosciences :
- `[ ]` Le Docstring python doit inclure au moins une mention explicite à "Virieux" (pour le stangerred-grid) ou "Komatitsch" / "PML" déduite automatiquement du contexte Fortran.
- `[ ]` Vérifiez si les équations aux dérivées partielles élastiques sont re-traduites de façon propre dans le commentaire.

---
> [!TIP]
> Si toutes ces cases sont cochées après le run local, le pipeline LangGraph peut officiellement être déployé en production (`main branch`) pour le traitement à l'échelle des milliers de fichiers du dépôt HPC TotalEnergies.
