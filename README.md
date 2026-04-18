<div align="center">
  <h1>Fortran → Fortran GPU + Cython</h1>
  <p><strong>Agent de transformation Fortran scientifique vers GPU (OpenACC) avec wrapper Cython.<br>
  Déployé sur Azure — Mistral-Large + LangGraph + Loki (loki-ifs).</strong></p>

  [![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)
  [![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
  [![MCP](https://img.shields.io/badge/MCP-Ready-green.svg)](https://modelcontextprotocol.io/)
</div>

---

## Overview

Pipeline multi-agents (LangGraph + Loki + Azure Mistral-Large) pour transformer du Fortran 90 scientifique en :

1. **Fortran GPU-ready** — annotations `PURE`/`ELEMENTAL` + pragmas OpenACC (`nvfortran -acc -gpu=cc80`)
2. **Wrapper Cython** — interface Python/NumPy avec typed memoryviews (`np.float64[:,:]`) pour appel depuis Python

Le pipeline utilise **Loki (loki-ifs / ECMWF)** pour l'analyse AST et la régénération du code Fortran, et **Azure Mistral-Large** pour les décisions sémantiques (PURE/ELEMENTAL, pragmas OpenACC, wrapper Cython).

---

## Architecture (Phase 1)

```
init
  ↓
parser          Loki AST — isole les routines, extrait schéma (params / statics / state)
  ↓
pure_elemental  LLM — annote PURE/ELEMENTAL sur les kernels sans effets de bord
  ↓
openacc         LLM (Loki-informed) — !$acc parallel loop / !$acc data copyin/copyout
  ↓
cython_wrapper  LLM — génère .pyx + kernel_c.h + pyproject.toml build config
  ↓
validation      nvfortran -acc + Cython build_ext
  ↓
END             output/fortran_gpu/kernel_gpu.f90  +  output/cython/module.pyx
```

---

## Quickstart

### Pré-requis

- Python 3.12+, [uv](https://github.com/astral-sh/uv)
- Loki installé localement (`./loki`) — ECMWF Fortran AST toolkit
- Azure Mistral-Large endpoint + API key (`.env`)
- `nvfortran` (NVIDIA HPC SDK) pour la compilation GPU

### Installation

```bash
cp .env.example .env
# Renseigner AZURE_MISTRAL_ENDPOINT et AZURE_MISTRAL_API_KEY

uv sync
```

### Usage CLI

```bash
# Pipeline Phase 1 — Fortran → Fortran GPU + Cython
uv run agent-gpu translate "/path/to/kernel.f90"

# Pipeline Phase 2 (JAX, expérimental)
uv run agent-pipeline translate "/path/to/kernel.f90"

# Profil de performance
uv run agent-profile "/path/to/kernel.f90"
```

### Usage MCP (IDE)

```bash
docker compose up -d --build
```

Config MCP (Antigravity / VS Code / NeoVim) :

```json
{
  "mcpServers": {
    "fortran-gpu-agent": {
      "transport": "sse",
      "url": "http://localhost:8000/sse",
      "env": { "Authorization": "Bearer your_key" }
    }
  }
}
```

Outils MCP exposés : `translate_kernel_gpu`, `translate_kernel` (JAX), `ask_agent`, `profile_kernels`.

---

## Pipeline détaillé

### 1. Parser (Loki AST)

- Parse les fichiers `.f90` avec Loki (fparser + REGEX fallback)
- Workaround pour les blocs `PROGRAM` (convertis en `SUBROUTINE`)
- Extrait le schéma global : `params`, `statics` (LOGICAL PARAMETER), `state` (arrays)
- Détecte par routine : `INTENT`, `SAVE`, boucles, I/O, dimensions de tableaux

### 2. PURE / ELEMENTAL

- LLM analyse chaque kernel : effets de bord, I/O, SAVE, COMMON
- Ajoute `PURE` ou `ELEMENTAL` aux subroutines éligibles (sans I/O, sans SAVE)
- Rend les `INTENT` explicites si nécessaire

### 3. OpenACC Insert

- Analyse Loki (boucles, INTENT IN/OUT/INOUT) fournie comme contexte au LLM
- LLM génère `!$acc parallel loop`, `!$acc loop vector`, `!$acc data copyin/copyout`
- Ajoute `!$acc routine seq` aux fonctions appelées depuis des kernels GPU
- Sauvegardé dans `output/fortran_gpu/kernel_gpu.f90`

### 4. Cython Wrapper

- LLM génère `.pyx` à partir des signatures Loki (nom, intent, types, dimensions)
- `cdef extern from "kernel_c.h"` + `cpdef` avec NumPy typed memoryviews
- `np.asfortranarray()` pour le layout mémoire Fortran (column-major)
- Génère `kernel_c.h` (C header iso_c_binding) et `pyproject.toml` (build config nvfortran)

### 5. Validation

- `nvfortran -acc -gpu=cc80 -shared -fPIC -o kernel_gpu.so kernel_gpu.f90`
- `python setup.py build_ext --inplace` (Cython)
- Rapport dans `output/fortran_gpu/validation.log`

---

## Sorties

```
output/
├── fortran_gpu/
│   ├── kernel_pure.f90      PURE/ELEMENTAL annotated
│   ├── kernel_gpu.f90       OpenACC pragmas
│   ├── kernel_gpu.so        Compiled shared library (si nvfortran disponible)
│   └── validation.log
├── cython/
│   ├── module.pyx           Cython wrapper
│   └── kernel_c.h           C header (iso_c_binding)
└── pyproject.toml           Build config (nvfortran + Cython)
```

---

## Infrastructure Azure

| VM | Type | Rôle |
|----|------|------|
| vm-orchestrator | Standard_D8s_v5 | LangGraph + Loki parsing |
| vm-gpu-a100 | Standard_NC24ads_A100_v4 | nvfortran + Cython compilation |

```bash
cd infrastructure && ./deploy.sh
```

---

## Roadmap

| Phase | Statut | Description |
|-------|--------|-------------|
| **Phase 1** | En cours | Fortran → Fortran GPU (PURE/ELEMENTAL + OpenACC + Cython) |
| **Phase 2** | Planifié | Fortran GPU → JAX (réactiver `translation_graph.py`) |
| **Phase 3** | Futur | Différentiation automatique (jax.grad) + surrogats FNO |

---

## Dépendances clés

| Paquet | Rôle |
|--------|------|
| `langgraph`, `langchain-openai` | Orchestration multi-agents |
| `loki @ file://./loki` | Parsing et transformation AST Fortran (ECMWF) |
| `fastmcp` | Serveur MCP HTTP/SSE |
| `Cython`, `numpy` | Wrapper Python/Fortran |
| `nvfortran` (NVIDIA HPC SDK) | Compilation Fortran GPU (-acc -gpu=cc80) |
| `jax[cpu]`, `flax`, `equinox` | Phase 2 — pipeline JAX (expérimental) |

---

## Licence

Propriétaire — Usage TotalEnergies Exascale.
