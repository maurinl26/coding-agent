<div align="center">
  <img src="https://raw.githubusercontent.com/modelcontextprotocol/logo/main/mcp-logo.svg" width="120" alt="MCP Logo" />
  <h1>🤖 Local Code Agent: Fortran ↔ JAX HPC</h1>
  <p><strong>Proprietary Evaluation Agent for TotalEnergies Exascale translation. Scaled via LangGraph, Loki parsing, and GHEX MPI abstractions.</strong></p>
  
  [![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)
  [![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
  [![MCP](https://img.shields.io/badge/MCP-Ready-green.svg)](https://modelcontextprotocol.io/)
  
  <br />
</div>

## 🌟 Overview

**Local Code Agent SaaS** is a powerful, self-hosted AI coding assistant designed for enterprise environments where **code privacy is paramount**. 

Instead of sending your proprietary source code to third-party cloud APIs (like OpenAI or Anthropic), this project spins up a **sandbox Docker container** that orchestrates a local instance of **Mistral NeMo 12B** via Ollama. It exposes its capabilities through a secure **FastMCP HTTP/SSE Server**, allowing seamless integration into modern IDEs supporting the Model Context Protocol (MCP).

### ✨ Features
* **🔒 Zero Data Leakage**: Your code never leaves your infrastructure. 100% private.
* **🧠 Mistral NeMo 12B**: State-of-the-art open-weights model quantization (4-bit) running smoothly on consumer hardware (e.g., Apple Silicon M-series with 16GB+ RAM).
* **🔌 MCP Native**: Out-of-the-box compatibility with **Antigravity**, **Claude Desktop**, **VS Code** (via Roo Code / Cline), and **NeoVim**.
* **🛡️ Hosted Workspace Ready**: Implements IDE-driven file readers—the AI requests the IDE to read/write files rather than having direct disk access, enabling advanced DevContainer isolation.
* **🔑 API Key Security**: Built-in authentication middleware to protect your MCP server if exposed to the public internet.

---

## 🏗️ Architecture

```mermaid
graph LR
    A[IDE Client\nAntigravity/VSCode] -- "MCP Protocol\nHTTP/SSE" --> B(Reverse Proxy / Cloudflare)
    B -- "Bearer Auth" --> C[Docker Sandbox\nFastMCP Server]
    C -- "ReAct Agent\nLangChain" --> D[(Ollama\nMistral NeMo 12B)]
    C -. "Tool Requests\n(read_file)" .-> A
```

---

## 🚀 Quickstart

### 1. Requirements
* [Docker](https://www.docker.com/) & Docker Compose (or [OrbStack](https://orbstack.dev/) recommended for Mac)
* Python 3.12+ (for local dev only)

### 2. Run the Infrastructure

Clone the repository and set up your secure API key:

```bash
git clone https://github.com/yourusername/coding-agent.git
cd coding-agent

# Set up your environment variables
cp .env.example .env
# Edit .env and set API_KEY=your_secure_password
```

Spin up the agent sandbox and Ollama:
```bash
docker compose up -d --build
```
*(On the first run, the Mistral NeMo 12B model weighting ~7GB will be pulled automatically).*

### 3. Connect your IDEs

#### Antigravity Setup
Open your `mcp_config.json` file in Antigravity (or use the visual MCP settings menu):

```json
{
  "mcpServers": {
    "fortran-jax-agent": {
      "command": "node",
      "args": ["-e", "console.log('Connecting via SSE')"],
      "transport": "sse",
      "url": "http://localhost:8000/sse",
      "env": {
        "Authorization": "Bearer your_secure_password"
      }
    }
  }
}
```

#### LazyVIM Setup
To use these agents within LazyVIM, you need an MCP-compatible plugin such as `mcp.nvim` (or `neo-mcp`). Add the following to your Neovim Lua config (e.g., `lua/plugins/mcp.lua`):

```lua
return {
  {
    "your-mcp/plugin.nvim",
    opts = {
      servers = {
        fortran_jax = {
          protocol = "sse",
          endpoint = "http://localhost:8000/sse",
          headers = {
            ["Authorization"] = "Bearer your_secure_password",
          }
        }
      }
    }
  }
}
```
Reload LazyVIM. You can now execute MCP tools like `:MCPCall translate_kernel filepath=%` directly from an open `.f90` buffer!

---

## 🛒 Distribution Strategy (SaaS)

If you intend to distribute these scientific translation agents to external clients (B2B research centers), the repository provides two main distribution architectures depending on their security needs.

### 1. Cloud-Hosted SaaS (Zero-Setup Approach)
Using the provided `client-portal/` (Next.js + Stripe), you can host the FastMCP server (`server.py`) and the LangGraph orchestrator centrally. 
- **The Flow**: Clients subscribe via the portal -> They receive a unique API Key -> They configure their Antigravity/LazyVIM to point to your secure Cloudflare tunnel `https://mcp.yourdomain.com/sse`.
- **Pros**: Client has zero compute cost; instant updates to the agent logic.
- **Cons**: Client's Fortran code must transit over the internet to your server.

### 2. On-Premise Docker Enterprise Deployment (High-Security Approach)
For institutions that cannot allow source code to leave their localized network.
- **The Flow**: Clients purchase a license through the portal -> They unlock access to a private Docker Registry -> They pull a pre-compiled `ghcr.io/your-brand/fortran-agents-engine` image -> They run it locally.
- **Pros**: 100% Data Privacy (Zero Data Leakage). 
- **Cons**: Client needs capable local hardware (or local GPU clusters) to run the Ollama LLM and the profiling.

```bash
cd client-portal
npm install
npm run dev
```

---

## 🔬 Scientific Porting: Fortran to JAX (seismic_cpml)

This repository hosts a dedicated multi-agent pipeline designed to automatically translate legacy Fortran 90 simulations (`seismic_cpml`) into high-performance JAX (Python) modules.

### The Agent Workflow (LangGraph)
1. **Parser Agent**: Extracts pure numerical kernels from `.f90` files using ECMWF `loki-ifs`. *(MPI and Halos are delegated down the line)*
2. **Translation Agent (LLM)**: Analyzes Fortran syntax and maps it to specific JAX vectorization patterns (`jax.lax.scan`, `jax.vmap`).
3. **Docstring Agent (LLM)**: Inspects the literature context, links the original `.f90` file, infers the research paper, and explicitly writes the exact mathematical formula being calculated inside the python docstrings.
4. **Reproducibility Agent**: Generates `f2py` testing harnesses and validates the JAX output against the native Fortran binaries using `np.testing.assert_allclose`.
5. **Performance Agent**: Evaluates generated JAX kernels against the Fortran execution on CPU vs GPU bounds.

### 🧪 How to test the translation agents right now?

We designed the pipeline to be exposed seamlessly either locally via your IDE, or automatically in GitHub PRs.

**Method 1: Interactive IDE Testing (Antigravity/LazyVIM)**
If you are connected to the FastMCP server, the agent exposes two direct tools:
- `translate_kernel(filepath: str)`: Supply the absolute path to a Fortran file, and the multi-agent graph will process it and return the pure JAX code.
- `profile_kernels(filepath: str)`: Compare an existing Fortran/JAX pair for performance benchmarks.

**Method 2: Command Line Interface (CLI) via UV**
We integrated standard pyproject configuration. Using the fast `uv` package manager, you can trigger individual agents natively:

```bash
# Execute the full translation pipeline (Parser -> Translator -> Docstring -> Tests -> Perf)
uv run agent-pipeline translate "/path/to/seismic_CPML_2D_isotropic_second_order.f90"

# Or trigger isolated agents directly:
uv run agent-translate "/path/to/seismic_CPML_2D_isotropic_second_order.f90"
uv run agent-profile "/path/to/seismic_CPML_2D_isotropic_second_order.f90"
```

**Method 3: GitHub Actions (Automated PRs)**
The repository comes equipped with `.github/workflows/jax-translation.yml` and a specific `Dockerfile.ci`.
Simply commit a new `.f90` file to a branch and push. The CI will trigger the extraction agents and automatically inject the `_jax.py` results into your PR.

---

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
