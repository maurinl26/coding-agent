<div align="center">
  <img src="https://raw.githubusercontent.com/modelcontextprotocol/logo/main/mcp-logo.svg" width="120" alt="MCP Logo" />
  <h1>🤖 Local Code Agent SaaS</h1>
  <p><strong>A 100% Private, Secure, and Agentic Coding Assistant powered by Mistral NeMo 12B and the Model Context Protocol (MCP).</strong></p>
  
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
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

### 3. Connect your IDE (Antigravity)

Open your `mcp_config.json` file in Antigravity or Claude Desktop:

```json
{
  "mcpServers": {
    "local-saas-agent": {
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

---

## 🛒 Client Portal (SaaS Billing)

This repository also includes a **Next.js + Vercel + Stripe** billing portal in the `client-portal/` directory, allowing you to monetize access to your hosted MCP agent securely.

```bash
cd client-portal
npm install
npm run dev
```

---

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
