# Connoisseur Companion: MCP Subsystem & Agent Host

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Model Context Protocol](https://img.shields.io/badge/Protocol-MCP%20FastMCP-orange.svg)](https://modelcontextprotocol.io/)
[![LLM Foundation](https://img.shields.io/badge/LLM-IBM%20Granite%20%7C%20Claude-black.svg)](https://www.ibm.com/granite)

---

## 📌 Subsystem Overview

This folder contains the core **Model Context Protocol (MCP)** implementation and **Gradio Agent Host** for the Connoisseur Companion platform.

- **`server.py`**: FastMCP server exposing structured restaurant search (`get_restaurant_info`), semantic vibe discovery (`recommend_by_vibe`), full review retrieval (`get_review`), and the California Culinary Map resource (`culinary-map://california`).
- **`client.py`**: MCP client managing stdio transport, dynamic tool and resource discovery, directory root boundaries, and server sampling via Anthropic Claude.
- **`app.py`**: MCP Host and interactive Gradio chat interface featuring an autonomous ReAct agent loop powered by IBM Granite (`ibm/granite-4-h-small` on Watsonx.ai).
- **`restaurant_data_management.py`**: Data validation pipeline using Pydantic v2 and LLM self-healing auto-repair loops with full unit test coverage.
- **`test.py`**: Automated end-to-end integration script to verify MCP server connectivity and tool execution.

---

## 🖥️ Running the Application

### 1. Test MCP Server & Tools
```bash
python test.py
```

### 2. Verify Client Discovery & Sampling
```bash
python client.py
```

### 3. Launch Gradio Web Application
```bash
python app.py
```

### 4. Run Unit Tests
```bash
python -m unittest restaurant_data_management.py
```
