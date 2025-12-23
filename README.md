# Code-Archaeologist
# Code Archaeologist: Autonomous ReAct Agent with RAG & Memory

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Model](https://img.shields.io/badge/Model-Deep--100%20(Unsloth)-orange)
![Architecture](https://img.shields.io/badge/Architecture-ReAct-green)
![Status](https://img.shields.io/badge/Status-Experimental-yellow)

**Code-Archaeologist** is an advanced autonomous agent capable of writing, executing, and fixing Python code. Built upon a fine-tuned **Deep-100 (DeepSeek/Qwen2.5)** model, it utilizes the **ReAct (Reasoning + Acting)** architecture to solve complex problems by orchestrating tools, maintaining memory context, and retrieving domain knowledge via RAG.

##  Key Features

* ** ReAct Architecture:** Implements the *Thought → Action → Observation* loop to reason through problems.
* ** Tool Use & Execution:** Writes Python code and executes it in a controlled sandbox environment using `subprocess`.
* ** RAG (Retrieval-Augmented Generation):** Consults a knowledge base (Vector DB using `sentence-transformers`) for theoretical questions.
* ** Dynamic Memory:** Maintains conversation history (Context Awareness) to handle follow-up questions.
* ** Self-Correction & Safety:**
    * Detects and fixes execution errors (e.g., `NameError`, `SyntaxError`).
    * Enforces safety constraints (blocks `sys`, `input`, and infinite loops via timeouts).
    * "Smart Regex" parsing to handle unstructured model outputs.

##  Architecture

1.  **Orchestrator:** The Deep-100 LLM generates thoughts and decides which tool to use.
2.  **Tools:**
    * `python_executor`: Runs code snippets.
    * `rag_retrieval`: Searches semantic embeddings of documentation.
3.  **Safety Layer:** Filters malicious imports and enforces execution timeouts.

##  Installation & Usage

This project is optimized for Google Colab (using T4 GPU) due to `unsloth` dependencies.

### Prerequisites
* Python 3.10+
* NVIDIA GPU (T4 or better)
* Google Colab (Recommended)

### Installation
```bash
pip install "unsloth[colab-new] @ git+[https://github.com/unslothai/unsloth.git](https://github.com/unslothai/unsloth.git)"
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
pip install sentence-transformers
