# MarxBot: Ask Marx Anything

A Retrieval-Augmented Generation (RAG) chatbot built with TinyLlama
via Ollama, designed to answer questions based on classic Marxist texts.

A practical demonstration of local RAG pipelines on domain-specific
document collections — the same architecture used in enterprise
knowledge management systems, applied here to political philosophy.

---

## Architecture

User query → ChromaDB vector search → relevant text chunks →
TinyLlama (local, via Ollama) → answer with source context

---

## Features

- Loads .txt, .pdf, .docx versions of Marxist works
- Fully local — no API keys, no data leaves your machine
- Vector-based retrieval for contextually relevant answers
- Same RAG pattern as production enterprise systems

---

## Getting Started

```bash
git clone https://github.com/joyboseroy/marxbot.git
cd marxbot
pip install -r requirements.txt

# Install Ollama from https://ollama.com
ollama run tinyllama

# Get texts from https://www.marxists.org/archive/marx/works/
# Place in marxtexts/ folder

python marxbot.py
```

---

## Example
