# Support KB Agent

A RAG-powered support agent I threw together using FAISS, LangGraph, and Gemini.

## What's this?

Basically, I wanted to build a simple knowledge base agent that could answer support questions. Nothing crazy - just load some docs, chunk them up, index them, and let an LLM answer questions based on what's in there.

Here's what it does:
- Reads markdown and text files from a folder
- Chunks them into bite-sized pieces and stores embeddings in FAISS
- Uses LangGraph to handle the retrieve → generate flow

## Getting started

```bash
# create a virtual env
python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on windows

# install deps
pip install -r requirements.txt

# set up your api key
cp .env.example .env
# then edit .env and add your GEMINI_API_KEY
```

## How to use it

```bash
# make sure venv is active
source venv/bin/activate

# first, index your docs (do this once, or whenever you add new docs)
python -m src.main ingest

# ask questions!
python -m src.main query "How do I get a refund?"
python -m src.main query "What shipping options do you have?"

```

## Folder structure

```
support-kb-agent/
├── src/
│   ├── config.py       # env vars and settings
│   ├── loader.py       # loading and chunking docs
│   ├── store.py        # faiss vector store stuff
│   ├── graph.py        # the langgraph workflow
│   └── main.py         # cli entrypoint
├── data/               # drop your docs here
├── index/              # where the faiss index lives
├── requirements.txt
└── .env.example
```

## Some design notes

### TF-IDF over neural embeddings

Yeah I know, TF-IDF is "old school". But I kept hitting Gemini API rate limits during dev and it was driving me crazy. TF-IDF works offline and honestly does a decent job for keyword-heavy support docs. Would definitely switch to sentence-transformers or similar for anything production-grade though.

### Why FAISS?

It's simple and fast. Looked at ChromaDB but FAISS needed less setup for something this small. No regrets so far.

### The LangGraph flow

Super minimal - just retrieve and generate nodes. Could add reranking or query rewriting but I wanted to keep the first version simple.

### Chunking

500 char chunks with 50 char overlap. Not fancy, just practical.

## Example

```
🔍 What is the refund policy?
============================================================
Based on the knowledge base, refunds are processed within 5-7 
business days after we receive your returned item. Reach out to 
support@example.com with your order number to start a return.

---
Sources: sample_policy.txt
```

## What's next

- [ ] Use actual embeddings when I get more API quota
- [ ] Add a reranking step
- [ ] Conversation memory so it remembers context
- [ ] Better error handling (it's pretty rough right now)
