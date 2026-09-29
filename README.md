# 🏦 Banking PDF RAG Chatbot

A local Retrieval-Augmented Generation (RAG) chatbot that answers banking-related questions using a banking FAQ PDF as its knowledge source.

The project uses Ollama for local language generation and embeddings, LangChain for the RAG pipeline, ChromaDB for vector storage, and Streamlit for the chatbot interface.

---

## 📌 Project Overview

The goal of this project is to build a chatbot that can retrieve relevant information from a banking FAQ document and use that information to generate an answer.

Instead of allowing the language model to answer freely, the chatbot first searches the banking FAQ knowledge base and provides the retrieved information to the language model.

This helps keep the responses grounded in the provided banking information.

---

## RAG Architecture

```text
Banking FAQ CSV
       ↓
Data Cleaning
       ↓
Cleaned FAQ Dataset
       ↓
PDF Generation
       ↓
Banking FAQ PDF
       ↓
PDF Text Extraction
       ↓
FAQ-aware Document Creation
       ↓
Ollama Embeddings
(nomic-embed-text)
       ↓
ChromaDB
       ↓
Semantic Retrieval
       ↓
Relevant FAQ Context
       ↓
Llama 3.2
       ↓
Generated Answer
       ↓
Streamlit Chatbot