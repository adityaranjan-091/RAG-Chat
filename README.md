# RAG-Chat

A simple yet powerful Retrieval-Augmented Generation (RAG) application that allows you to chat with your PDF documents using Google's Gemini Pro model.

## Features

-   **PDF Parsing**: Automatically loads and splits PDF documents.
-   **Vector Search**: Uses FAISS (Facebook AI Similarity Search) for efficient similarity search.
-   **Google Gemini Integration**: Leverages `gemini-2.5-flash` for high-quality, fast responses and `text-embedding-004` for embeddings.
-   **Persistent Index**: Saves the FAISS index locally to avoid reprocessing the PDF every time.
-   **Context-Aware**: Retrieves relevant context from the PDF to answer user queries accurately.
-   **CLI Interface**: Simple and direct command-line interface for chatting.

## Prerequisites

-   Python 3.8+
-   A Google Cloud API Key with access to Gemini API