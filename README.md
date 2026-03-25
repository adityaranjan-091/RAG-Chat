# <center>RAG Chat</center>

A Retrieval-Augmented Generation (RAG) chatbot that lets you upload a PDF and ask questions about it. Built with **Google Gemini**, **LangChain**, **ChromaDB**, and **Streamlit**.

---

## Features

- **PDF Upload & Processing** — Upload any PDF; the app extracts text, splits it into chunks, and creates vector embeddings
- **Conversational Chat** — Ask questions and get answers grounded in the document content
- **Follow-up Questions** — Chat history is maintained so follow-ups are context-aware
- **Source Citations** — Each answer includes expandable source snippets showing which parts of the document were used
- **Local Embeddings** — Uses `all-MiniLM-L6-v2` (sentence-transformers) for embeddings — runs locally, no API cost
- **Gemini 2.5 Flash** — Uses Google's Gemini model for fast, high-quality answer generation

---

## Tech Stack

| Component        | Technology                          |
| ---------------- | ----------------------------------- |
| **LLM**          | Google Gemini 2.5 Flash             |
| **Embeddings**   | sentence-transformers (MiniLM)      |
| **Orchestration**| LangChain (LCEL)                    |
| **Vector DB**    | ChromaDB (local persistence)        |
| **UI**           | Streamlit                           |
| **PDF Parsing**  | PyPDF                               |

---

## Project Structure

```
RAG-Chat/
├── app.py              # Streamlit chat interface
├── config.py           # Centralised configuration & API key validation
├── ingestion.py        # PDF loading → text splitting → embedding → ChromaDB
├── rag_chain.py        # Conversational retrieval chain (LCEL)
├── requirements.txt    # Python dependencies
├── .env                # API key (not tracked by git)
└── .gitignore
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- A Google API key (free tier)

### 1. Get a Google API Key

1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
2. Sign in with your Google account
3. Click **Create API Key** and copy it

### 2. Clone the Repository

```bash
git clone https://github.com/<adityaranjan-091>/RAG-Chat.git
cd RAG-Chat
```

### 3. Create a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Set Up Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_api_key_here
```

### 6. Run the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Usage

1. **Upload a PDF** using the sidebar file uploader
2. Click **⚡ Process Document** — the app will read, split, and embed the PDF
3. **Ask questions** in the chat input at the bottom
4. View **source snippets** by expanding the "📚 Source Snippets" section under each answer
5. Ask **follow-up questions** — the conversation is context-aware

---

## Configuration

All settings are centralised in `config.py`:

| Setting            | Default              | Description                           |
| ------------------ | -------------------- | ------------------------------------- |
| `EMBEDDING_MODEL`  | `all-MiniLM-L6-v2`  | Sentence-transformers model for embeddings |
| `LLM_MODEL`        | `gemini-2.5-flash`   | Google Gemini model for generation    |
| `LLM_TEMPERATURE`  | `0.3`                | Controls randomness of responses      |
| `RETRIEVER_K`      | `5`                  | Number of chunks retrieved per query  |
| `CHROMA_PERSIST_DIR` | `./chroma_db`      | Directory for ChromaDB persistence    |

---

## How It Works

```
PDF Upload → PyPDF Loader → Text Splitter → Embeddings (MiniLM) → ChromaDB
                                                                       ↓
User Question → Condense with History → Retrieve Chunks → Gemini LLM → Answer
```

1. **Ingestion**: The PDF is parsed into pages, split into ~1000-character overlapping chunks, embedded using sentence-transformers, and stored in ChromaDB
2. **Retrieval**: When you ask a question, the top-k most similar chunks are retrieved from ChromaDB
3. **Generation**: The retrieved chunks are passed as context to Gemini, which generates a grounded answer
4. **History**: Follow-up questions are rephrased into standalone queries using the conversation history

---

## License

This project is open-source and available under the [MIT License](LICENSE).
