import os
import tempfile
from typing import BinaryIO

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from config import (
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)

# ---------------------------------------------------------------------------
# Text splitter — tuned for ~1 000-char chunks with 200-char overlap so
# that context is never cut off mid-sentence across chunk boundaries.
# ---------------------------------------------------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return configured HuggingFace sentence-transformers embeddings (runs locally)."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
    )


def ingest_pdf(uploaded_file: BinaryIO, filename: str = "upload.pdf") -> Chroma:
    """
    Process an uploaded PDF and store its chunks in ChromaDB.

    Parameters
    ----------
    uploaded_file : BinaryIO
        A file-like object (e.g. from Streamlit's file_uploader).
    filename : str
        Original filename — used only for metadata.

    Returns
    -------
    Chroma
        A LangChain Chroma vectorstore ready for retrieval.
    """
    # --- Step 1: Write uploaded bytes to a temp file so PyPDFLoader can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        # --- Step 2: Load pages from the PDF
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()

        if not pages:
            raise ValueError("The PDF appears to be empty or unreadable.")

        # Add source metadata to each page
        for page in pages:
            page.metadata["source"] = filename

        # --- Step 3: Split pages into smaller chunks
        chunks = text_splitter.split_documents(pages)

        # --- Step 4: Embed and store in ChromaDB
        embeddings = get_embeddings()
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
            collection_name=COLLECTION_NAME,
        )

        return vectorstore

    finally:
        # Clean up temp file
        os.unlink(tmp_path)


def load_existing_vectorstore() -> Chroma | None:
    """
    Load the existing persisted ChromaDB vectorstore (if any).

    Returns None if the persistence directory does not exist yet.
    """
    if not os.path.exists(CHROMA_PERSIST_DIR):
        return None

    embeddings = get_embeddings()
    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
