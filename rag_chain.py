from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma

from config import GOOGLE_API_KEY, LLM_MODEL, LLM_TEMPERATURE, RETRIEVER_K

# ---------------------------------------------------------------------------
# Custom prompts
# ---------------------------------------------------------------------------

# Prompt to rephrase a follow-up question into a standalone question
CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given the following conversation and a follow-up question, "
            "rephrase the follow-up question to be a standalone question. "
            "If the follow-up question is already standalone, return it as-is.",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

# Main QA prompt — instructs the model to only use provided context
QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful AI assistant. Use ONLY the following context "
            "extracted from the uploaded document to answer the question. "
            "If the answer is not in the context, say "
            '"I couldn\'t find that information in the uploaded document."\n\n'
            "Context:\n{context}\n\n"
            "Instructions:\n"
            "- Answer clearly and concisely.\n"
            "- If relevant, mention which part of the document the information comes from.\n"
            "- Use bullet points or numbered lists for complex answers.",
        ),
        ("human", "{question}"),
    ]
)


def _format_docs(docs: list) -> str:
    """Join document page_content with double newlines."""
    return "\n\n".join(doc.page_content for doc in docs)


def get_chain(vectorstore: Chroma):
    """
    Build and return a conversational retrieval chain using LCEL.

    Parameters
    ----------
    vectorstore : Chroma
        The ChromaDB vectorstore loaded with document chunks.

    Returns
    -------
    A callable that accepts {"question": str, "chat_history": list}
    and returns {"answer": str, "source_documents": list}.
    """
    # Initialise Gemini as the LLM
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        google_api_key=GOOGLE_API_KEY,
    )

    # Build a retriever from the vectorstore
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K},
    )

    # Chain to condense follow-up questions
    condense_chain = CONDENSE_PROMPT | llm | StrOutputParser()

    def process_query(inputs: dict) -> dict:
        """
        Full RAG pipeline:
        1. Condense question using chat history (if any)
        2. Retrieve relevant docs
        3. Generate answer from context
        """
        question = inputs["question"]
        chat_history = inputs.get("chat_history", [])

        # Step 1: If there's chat history, condense the question
        if chat_history:
            standalone_question = condense_chain.invoke(
                {"question": question, "chat_history": chat_history}
            )
        else:
            standalone_question = question

        # Step 2: Retrieve relevant documents
        source_docs = retriever.invoke(standalone_question)

        # Step 3: Generate the answer
        context = _format_docs(source_docs)
        answer_response = (QA_PROMPT | llm | StrOutputParser()).invoke(
            {"context": context, "question": standalone_question}
        )

        return {
            "answer": answer_response,
            "source_documents": source_docs,
        }

    return process_query
