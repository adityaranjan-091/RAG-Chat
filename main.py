import os
import dotenv
import getpass
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA

dotenv.load_dotenv()
# --- CONFIGURATION ---
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google API Key: ")


PDF_PATH = "Attention Is All You Need.pdf"
FAISS_INDEX_PATH = "faiss_index"

def main():
    # --- STEP 1: SETUP EMBEDDINGS ---
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vectorstore = None
    
    # --- STEP 2: CHECK IF DATABASE EXISTS ---
    if os.path.exists(FAISS_INDEX_PATH):
        print("Found existing FAISS index. Loading...")
        # Load the existing database from the folder
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    else:
        print("Creating new FAISS index from PDF...")
        # Load and split PDF only if database doesn't exist
        if not os.path.exists(PDF_PATH):
            print(f"Error: File '{PDF_PATH}' not found.")
            return

        loader = PyPDFLoader(PDF_PATH)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        # Create and Save the Vector Store
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        vectorstore.save_local(FAISS_INDEX_PATH)
        print("Index created and saved locally!")

    # --- STEP 3: SETUP CHAT CHAIN ---
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4)

    system_prompt = (
        "You are a helpful assistant. Use the provided context to answer questions. "
        "If the answer is not in the context, say you don't know."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
    )

    # --- STEP 4: CHAT LOOP ---
    print("\n--- Chat with your PDF (Type 'exit' to quit) ---")
    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit", "q"]:
            break
        
        response = rag_chain.invoke({"query": query})
        print(f"AI: {response['result']}")

if __name__ == "__main__":
    main()