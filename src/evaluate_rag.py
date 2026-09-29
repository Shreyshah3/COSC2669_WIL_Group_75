from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma


# ==========================================
# STEP 1: Connect to embeddings
# ==========================================

print("STEP 1: Connecting to embeddings...")

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

print("Embedding model connected!")


# ==========================================
# STEP 2: Load ChromaDB
# ==========================================

print("\nSTEP 2: Loading ChromaDB...")

vectorstore = Chroma(
    collection_name="banking_faq",
    embedding_function=embeddings,
    persist_directory="vectorstore/banking_faq"
)

print(
    f"Documents in database: "
    f"{vectorstore._collection.count()}"
)


# ==========================================
# STEP 3: Connect Llama
# ==========================================

print("\nSTEP 3: Connecting to Llama 3.2...")

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

print("Llama 3.2 connected!")


# ==========================================
# STEP 4: Create retriever
# ==========================================

retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 4
    }
)


# ==========================================
# STEP 5: Test questions
# ==========================================

questions = [
    "What is a fixed deposit?",
    "What does KYC mean?",
    "How do I block my debit card?",
    "What is a savings account?",
    "What is the capital of Australia?"
]


# ==========================================
# STEP 6: Test RAG
# ==========================================

print("\n========== RAG ANSWER EVALUATION ==========")


for number, question in enumerate(
    questions,
    start=1
):

    print("\n")
    print("=" * 60)
    print(f"QUESTION {number}")
    print("=" * 60)

    print(f"User: {question}")


    # --------------------------------------
    # Retrieve documents
    # --------------------------------------

    retrieved_docs = retriever.invoke(
        question
    )


    print("\nRetrieved FAQ IDs:")

    print([
        doc.metadata.get("faq_id")
        for doc in retrieved_docs
    ])


    # --------------------------------------
    # Build context
    # --------------------------------------

    context = "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )


    # --------------------------------------
    # Build prompt
    # --------------------------------------

    prompt = f"""
You are a banking FAQ assistant.

Answer the user's question using ONLY the
information in the provided context.

If the answer cannot be found in the context,
respond exactly:

I could not find this information in the banking FAQ.

Do not use outside knowledge.
Do not invent information.

Context:
{context}

User question:
{question}

Answer:
"""


    # --------------------------------------
    # Generate answer
    # --------------------------------------

    response = llm.invoke(
        prompt
    )


    print("\nAnswer:")

    print(response.content)


print("\n")
print("=" * 60)
print("RAG ANSWER EVALUATION COMPLETED")
print("=" * 60)