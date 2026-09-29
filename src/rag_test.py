from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma


# ==========================================
# STEP 1: Connect to embedding model
# ==========================================

print("STEP 1: Connecting to embedding model...")

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

print("ChromaDB loaded!")

print(
    f"Documents in database: "
    f"{vectorstore._collection.count()}"
)


# ==========================================
# STEP 3: Create retriever
# ==========================================

print("\nSTEP 3: Creating retriever...")

retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 4
    }
)

print("Retriever created!")


# ==========================================
# STEP 4: Connect Llama 3.2
# ==========================================

print("\nSTEP 4: Connecting to Llama 3.2...")

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

print("Llama 3.2 connected!")


# ==========================================
# STEP 5: Ask a question
# ==========================================

question = "What is a fixed deposit?"

print("\n========== USER QUESTION ==========")
print(question)


# ==========================================
# STEP 6: Retrieve relevant documents
# ==========================================

print("\nSTEP 6: Retrieving relevant FAQs...")

retrieved_docs = retriever.invoke(question)

print(
    f"Retrieved documents: "
    f"{len(retrieved_docs)}"
)


for i, doc in enumerate(
    retrieved_docs,
    start=1
):

    print("\n----------------------------------------")
    print(f"RETRIEVED DOCUMENT {i}")
    print("----------------------------------------")

    print(doc.page_content)

    print("\nMetadata:")
    print(doc.metadata)


# ==========================================
# STEP 7: Build context
# ==========================================

print("\nSTEP 7: Building context...")

context = "\n\n".join(
    doc.page_content
    for doc in retrieved_docs
)

print("Context created!")


# ==========================================
# STEP 8: Create RAG prompt
# ==========================================

prompt = f"""
You are a helpful banking FAQ assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer cannot be found in the context, say:
"I could not find this information in the banking FAQ."

Do not invent or assume information.

Context:
{context}

User question:
{question}

Answer:
"""


# ==========================================
# STEP 9: Generate answer
# ==========================================

print("\nSTEP 8: Sending context to Llama 3.2...")

response = llm.invoke(prompt)

print("\n========== FINAL RAG ANSWER ==========")

print(response.content)

print("\nRAG test completed successfully!")