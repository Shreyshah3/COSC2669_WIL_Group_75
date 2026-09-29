from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma


# ==========================================
# STEP 1: Connect to embeddings
# ==========================================

print("Connecting to embedding model...")

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# ==========================================
# STEP 2: Load ChromaDB
# ==========================================

print("Loading banking FAQ database...")

vectorstore = Chroma(
    collection_name="banking_faq",
    embedding_function=embeddings,
    persist_directory="vectorstore/banking_faq"
)

print(
    f"Database loaded: "
    f"{vectorstore._collection.count()} FAQs"
)


# ==========================================
# STEP 3: Create retriever
# ==========================================

retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 3
    }
)


# ==========================================
# STEP 4: Connect Llama
# ==========================================

print("Connecting to Llama 3.2...")

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

print("\n========================================")
print("      BANKING PDF RAG CHATBOT")
print("========================================")
print("Ask questions about the banking FAQ.")
print("Type 'exit' to quit.")
print("========================================\n")


# ==========================================
# STEP 5: Chat loop
# ==========================================

while True:

    question = input("\nYou: ").strip()

    if question.lower() == "exit":
        print("\nGoodbye!")
        break

    if not question:
        continue


    # ======================================
    # Retrieve relevant FAQs
    # ======================================

    retrieved_docs = retriever.invoke(
        question
    )


    # ======================================
    # Build context
    # ======================================

    context = "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )


    # ======================================
    # Build RAG prompt
    # ======================================

    prompt = f"""
You are a helpful banking FAQ assistant.

Answer the user's question using ONLY the
information provided in the context.

If the answer cannot be found in the context,
say:

"I could not find this information in the banking FAQ."

Do not invent information.
Do not use outside knowledge.

Context:
{context}

User question:
{question}

Answer:
"""


    # ======================================
    # Generate answer
    # ======================================

    response = llm.invoke(prompt)


    print("\nBot:")
print(response.content)

print("\nSources:")

for doc in retrieved_docs:
    print(
        f"- FAQ ID {doc.metadata.get('faq_id')} "
        f"| Section: {doc.metadata.get('section')} "
        f"| Source: {doc.metadata.get('source')}"
    )


print("\nChatbot stopped.")