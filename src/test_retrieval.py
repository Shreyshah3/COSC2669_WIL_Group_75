from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


# ==========================================
# STEP 1: Connect to embedding model
# ==========================================

print("STEP 1: Connecting to Ollama embeddings...")

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

print("Embedding model connected!")


# ==========================================
# STEP 2: Load existing ChromaDB
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
# STEP 3: Test questions
# ==========================================

test_questions = [
    "Where can I keep my money and earn interest?",
    "How can I verify my identity with the bank?",
    "What should I do if I want to stop using my bank card?",
    "What is a fixed deposit?",
    "How can I transfer money to another person?"
]


# ==========================================
# STEP 4: Perform semantic retrieval
# ==========================================

print("\n========== RETRIEVAL TEST ==========")

for number, question in enumerate(test_questions, start=1):

    print("\n")
    print("=" * 60)
    print(f"QUESTION {number}")
    print("=" * 60)

    print(f"User question: {question}")

    results = vectorstore.similarity_search(
        question,
        k=3
    )

    print("\nTop 3 retrieved FAQs:")

    for rank, result in enumerate(results, start=1):

        print("\n----------------------------------------")
        print(f"RESULT {rank}")
        print("----------------------------------------")

        print(result.page_content)

        print("\nMetadata:")
        print(result.metadata)


print("\n")
print("=" * 60)
print("RETRIEVAL TEST COMPLETED SUCCESSFULLY!")
print("=" * 60)