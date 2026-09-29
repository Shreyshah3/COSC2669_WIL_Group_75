import pandas as pd
from pathlib import Path

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


# ==========================================
# STEP 1: Load dataset
# ==========================================

print("STEP 1: Loading dataset...")

input_file = Path("data/processed/banking_faq_clean.csv")
df = pd.read_csv(input_file)

print(f"Total FAQs available: {len(df)}")


# ==========================================
# STEP 2: Use only 3 FAQs for testing
# ==========================================

print("\nSTEP 2: Creating 3 test documents...")

test_df = df.head(3)

documents = []

for _, row in test_df.iterrows():

    content = (
        f"FAQ ID: {row['faq_id']}\n"
        f"Section: {row['section']}\n"
        f"Question: {row['question']}\n"
        f"Answer: {row['answer']}"
    )

    documents.append(
        Document(
            page_content=content,
            metadata={
                "faq_id": int(row["faq_id"]),
                "section": row["section"],
                "source": "banking_faq.pdf"
            }
        )
    )

print(f"Test documents created: {len(documents)}")


# ==========================================
# STEP 3: Connect embeddings
# ==========================================

print("\nSTEP 3: Connecting to Ollama...")

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

print("Embedding model connected!")


# ==========================================
# STEP 4: Create test ChromaDB
# ==========================================

print("\nSTEP 4: Creating test ChromaDB...")

test_path = Path("vectorstore/test")

test_path.mkdir(parents=True, exist_ok=True)

vectorstore = Chroma(
    collection_name="banking_faq_test",
    embedding_function=embeddings,
    persist_directory=str(test_path)
)

print("ChromaDB connected!")


# ==========================================
# STEP 5: Add 3 documents
# ==========================================

print("\nSTEP 5: Adding 3 documents...")

vectorstore.add_documents(documents)

print("3 documents successfully added!")


# ==========================================
# STEP 6: Check count
# ==========================================

print("\n========== CHROMA TEST RESULTS ==========")

count = vectorstore._collection.count()

print(f"Documents in ChromaDB: {count}")


# ==========================================
# STEP 7: Test semantic search
# ==========================================

print("\nSTEP 7: Testing semantic search...")

query = "What is a savings account?"

print(f"Query: {query}")

results = vectorstore.similarity_search(
    query,
    k=2
)

print("\nRetrieved documents:")

for i, result in enumerate(results, start=1):

    print("\n--------------------------------")
    print(f"RESULT {i}")
    print("--------------------------------")

    print(result.page_content)
    print("\nMetadata:")
    print(result.metadata)


print("\nChromaDB test completed successfully!")