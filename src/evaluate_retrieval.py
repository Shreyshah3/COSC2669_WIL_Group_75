import pandas as pd

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


# ==========================================
# STEP 1: Connect to embeddings
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

print(
    f"Documents in database: "
    f"{vectorstore._collection.count()}"
)


# ==========================================
# STEP 3: Load cleaned dataset
# ==========================================

print("\nSTEP 3: Loading FAQ dataset...")

df = pd.read_csv(
    "data/processed/banking_faq_clean.csv"
)

print(f"FAQs available: {len(df)}")


# ==========================================
# STEP 4: Select test questions
# ==========================================

test_data = df.sample(
    n=10,
    random_state=42
)


# ==========================================
# STEP 5: Evaluate retrieval
# ==========================================

print("\n========== RETRIEVAL EVALUATION ==========")

correct = 0

total = len(test_data)

for _, row in test_data.iterrows():

    faq_id = int(row["faq_id"])
    question = row["question"]

    results = vectorstore.similarity_search(
        question,
        k=3
    )

    retrieved_ids = [
        doc.metadata.get("faq_id")
        for doc in results
    ]

    is_correct = (
        faq_id == retrieved_ids[0]
    )

    if is_correct:
        correct += 1

    print("\n----------------------------------------")

    print(f"Question: {question}")

    print(f"Expected FAQ ID: {faq_id}")

    print(
        f"Retrieved FAQ IDs: "
        f"{retrieved_ids}"
    )

    if is_correct:
        print("Result: CORRECT")
    else:
        print("Result: INCORRECT")


# ==========================================
# STEP 6: Calculate accuracy
# ==========================================

accuracy = (
    correct / total
) * 100


print("\n========================================")
print("RETRIEVAL EVALUATION RESULTS")
print("========================================")

print(f"Total test questions: {total}")

print(f"Correct top-1 retrievals: {correct}")

print(
    f"Top-1 retrieval accuracy: "
    f"{accuracy:.2f}%"
)

print("========================================")

print("\nEvaluation completed successfully!")