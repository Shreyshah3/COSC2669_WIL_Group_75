from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


# ==========================================
# STEP 1: Load PDF
# ==========================================

pdf_path = Path("data/pdf/banking_faq.pdf")
vectorstore_path = Path("vectorstore/banking_faq")

print("STEP 1: Loading PDF...")
print(f"PDF: {pdf_path}")

loader = PyPDFLoader(str(pdf_path))
pages = loader.load()

print(f"PDF pages loaded: {len(pages)}")


# ==========================================
# STEP 2: Combine PDF text
# ==========================================

print("\nSTEP 2: Extracting PDF text...")

full_text = "\n".join(
    page.page_content
    for page in pages
)

print(f"Extracted characters: {len(full_text)}")


# ==========================================
# STEP 3: Find FAQ section
# ==========================================

print("\nSTEP 3: Preparing FAQ content...")

faq_start = full_text.find("FAQ ID:")

if faq_start == -1:
    raise ValueError(
        "Could not find FAQ ID in the PDF."
    )

faq_text = full_text[faq_start:]


# ==========================================
# STEP 4: Split into FAQ blocks
# ==========================================

print("\nSTEP 4: Splitting PDF into FAQs...")

import re

faq_blocks = re.split(
    r"(?=FAQ ID:\s*\d+)",
    faq_text
)

faq_blocks = [
    block.strip()
    for block in faq_blocks
    if block.strip()
]

print(f"FAQ blocks found: {len(faq_blocks)}")

if len(faq_blocks) != 989:
    raise ValueError(
        f"Expected 989 FAQs, but found {len(faq_blocks)}."
    )


# ==========================================
# STEP 5: Create LangChain Documents
# ==========================================

print("\nSTEP 5: Creating LangChain documents...")

documents = []

for block in faq_blocks:

    faq_match = re.search(
        r"FAQ ID:\s*(\d+)",
        block
    )

    section_match = re.search(
        r"Section:\s*(.*?)\n",
        block
    )

    question_match = re.search(
        r"Question:\s*(.*?)\n",
        block
    )

    answer_match = re.search(
        r"Answer:\s*(.*)",
        block,
        re.DOTALL
    )

    if not faq_match:
        continue

    faq_id = int(faq_match.group(1))

    section = (
        section_match.group(1).strip()
        if section_match
        else "Unknown"
    )

    question = (
        question_match.group(1).strip()
        if question_match
        else ""
    )

    answer = (
        answer_match.group(1).strip()
        if answer_match
        else ""
    )

    content = (
        f"FAQ ID: {faq_id}\n"
        f"Section: {section}\n"
        f"Question: {question}\n"
        f"Answer: {answer}"
    )

    documents.append(
        Document(
            page_content=content,
            metadata={
                "faq_id": faq_id,
                "section": section,
                "source": "banking_faq.pdf"
            }
        )
    )


print(
    f"LangChain documents created: "
    f"{len(documents)}"
)


# ==========================================
# STEP 6: Connect Ollama embeddings
# ==========================================

print("\nSTEP 6: Connecting to Ollama embeddings...")

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

print("Embedding model connected!")


# ==========================================
# STEP 7: Create ChromaDB
# ==========================================

print("\nSTEP 7: Creating persistent ChromaDB...")

vectorstore_path.mkdir(
    parents=True,
    exist_ok=True
)

vectorstore = Chroma(
    collection_name="banking_faq",
    embedding_function=embeddings,
    persist_directory=str(vectorstore_path)
)

print("ChromaDB connected!")


# ==========================================
# STEP 8: Add documents in batches
# ==========================================

print("\nSTEP 8: Generating embeddings...")

batch_size = 50
total = len(documents)

for start in range(0, total, batch_size):

    end = min(
        start + batch_size,
        total
    )

    batch = documents[start:end]

    print(
        f"Processing FAQs "
        f"{start + 1} - {end} "
        f"of {total}..."
    )

    vectorstore.add_documents(batch)

    print(
        f"Batch completed: "
        f"{end}/{total}"
    )


# ==========================================
# STEP 9: Verify database
# ==========================================

print("\n========== FINAL VECTOR STORE RESULTS ==========")

count = vectorstore._collection.count()

print(f"PDF pages: {len(pages)}")
print(f"FAQ documents: {len(documents)}")
print(f"Documents stored in ChromaDB: {count}")

if count != 989:
    raise ValueError(
        f"Expected 989 documents in ChromaDB, "
        f"but found {count}."
    )

print("\nSUCCESS!")
print("The production ChromaDB was built directly from the PDF.")
print(f"Location: {vectorstore_path}")