import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


# ==========================================
# STEP 1: Load the PDF
# ==========================================

pdf_path = Path("data/pdf/banking_faq.pdf")

print("STEP 1: Loading banking PDF...")
print(f"PDF: {pdf_path}")

loader = PyPDFLoader(str(pdf_path))

pages = loader.load()

print(f"PDF pages loaded: {len(pages)}")


# ==========================================
# STEP 2: Combine PDF page text
# ==========================================

print("\nSTEP 2: Combining PDF text...")

full_text = "\n".join(
    page.page_content
    for page in pages
)

print(f"Total extracted characters: {len(full_text)}")


# ==========================================
# STEP 3: Remove PDF title/header
# ==========================================

print("\nSTEP 3: Preparing FAQ text...")

faq_start = full_text.find("FAQ ID:")

if faq_start == -1:
    raise ValueError(
        "ERROR: Could not find 'FAQ ID:' in the PDF."
    )

faq_text = full_text[faq_start:]


# ==========================================
# STEP 4: Split PDF text by FAQ ID
# ==========================================

print("\nSTEP 4: Splitting PDF into individual FAQs...")

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


# ==========================================
# STEP 5: Create LangChain Documents
# ==========================================

print("\nSTEP 5: Creating LangChain Documents...")

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

    document = Document(
        page_content=content,
        metadata={
            "faq_id": faq_id,
            "section": section,
            "source": "banking_faq.pdf"
        }
    )

    documents.append(document)


# ==========================================
# STEP 6: Display results
# ==========================================

print("\n========== PDF DOCUMENT RESULTS ==========")

print(f"PDF pages: {len(pages)}")
print(f"FAQ blocks found: {len(faq_blocks)}")
print(f"LangChain documents created: {len(documents)}")


# ==========================================
# STEP 7: Display first 5 FAQs
# ==========================================

for i, document in enumerate(documents[:5], start=1):

    print("\n----------------------------------------")
    print(f"DOCUMENT {i}")
    print("----------------------------------------")

    print(document.page_content)

    print("\nMetadata:")
    print(document.metadata)


# ==========================================
# STEP 8: Document statistics
# ==========================================

if documents:

    lengths = [
        len(document.page_content)
        for document in documents
    ]

    print("\n========== DOCUMENT STATISTICS ==========")

    print(
        f"Minimum length: "
        f"{min(lengths)} characters"
    )

    print(
        f"Maximum length: "
        f"{max(lengths)} characters"
    )

    print(
        f"Average length: "
        f"{sum(lengths) / len(lengths):.2f} characters"
    )


print("\nPDF → LangChain document conversion completed!")