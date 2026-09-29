from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


# --------------------------------------------------
# 1. PDF location
# --------------------------------------------------

pdf_path = Path("data/pdf/banking_faq.pdf")

print("Loading PDF...")
print(f"PDF path: {pdf_path}")


# --------------------------------------------------
# 2. Load PDF
# --------------------------------------------------

loader = PyPDFLoader(str(pdf_path))

documents = loader.load()


# --------------------------------------------------
# 3. Display results
# --------------------------------------------------

print("\n========== PDF LOADING RESULTS ==========")

print(f"Total pages loaded: {len(documents)}")


# --------------------------------------------------
# 4. Inspect first page
# --------------------------------------------------

if documents:

    first_document = documents[0]

    print("\n========== FIRST PAGE ==========")

    print("\nPage content:")
    print(first_document.page_content[:2000])

    print("\nMetadata:")
    print(first_document.metadata)


# --------------------------------------------------
# 5. Completion
# --------------------------------------------------

print("\nPDF loading completed successfully!")