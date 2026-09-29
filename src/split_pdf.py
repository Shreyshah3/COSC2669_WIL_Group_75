import pandas as pd
from pathlib import Path

from langchain_core.documents import Document


# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

input_file = Path("data/processed/banking_faq_clean.csv")

print("Loading cleaned FAQ dataset...")
print(f"Dataset: {input_file}")


# --------------------------------------------------
# 2. Load cleaned dataset
# --------------------------------------------------

df = pd.read_csv(input_file)

print(f"Total FAQs loaded: {len(df)}")


# --------------------------------------------------
# 3. Convert each FAQ into a LangChain Document
# --------------------------------------------------

documents = []

for _, row in df.iterrows():

    faq_id = row["faq_id"]
    section = row["section"]
    question = row["question"]
    answer = row["answer"]

    content = (
        f"FAQ ID: {faq_id}\n"
        f"Section: {section}\n"
        f"Question: {question}\n"
        f"Answer: {answer}"
    )

    document = Document(
        page_content=content,
        metadata={
            "faq_id": int(faq_id),
            "section": section,
            "source": "banking_faq.pdf"
        }
    )

    documents.append(document)


# --------------------------------------------------
# 4. Results
# --------------------------------------------------

print("\n========== FAQ CHUNKING RESULTS ==========")

print(f"Original FAQs: {len(df)}")
print(f"LangChain documents: {len(documents)}")


# --------------------------------------------------
# 5. Inspect first 5 documents
# --------------------------------------------------

for i, document in enumerate(documents[:5], start=1):

    print("\n----------------------------------------")
    print(f"DOCUMENT {i}")

    print("\nContent:")
    print(document.page_content)

    print("\nMetadata:")
    print(document.metadata)


# --------------------------------------------------
# 6. Document statistics
# --------------------------------------------------

document_lengths = [
    len(document.page_content)
    for document in documents
]

print("\n========== DOCUMENT STATISTICS ==========")

print(
    f"Minimum document length: "
    f"{min(document_lengths)} characters"
)

print(
    f"Maximum document length: "
    f"{max(document_lengths)} characters"
)

print(
    f"Average document length: "
    f"{sum(document_lengths) / len(document_lengths):.2f} characters"
)


print("\nFAQ chunking completed successfully!")