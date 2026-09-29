import pandas as pd
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak
)


# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

input_file = Path("data/processed/banking_faq_clean.csv")
output_folder = Path("data/pdf")

output_folder.mkdir(parents=True, exist_ok=True)

output_file = output_folder / "banking_faq.pdf"


# --------------------------------------------------
# 2. Load cleaned dataset
# --------------------------------------------------

print("Loading cleaned dataset...")

df = pd.read_csv(input_file)

print(f"Records loaded: {len(df)}")


# --------------------------------------------------
# 3. Create PDF document
# --------------------------------------------------

document = SimpleDocTemplate(
    str(output_file),
    pagesize=A4,
    rightMargin=20 * mm,
    leftMargin=20 * mm,
    topMargin=20 * mm,
    bottomMargin=20 * mm
)


# --------------------------------------------------
# 4. Create styles
# --------------------------------------------------

styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "TitleStyle",
    parent=styles["Title"],
    alignment=TA_CENTER,
    fontSize=20,
    leading=24,
    spaceAfter=10
)

subtitle_style = ParagraphStyle(
    "SubtitleStyle",
    parent=styles["Normal"],
    alignment=TA_CENTER,
    fontSize=10,
    leading=14,
    spaceAfter=20
)

section_style = ParagraphStyle(
    "SectionStyle",
    parent=styles["Heading2"],
    fontSize=13,
    leading=16,
    spaceBefore=8,
    spaceAfter=8
)

question_style = ParagraphStyle(
    "QuestionStyle",
    parent=styles["Heading3"],
    fontSize=11,
    leading=15,
    spaceBefore=5,
    spaceAfter=5
)

answer_style = ParagraphStyle(
    "AnswerStyle",
    parent=styles["BodyText"],
    fontSize=10,
    leading=15,
    spaceAfter=8
)

id_style = ParagraphStyle(
    "IDStyle",
    parent=styles["Normal"],
    fontSize=8,
    textColor=colors.grey,
    spaceAfter=3
)


# --------------------------------------------------
# 5. Build PDF content
# --------------------------------------------------

story = []


# Title

story.append(
    Paragraph(
        "Banking Frequently Asked Questions",
        title_style
    )
)

story.append(
    Paragraph(
        "Banking FAQ Knowledge Base",
        subtitle_style
    )
)

story.append(
    Paragraph(
        f"Total FAQs: {len(df)}",
        subtitle_style
    )
)

story.append(Spacer(1, 10))


# --------------------------------------------------
# 6. Add each FAQ
# --------------------------------------------------

for _, row in df.iterrows():

    faq_id = row["faq_id"]
    section = str(row["section"])
    question = str(row["question"])
    answer = str(row["answer"])

    story.append(
        Paragraph(
            f"FAQ ID: {faq_id}",
            id_style
        )
    )

    story.append(
        Paragraph(
            f"Section: {section}",
            section_style
        )
    )

    story.append(
        Paragraph(
            f"Question: {question}",
            question_style
        )
    )

    story.append(
        Paragraph(
            f"Answer: {answer}",
            answer_style
        )
    )

    story.append(
        Spacer(1, 8)
    )


# --------------------------------------------------
# 7. Generate PDF
# --------------------------------------------------

print("Creating PDF...")

document.build(story)

print("\nPDF created successfully!")
print(f"Location: {output_file}")