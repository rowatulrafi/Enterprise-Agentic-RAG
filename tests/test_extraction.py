import os
import pymupdf

from src.config import RAW_PDF_DIR


pdf_path = os.path.join(
    RAW_PDF_DIR,
    "RLC.pdf"
)

with pymupdf.open(pdf_path) as pdf:

    page = pdf[4]   # zero-indexed -> PDF page 5

    print("\n" + "=" * 80)
    print("DEFAULT EXTRACTION")
    print("=" * 80)

    print(
        page.get_text("text")
    )

    print("\n" + "=" * 80)
    print("SORTED EXTRACTION")
    print("=" * 80)

    print(
        page.get_text(
            "text",
            sort=True
        )
    )