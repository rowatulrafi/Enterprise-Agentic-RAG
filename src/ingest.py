import os
import re
import json
import shutil
import pytesseract
import pymupdf

from pdf2image import convert_from_path
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    llm,
    embeddings,
    RAW_PDF_DIR,
    CLEAN_JSON_DIR,
    REVIEW_DIR,
    CHROMA_DIR,
    DATA_DIR,
)

from src.state import DocumentValidation


# ============================================================
# LLM VALIDATOR
# ============================================================

structured_validator = llm.with_structured_output(
    DocumentValidation,
    method="json_schema"
)

validation_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a document ingestion quality-control assistant.

Evaluate extracted text for suitability in a Retrieval-Augmented
Generation knowledge base.

A page does NOT need to be perfectly formatted.

Accept the text if:
- the main factual content is readable
- sentences or table values remain understandable
- minor OCR errors, headers, footers, or formatting artifacts exist

Reject or flag the text only if:
- important text is mostly unreadable
- characters are severely corrupted
- numbers cannot reliably be associated with their labels
- the page is dominated by unusable OCR artifacts

Do not penalize a page simply because it contains financial tables,
percentages, charts, headings, or dense numerical content.

Score readability from 0 to 100.
"""
    ),
    (
        "human",
        "Evaluate this extracted text:\n\n{text}"
    ),
])

validator_chain = validation_prompt | structured_validator


# ============================================================
# TEXT UTILITIES
# ============================================================

def clean_text_for_llm(text: str) -> str:
    text = re.sub(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]",
        "",
        text,
    )

    return text.replace("\\", "/").strip()


def is_native_text_usable(text: str) -> bool:
    """
    Basic deterministic check for whether native PDF extraction
    produced sufficiently useful text.
    """

    text = clean_text_for_llm(text)

    if len(text) < 100:
        return False

    alphanumeric_count = sum(
        char.isalnum()
        for char in text
    )

    alphanumeric_ratio = (
        alphanumeric_count / max(len(text), 1)
    )

    return alphanumeric_ratio >= 0.40


# ============================================================
# OCR FALLBACK
# ============================================================

def extract_page_with_ocr(
    pdf_path: str,
    page_num: int,
) -> str:
    """
    OCR only the requested page instead of rasterizing
    the entire PDF.
    """

    images = convert_from_path(
        pdf_path,
        dpi=300,
        first_page=page_num,
        last_page=page_num,
    )

    if not images:
        return ""

    return clean_text_for_llm(
        pytesseract.image_to_string(images[0])
    )


def validate_ocr_text(
    text: str,
):
    """
    Use the LLM only for OCR text that requires quality validation.
    """

    return validator_chain.invoke({
        "text": text[:4000]
    })


# ============================================================
# PHASE 1 — DOCUMENT EXTRACTION
# ============================================================

def process_pdfs():
    """
    Native-text-first PDF ingestion.

    1. Attempt native PyMuPDF text extraction.
    2. Fall back to OCR only when necessary.
    3. Validate OCR output.
    4. Route genuinely poor pages to human review.
    """

    print("\n" + "=" * 60)
    print("📄 PHASE 1: DOCUMENT INGESTION")
    print("=" * 60)

    os.makedirs(CLEAN_JSON_DIR, exist_ok=True)
    os.makedirs(REVIEW_DIR, exist_ok=True)

    pdf_files = [
        filename
        for filename in os.listdir(RAW_PDF_DIR)
        if filename.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print(
            f"⚠️ No PDFs found in '{RAW_PDF_DIR}'."
        )
        return False

    for pdf_filename in pdf_files:

        base_name = os.path.splitext(
            pdf_filename
        )[0]

        json_filepath = os.path.join(
            CLEAN_JSON_DIR,
            f"{base_name}.json",
        )

        # ----------------------------------------
        # Skip previously processed documents
        # ----------------------------------------

        if os.path.exists(json_filepath):
            print(
                f"⏭️ Skipping: {pdf_filename} "
                "(clean JSON already exists)"
            )
            continue

        print(f"\n▶️ Processing: {pdf_filename}")

        pdf_path = os.path.join(
            RAW_PDF_DIR,
            pdf_filename,
        )

        clean_pages = []

        # ----------------------------------------
        # Open PDF
        # ----------------------------------------

        with pymupdf.open(pdf_path) as pdf:

            total_pages = len(pdf)

            print(
                f"   Pages detected: {total_pages}"
            )

            for index, page in enumerate(pdf):

                page_num = index + 1

                print(
                    f"\n  📄 Page {page_num}/{total_pages}"
                )

                # ========================================
                # STEP 1 — Native extraction
                # ========================================

                native_text = clean_text_for_llm(
                    page.get_text("text")
                )

                if is_native_text_usable(
                    native_text
                ):

                    print(
                        "     ✅ Native text accepted"
                    )

                    clean_pages.append({
                        "page_num": page_num,
                        "text": native_text,
                        "extraction_method": "native",
                    })

                    continue

                # ========================================
                # STEP 2 — OCR fallback
                # ========================================

                print(
                    "     🔍 Native text insufficient "
                    "→ OCR fallback"
                )

                ocr_text = extract_page_with_ocr(
                    pdf_path,
                    page_num,
                )

                if len(ocr_text) < 20:

                    print(
                        "     ⚠️ OCR text too short"
                    )

                    review_file = os.path.join(
                        REVIEW_DIR,
                        f"{base_name}_page_"
                        f"{page_num}_REVIEW.txt",
                    )

                    with open(
                        review_file,
                        "w",
                        encoding="utf-8",
                    ) as file:

                        file.write(
                            "--- HUMAN REVIEW REQUIRED ---\n"
                            "Reason: OCR produced insufficient "
                            "text.\n\n"
                            f"{ocr_text}"
                        )

                    continue

                # ========================================
                # STEP 3 — Validate OCR output
                # ========================================

                try:

                    result = validate_ocr_text(
                        ocr_text
                    )

                    print(
                        f"     Quality score: "
                        f"{result.confidence_score}"
                    )

                    print(
                        f"     Garbled: "
                        f"{result.is_garbled}"
                    )

                    if (
                        result.confidence_score >= 70
                        and not result.is_garbled
                    ):

                        print(
                            "     ✅ OCR accepted"
                        )

                        clean_pages.append({
                            "page_num": page_num,
                            "text": ocr_text,
                            "extraction_method": "ocr",
                            "quality_score":
                                result.confidence_score,
                        })

                    else:

                        review_file = os.path.join(
                            REVIEW_DIR,
                            f"{base_name}_page_"
                            f"{page_num}_REVIEW.txt",
                        )

                        with open(
                            review_file,
                            "w",
                            encoding="utf-8",
                        ) as file:

                            file.write(
                                "--- HUMAN REVIEW REQUIRED ---\n"
                                f"Quality score: "
                                f"{result.confidence_score}\n"
                                f"Reason: "
                                f"{result.reasoning}\n\n"
                                f"{ocr_text}"
                            )

                        print(
                            "     ❌ Sent to human review"
                        )

                except Exception as error:

                    print(
                        f"     ❌ Validator failed: "
                        f"{error}"
                    )

                    review_file = os.path.join(
                        REVIEW_DIR,
                        f"{base_name}_page_"
                        f"{page_num}_REVIEW.txt",
                    )

                    with open(
                        review_file,
                        "w",
                        encoding="utf-8",
                    ) as file:

                        file.write(
                            "--- HUMAN REVIEW REQUIRED ---\n"
                            f"Validator error: "
                            f"{error}\n\n"
                            f"{ocr_text}"
                        )

        # ----------------------------------------
        # Save document staging JSON
        # ----------------------------------------

        if clean_pages:

            with open(
                json_filepath,
                "w",
                encoding="utf-8",
            ) as file:

                json.dump(
                    clean_pages,
                    file,
                    indent=2,
                    ensure_ascii=False,
                )

            native_count = sum(
                page["extraction_method"]
                == "native"
                for page in clean_pages
            )

            ocr_count = sum(
                page["extraction_method"]
                == "ocr"
                for page in clean_pages
            )

            print(
                f"\n✅ Saved {len(clean_pages)} pages"
            )

            print(
                f"   Native : {native_count}"
            )

            print(
                f"   OCR    : {ocr_count}"
            )

        else:

            print(
                f"\n⚠️ No usable pages extracted "
                f"from {pdf_filename}"
            )

    return True


# ============================================================
# PHASE 2 — BUILD RETRIEVAL DATABASES
# ============================================================

def build_hybrid_databases():

    print("\n" + "=" * 60)
    print("🗄️ PHASE 2: HYBRID DATABASE INDEXING")
    print("=" * 60)

    json_files = [
        filename
        for filename in os.listdir(
            CLEAN_JSON_DIR
        )
        if filename.lower().endswith(".json")
    ]

    if not json_files:
        print(
            "⚠️ No clean JSON data available."
        )
        return

    documents = []

    for json_file in json_files:

        json_path = os.path.join(
            CLEAN_JSON_DIR,
            json_file,
        )

        with open(
            json_path,
            "r",
            encoding="utf-8",
        ) as file:

            pages = json.load(file)

        for page in pages:

            metadata = {
                "source": json_file,
                "page": page["page_num"],
                "extraction_method":
                    page.get(
                        "extraction_method",
                        "unknown",
                    ),
            }

            documents.append(
                Document(
                    page_content=page["text"],
                    metadata=metadata,
                )
            )

    # ----------------------------------------
    # Chunk documents
    # ----------------------------------------

    text_splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
    )

    chunks = (
        text_splitter
        .split_documents(documents)
    )

    print(
        f"Prepared {len(chunks)} chunks "
        f"from {len(documents)} pages."
    )

    # ----------------------------------------
    # Remove previous Chroma database
    # ----------------------------------------

    if os.path.exists(CHROMA_DIR):

        print(
            "🧹 Clearing old Chroma database..."
        )

        shutil.rmtree(CHROMA_DIR)

    # ----------------------------------------
    # Dense vector database
    # ----------------------------------------

    print(
        "🔢 Building Chroma vector database..."
    )

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name="ultimate_docs",
    )

    # ----------------------------------------
    # BM25 corpus WITH metadata
    # ----------------------------------------

    bm25_data = [
        {
            "content": chunk.page_content,
            "metadata": chunk.metadata,
        }
        for chunk in chunks
    ]

    bm25_data_path = os.path.join(
        DATA_DIR,
        "bm25_corpus.json",
    )

    with open(
        bm25_data_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            bm25_data,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "🎉 Chroma and BM25 databases "
        "successfully synchronized."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    if process_pdfs():
        build_hybrid_databases()