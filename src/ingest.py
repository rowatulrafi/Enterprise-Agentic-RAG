import os
import re
import json
import shutil

import pymupdf
import pytesseract

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
# DIRECTORIES
# ============================================================

os.makedirs(RAW_PDF_DIR, exist_ok=True)
os.makedirs(CLEAN_JSON_DIR, exist_ok=True)
os.makedirs(REVIEW_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# ============================================================
# OPTIONAL TESSERACT CONFIG
# ============================================================

tesseract_cmd = os.getenv("TESSERACT_CMD")

if tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd


# ============================================================
# OCR VALIDATION
# ============================================================

structured_validator = llm.with_structured_output(
    DocumentValidation,
    method="json_schema",
)


validation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a document-quality validation agent.

Your task is to determine whether OCR-extracted text from a PDF
page is suitable for use in a Retrieval-Augmented Generation
(RAG) knowledge base.

A page does NOT need perfect formatting.

ACCEPT text when:
- the main prose is readable and coherent
- minor OCR errors exist but meaning is preserved
- headers, footers, page numbers, or formatting artifacts exist
- tables contain understandable labels and values
- mathematical or technical notation is imperfect but surrounding
  textual meaning remains understandable

REJECT text when:
- characters are severely corrupted
- words are mostly meaningless
- important labels and values cannot be associated reliably
- OCR has destroyed the meaning of the page
- the page contains mostly unusable fragments

Do not penalize a page merely because it contains:
- financial tables
- equations
- charts
- technical terminology
- dense numerical information

Return:
- confidence_score from 0 to 100
- is_garbled
- a short explanation
""",
        ),
        (
            "human",
            """
Evaluate the following OCR text for RAG ingestion:

{text}
""",
        ),
    ]
)


validator_chain = (
    validation_prompt
    | structured_validator
)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text_for_llm(text: str) -> str:
    """
    Remove problematic control characters while preserving
    Unicode symbols such as arrows and mathematical notation.
    """

    if not text:
        return ""

    text = re.sub(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]",
        "",
        text,
    )

    text = text.replace("\\", "/")

    # Collapse excessive spaces but preserve line structure.
    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


# ============================================================
# NATIVE TEXT QUALITY CHECK
# ============================================================

def is_native_text_usable(text: str) -> bool:
    """
    Deterministic gate for native PDF text.

    Native text should be preferred over OCR whenever there is
    enough meaningful machine-readable content.
    """

    if not text:
        return False

    cleaned = text.strip()

    if len(cleaned) < 100:
        return False

    alnum_count = sum(
        char.isalnum()
        for char in cleaned
    )

    alnum_ratio = (
        alnum_count / len(cleaned)
        if cleaned
        else 0
    )

    # Require a minimum amount of actual textual content.
    word_count = len(
        re.findall(
            r"\b[\w'-]+\b",
            cleaned,
            flags=re.UNICODE,
        )
    )

    if word_count < 15:
        return False

    return alnum_ratio >= 0.40


# ============================================================
# PAGE-SPECIFIC OCR
# ============================================================

def extract_page_with_ocr(
    pdf_path: str,
    page_num: int,
) -> str:
    """
    OCR only the requested page.

    page_num is 1-indexed.
    """

    images = convert_from_path(
        pdf_path,
        dpi=300,
        first_page=page_num,
        last_page=page_num,
    )

    if not images:
        return ""

    raw_text = pytesseract.image_to_string(
        images[0]
    )

    return clean_text_for_llm(
        raw_text
    )


# ============================================================
# OCR VALIDATION
# ============================================================

def validate_ocr_text(
    text: str,
):
    """
    Validate OCR output using the local structured-output LLM.
    """

    return validator_chain.invoke(
        {
            "text": text[:4000]
        }
    )


# ============================================================
# HUMAN REVIEW QUEUE
# ============================================================

def send_to_human_review(
    base_name: str,
    page_num: int,
    text: str,
    reason: str,
    score=None,
):
    """
    Save problematic OCR output for manual review.
    """

    review_file = os.path.join(
        REVIEW_DIR,
        f"{base_name}_page_{page_num}_REVIEW.txt",
    )

    with open(
        review_file,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "--- HUMAN REVIEW REQUIRED ---\n"
        )

        file.write(
            f"Document: {base_name}\n"
        )

        file.write(
            f"Page: {page_num}\n"
        )

        if score is not None:
            file.write(
                f"Quality score: {score}\n"
            )

        file.write(
            f"Reason: {reason}\n\n"
        )

        file.write(
            "--- OCR OUTPUT ---\n\n"
        )

        file.write(
            text.strip()
        )

        file.write(
            "\n\n--- CORRECTED TEXT ---\n\n"
        )


# ============================================================
# PHASE 1
# ============================================================

def process_pdfs():
    """
    Phase 1:

    1. Attempt native PDF text extraction first.
    2. Accept usable native text immediately.
    3. Use OCR only when native extraction is insufficient.
    4. Validate OCR output.
    5. Route poor OCR to human review.
    """

    print(
        "\n"
        + "=" * 60
    )

    print(
        "📄 PHASE 1: NATIVE-FIRST DOCUMENT INGESTION"
    )

    print(
        "=" * 60
    )

    pdf_files = sorted(
        [
            filename
            for filename
            in os.listdir(RAW_PDF_DIR)
            if filename.lower().endswith(
                ".pdf"
            )
        ]
    )

    if not pdf_files:

        print(
            f"⚠️ No PDFs found in "
            f"'{RAW_PDF_DIR}'."
        )

        return False

    processed_any = False

    for pdf_filename in pdf_files:

        base_name = os.path.splitext(
            pdf_filename
        )[0]

        json_filepath = os.path.join(
            CLEAN_JSON_DIR,
            f"{base_name}.json",
        )

        # ----------------------------------------------------
        # SMART SKIP
        # ----------------------------------------------------

        if os.path.exists(
            json_filepath
        ):

            print(
                f"\n⏭️ Skipping extraction: "
                f"{pdf_filename} "
                "(clean JSON already exists)"
            )

            continue

        processed_any = True

        print(
            f"\n▶️ Processing: "
            f"{pdf_filename}"
        )

        pdf_path = os.path.join(
            RAW_PDF_DIR,
            pdf_filename,
        )

        clean_pages = []

        native_count = 0
        ocr_count = 0
        review_count = 0
        skipped_count = 0

        try:

            with pymupdf.open(
                pdf_path
            ) as pdf:

                total_pages = len(pdf)

                for page_index, page in enumerate(
                    pdf
                ):

                    page_num = (
                        page_index + 1
                    )

                    print(
                        f"\n  🔍 Page "
                        f"{page_num}/"
                        f"{total_pages}"
                    )

                    # =========================================
                    # STEP 1:
                    # NATIVE EXTRACTION
                    # =========================================

                    native_text = (
                        clean_text_for_llm(
                            page.get_text(
                                "text",
                                sort=True,
                            )
                        )
                    )

                    if is_native_text_usable(
                        native_text
                    ):

                        clean_pages.append(
                            {
                                "page_num":
                                    page_num,

                                "text":
                                    native_text,

                                "extraction_method":
                                    "native",
                            }
                        )

                        native_count += 1

                        print(
                            "    ✅ Native text "
                            "accepted."
                        )

                        # CRITICAL:
                        # DO NOT FALL THROUGH TO OCR.
                        continue

                    # =========================================
                    # STEP 2:
                    # OCR FALLBACK
                    # =========================================

                    print(
                        "    ⚠️ Native text "
                        "insufficient -> OCR fallback."
                    )

                    try:

                        ocr_text = (
                            extract_page_with_ocr(
                                pdf_path,
                                page_num,
                            )
                        )

                    except Exception as error:

                        print(
                            "    ❌ OCR failed: "
                            f"{error}"
                        )

                        review_count += 1

                        send_to_human_review(
                            base_name=
                                base_name,

                            page_num=
                                page_num,

                            text=
                                native_text,

                            reason=
                                (
                                    "OCR extraction "
                                    f"failed: {error}"
                                ),
                        )

                        continue

                    # -----------------------------------------
                    # Empty / nearly empty OCR
                    # -----------------------------------------

                    if len(
                        ocr_text.strip()
                    ) < 20:

                        print(
                            "    ⚪ OCR produced "
                            "almost no text. "
                            "Page skipped."
                        )

                        skipped_count += 1

                        continue

                    # =========================================
                    # STEP 3:
                    # LLM VALIDATION OF OCR
                    # =========================================

                    try:

                        result = (
                            validate_ocr_text(
                                ocr_text
                            )
                        )

                    except Exception as error:

                        print(
                            "    ❌ OCR validator "
                            f"failed: {error}"
                        )

                        review_count += 1

                        send_to_human_review(
                            base_name=
                                base_name,

                            page_num=
                                page_num,

                            text=
                                ocr_text,

                            reason=
                                (
                                    "OCR validator "
                                    f"failed: {error}"
                                ),
                        )

                        continue

                    print(
                        "    OCR quality: "
                        f"{result.confidence_score}/100"
                    )

                    print(
                        "    Garbled: "
                        f"{result.is_garbled}"
                    )

                    # =========================================
                    # STEP 4:
                    # ACCEPT GOOD OCR
                    # =========================================

                    if (
                        result.confidence_score
                        >= 70
                        and not
                        result.is_garbled
                    ):

                        clean_pages.append(
                            {
                                "page_num":
                                    page_num,

                                "text":
                                    ocr_text,

                                "extraction_method":
                                    "ocr",

                                "quality_score":
                                    result.confidence_score,
                            }
                        )

                        ocr_count += 1

                        print(
                            "    ✅ OCR text "
                            "accepted."
                        )

                        continue

                    # =========================================
                    # STEP 5:
                    # HUMAN REVIEW
                    # =========================================

                    review_count += 1

                    send_to_human_review(
                        base_name=
                            base_name,

                        page_num=
                            page_num,

                        text=
                            ocr_text,

                        score=
                            result.confidence_score,

                        reason=
                            result.reasoning,
                    )

                    print(
                        "    ❌ Sent to "
                        "human review."
                    )

        except Exception as error:

            print(
                f"\n❌ Could not process "
                f"{pdf_filename}: {error}"
            )

            continue

        # ----------------------------------------------------
        # SAVE CLEAN DOCUMENT
        # ----------------------------------------------------

        if clean_pages:

            clean_pages = sorted(
                clean_pages,
                key=lambda item:
                    item["page_num"],
            )

            with open(
                json_filepath,
                "w",
                encoding="utf-8",
            ) as file:

                json.dump(
                    clean_pages,
                    file,
                    indent=4,
                    ensure_ascii=False,
                )

            print(
                "\n  "
                + "-" * 50
            )

            print(
                f"  ✅ Saved "
                f"{len(clean_pages)} "
                "retrievable pages."
            )

            print(
                f"     Native       : "
                f"{native_count}"
            )

            print(
                f"     OCR          : "
                f"{ocr_count}"
            )

            print(
                f"     Human review : "
                f"{review_count}"
            )

            print(
                f"     Skipped      : "
                f"{skipped_count}"
            )

        else:

            print(
                f"⚠️ No usable pages "
                f"found in {pdf_filename}."
            )

    if not processed_any:

        print(
            "\n✅ All PDFs already "
            "have staged clean JSON."
        )

    return True


# ============================================================
# PHASE 2
# ============================================================

def build_hybrid_databases():
    """
    Build synchronized:

    - Chroma dense-vector database
    - BM25 document corpus

    Metadata is preserved in BOTH stores.
    """

    print(
        "\n"
        + "=" * 60
    )

    print(
        "🗄️ PHASE 2: HYBRID DATABASE INDEXING"
    )

    print(
        "=" * 60
    )

    json_files = sorted(
        [
            filename
            for filename
            in os.listdir(
                CLEAN_JSON_DIR
            )
            if filename.lower().endswith(
                ".json"
            )
        ]
    )

    if not json_files:

        print(
            "⚠️ No clean JSON data "
            "available to index."
        )

        return

    documents = []

    total_pages = 0

    # --------------------------------------------------------
    # LOAD CLEAN DOCUMENTS
    # --------------------------------------------------------

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

            pages = json.load(
                file
            )

        for page in pages:

            text = (
                page
                .get(
                    "text",
                    ""
                )
                .strip()
            )

            if not text:
                continue

            extraction_method = (
                page.get(
                    "extraction_method",
                    "unknown",
                )
            )

            metadata = {
                "source":
                    json_file,

                "page":
                    page[
                        "page_num"
                    ],

                "extraction_method":
                    extraction_method,
            }

            if (
                "quality_score"
                in page
            ):

                metadata[
                    "quality_score"
                ] = page[
                    "quality_score"
                ]

            documents.append(
                Document(
                    page_content=text,
                    metadata=metadata,
                )
            )

            total_pages += 1

    if not documents:

        print(
            "⚠️ No valid document "
            "content found."
        )

        return

    # --------------------------------------------------------
    # CHUNKING
    # --------------------------------------------------------

    text_splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
    )

    chunks = (
        text_splitter
        .split_documents(
            documents
        )
    )

    print(
        f"\nPrepared "
        f"{len(chunks)} chunks "
        f"from {total_pages} pages."
    )

    # --------------------------------------------------------
    # CLEAR OLD CHROMA DB
    # --------------------------------------------------------

    if os.path.exists(
        CHROMA_DIR
    ):

        print(
            "🧹 Clearing old "
            "Chroma database..."
        )

        shutil.rmtree(
            CHROMA_DIR
        )

    # --------------------------------------------------------
    # DENSE VECTOR DATABASE
    # --------------------------------------------------------

    print(
        "🔹 Building Chroma "
        "vector database..."
    )

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=
            CHROMA_DIR,
        collection_name=
            "ultimate_docs",
    )

    # --------------------------------------------------------
    # BM25 CORPUS WITH METADATA
    # --------------------------------------------------------

    print(
        "🔹 Building BM25 "
        "corpus..."
    )

    bm25_data = [
        {
            "content":
                chunk.page_content,

            "metadata":
                chunk.metadata,
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
        "\n🎉 Ingestion complete."
    )

    print(
        f"   Pages  : "
        f"{total_pages}"
    )

    print(
        f"   Chunks : "
        f"{len(chunks)}"
    )

    print(
        "   Chroma and BM25 "
        "are synchronized."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    if process_pdfs():
        build_hybrid_databases()