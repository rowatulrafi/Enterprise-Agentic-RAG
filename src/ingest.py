import os
import re
import json
import shutil
import pytesseract
from pdf2image import convert_from_path
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    llm, embeddings, RAW_PDF_DIR, CLEAN_JSON_DIR, 
    REVIEW_DIR, CHROMA_DIR, DATA_DIR
)
from src.state import DocumentValidation

# --- INIT VALIDATOR ---
structured_validator = llm.with_structured_output(DocumentValidation)
validation_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an automated Data Quality Assurance agent evaluating OCR text from a PDF. "
               "If the text is readable and coherent, score it high. "
               "If the text contains scattered symbols, broken diagrams, or OCR garbage, score it low and mark it as garbled. "
               "IMPORTANT: Summarize issues in plain English without repeating raw control characters."),
    ("human", "Evaluate this extracted text:\n\n{text}")
])
validator_chain = validation_prompt | structured_validator

def clean_text_for_llm(text: str) -> str:
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    return text.replace('\\', '/').strip()

def process_pdfs():
    """Phase 1: OCR Extraction, LLM Validation, and Data Routing"""
    print("\n" + "="*40)
    print("📄 PHASE 1: MULTI-MODAL OCR INGESTION")
    print("="*40)
    
    pdf_files = [f for f in os.listdir(RAW_PDF_DIR) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"⚠️ No PDFs found in '{RAW_PDF_DIR}'.")
        return False

    processed_any = False

    for pdf_filename in pdf_files:
        base_name = pdf_filename.replace(".pdf", "")
        json_filepath = os.path.join(CLEAN_JSON_DIR, f"{base_name}.json")
        
        # --- 🛡️ THE SMART SKIP ---
        if os.path.exists(json_filepath):
            print(f"⏭️ Skipping Extraction for: {pdf_filename} (Clean JSON already exists)")
            continue
        # -------------------------

        processed_any = True
        print(f"\n▶️ Processing: {pdf_filename}")
        pdf_path = os.path.join(RAW_PDF_DIR, pdf_filename)
        images = convert_from_path(pdf_path)
        
        clean_pages = []
        
        for i, image in enumerate(images):
            page_num = i + 1
            print(f"  🔍 Scanning page {page_num}...")
            raw_text = clean_text_for_llm(pytesseract.image_to_string(image))
            
            if len(raw_text) < 20:
                print(f"    ⚠️ Page {page_num} flagged: Too short/blank.")
                continue

            try:
                result = validator_chain.invoke({"text": raw_text[:3000]})
                if result.confidence_score >= 80 and not result.is_garbled:
                    clean_pages.append({"page_num": page_num, "text": raw_text})
                else:
                    review_file = os.path.join(REVIEW_DIR, f"{base_name}_page_{page_num}_REVIEW.txt")
                    with open(review_file, "w") as f:
                        f.write(f"--- HUMAN REVIEW REQUIRED ---\nReason: {result.reasoning}\n\n{raw_text}")
                    print(f"    ❌ Page {page_num} flagged -> Sent to Human Review.")
            except Exception as e:
                print(f"    ❌ Page {page_num} crashed validator -> Sent to Human Review.")
                
        if clean_pages:
            with open(json_filepath, "w") as f:
                json.dump(clean_pages, f, indent=4)
            print(f"  ✅ Saved {len(clean_pages)} clean pages to staging.")

    if not processed_any:
        print("\n✅ All PDFs have already been processed.")
        
    return True

def build_hybrid_databases():
    """Phase 2: Text Chunking and Dual-Database Indexing"""
    print("\n" + "="*40)
    print("🗄️ PHASE 2: HYBRID DATABASE INDEXING")
    print("="*40)
    
    json_files = [f for f in os.listdir(CLEAN_JSON_DIR) if f.endswith('.json')]
    if not json_files:
        print("⚠️ No clean JSON data available to index.")
        return

    documents = []
    for json_file in json_files:
        with open(os.path.join(CLEAN_JSON_DIR, json_file), "r") as f:
            pages = json.load(f)
            for page in pages:
                meta = {"source": json_file, "page": page["page_num"]}
                documents.append(Document(page_content=page["text"], metadata=meta))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    # --- 🛡️ THE DATABASE WIPE ---
    if os.path.exists(CHROMA_DIR):
        print("🧹 Clearing old Chroma DB to prevent duplicates...")
        shutil.rmtree(CHROMA_DIR)
    # ----------------------------

    print(f"Prepared {len(chunks)} text chunks. Loading into databases...")

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name="ultimate_docs"
    )
    
    bm25_data_path = os.path.join(DATA_DIR, "bm25_corpus.json")
    bm25_texts = [chunk.page_content for chunk in chunks]
    with open(bm25_data_path, "w") as f:
        json.dump(bm25_texts, f)
        
    print("🎉 Ingestion Complete! Chroma and BM25 are fully synchronized and ready for querying.")

if __name__ == "__main__":
    if process_pdfs():
        build_hybrid_databases()