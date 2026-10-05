import os
import json
import glob

# Define absolute paths based on the master project structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REVIEW_DIR = os.path.join(BASE_DIR, "data", "human_review_queue")
CLEAN_DIR = os.path.join(BASE_DIR, "data", "clean_json")

def auto_rename_reviews():
    """Finds _REVIEW.txt files and renames them to _FIXED.txt after human confirmation."""
    review_files = glob.glob(os.path.join(REVIEW_DIR, "*_REVIEW.txt"))
    
    if not review_files:
        return True # No review files to rename, proceed normally

    print(f"\n⚠️ Found {len(review_files)} file(s) ending in '_REVIEW.txt'.")
    confirm = input("Have you manually pasted the corrections into ALL of these files? (y/n): ")
    
    if confirm.lower().strip() == 'y':
        for filepath in review_files:
            new_filepath = filepath.replace("_REVIEW.txt", "_FIXED.txt")
            os.rename(filepath, new_filepath)
        print("✅ Auto-renamed all reviewed files to '_FIXED.txt'.")
        return True
    else:
        print("🛑 Merge aborted. Please finish editing your files before running this script.")
        return False

def merge_fixed_files():
    print("\n" + "="*50)
    print("🔍 Initializing Human-in-the-Loop Merger...")
    print("="*50)
    
    # 0. Run the auto-rename safety check
    if not auto_rename_reviews():
        return
        
    # 1. Find all files marked as FIXED
    fixed_files = glob.glob(os.path.join(REVIEW_DIR, "*_FIXED.txt"))
    
    if not fixed_files:
        print("🤷 No files ending in '_FIXED.txt' found. Queue is clear!")
        return

    for filepath in fixed_files:
        filename = os.path.basename(filepath)
        
        # 2. Extract the base PDF name and page number from the filename
        parts = filename.split("_page_")
        if len(parts) != 2:
            print(f"⚠️ Skipping {filename}: Unrecognized naming format.")
            continue
            
        pdf_base = parts[0]
        page_num_str = parts[1].replace("_FIXED.txt", "")
        
        try:
            page_num = int(page_num_str)
        except ValueError:
            print(f"⚠️ Skipping {filename}: Could not parse page number '{page_num_str}'.")
            continue
            
        # 3. Read the manually corrected text
        with open(filepath, "r", encoding="utf-8") as f:
            corrected_text = f.read().strip()
            
        # 4. Load the existing clean JSON from the pipeline
        json_filepath = os.path.join(CLEAN_DIR, f"{pdf_base}.json")
        
        if not os.path.exists(json_filepath):
            print(f"⚠️ JSON not found for {pdf_base}. Creating a new one.")
            clean_data = []
        else:
            with open(json_filepath, "r", encoding="utf-8") as f:
                clean_data = json.load(f)
        
        # 5. Inject the text (overwrite if page exists, append if new)
        for entry in clean_data:
            if entry.get("page_num") == page_num:
                entry["text"] = corrected_text
                break
        else:
            clean_data.append({"page_num": page_num, "text": corrected_text})
            
        # 6. Sort the array so the pages stay in proper order for the Text Splitter
        clean_data = sorted(clean_data, key=lambda x: x["page_num"])
        
        # 7. Safely write back to the JSON file
        with open(json_filepath, "w", encoding="utf-8") as f:
            json.dump(clean_data, f, indent=4)
            
        print(f"✅ Successfully merged Page {page_num} into '{pdf_base}.json'")
        
        # 8. Cleanup: Delete the text file so it isn't processed twice
        os.remove(filepath)
        
    print("\n🎉 All fixed files successfully injected into your clean data staging area!")
    print("👉 Next step: Run 'python -m src.ingest' to update your Chroma and BM25 databases.")

if __name__ == "__main__":
    merge_fixed_files()