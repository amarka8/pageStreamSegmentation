import os
import time
import csv
from mistral import mistral_ocr
from textract import textract_ocr
from baker_ocr_gui import deepseek_extract

INPUT_FOLDER = "/Users/sahiljoshi/Documents/pageStreamSegmentation/data"
ORDER_CSV = "/Users/sahiljoshi/Documents/pageStreamSegmentation/ground_truth.csv"
CSV_OUTPUT_PATH = "deepseek_benchmark.csv"
OCR_ENGINE = "Mistral"  # or "Textract"

def read_order(csv_path, start_from=None):
    start_collecting = start_from is None
    ordered_names = []

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["Quartex Name"].strip()
            if not start_collecting:
                if name == start_from:
                    start_collecting = True
                else:
                    continue  # Skip until we hit the starting point
            ordered_names.append(name)

    return ordered_names


def auto_process_all_pdfs(repo_folder, order_csv, engine="Mistral", csv_path="deepseek_benchmark.csv"):
    order = read_order(order_csv, start_from="ghwb_0522")
    for name in order:
        pdf_name = name.strip() + ".pdf"
        file_path = os.path.join(repo_folder, pdf_name)
        if not os.path.isfile(file_path):
            print(f"⚠️ Skipping {pdf_name}: not found")
            continue

        print(f"\n📄 Processing in order: {pdf_name}")
        ocr_text, _ = mistral_ocr(file_path) if engine == "Mistral" else textract_ocr(file_path)
        if not ocr_text:
            print("❌ OCR failed.")
            continue

        try:
            deepseek_extract(ocr_text, file_id=name, csv_path=csv_path)
            print("✅ Gemini extraction completed.")
        except Exception as e:
            print(f"❌ Error for {pdf_name}: {e}")
            continue

        print("⏳ Waiting 60 seconds...")
        time.sleep(60)

if __name__ == "__main__":
    auto_process_all_pdfs(INPUT_FOLDER, ORDER_CSV, OCR_ENGINE, CSV_OUTPUT_PATH)
