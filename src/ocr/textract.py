from mistralai import Mistral
from pdf2image import convert_from_path
import boto3
import io
import json
import os
from dotenv import load_dotenv
# Load environment variables from .env file


load_dotenv()  # By default looks for .env file in current director

# This computes OCR with the help of Amazon Textract


def textract_ocr(pdf_path):
    images = convert_from_path(pdf_path, dpi=250)
    textract = boto3.client('textract')
    all_pages_text = []
    all_pages_md = []

    for i, img in enumerate(images):
        with io.BytesIO() as img_bytes:
            img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            response = textract.detect_document_text(Document={'Bytes': img_bytes.getvalue()})
        
        page_lines = []
        for item in response["Blocks"]:
            if item["BlockType"] == "LINE":
                page_lines.append(item["Text"])
        
        page_text = "\n".join(page_lines)
        page_md = "\n\n".join(page_lines)  # Markdown: separate paragraphs by blank lines

        all_pages_text.append(page_text)
        all_pages_md.append(page_md)

        print(f"OCR Completed for Page {i+1}")

    final_text = "\n\n".join(all_pages_text)
    final_markdown = "\n\n---\n\n".join(all_pages_md)  # Add page breaks in markdown

    return final_text, final_markdown