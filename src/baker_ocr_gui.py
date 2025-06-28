import gradio as gr
import tiktoken
from ocr.mistral import mistral_ocr
from ocr.textract import textract_ocr
import os
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
import openai
import json
import markdown2
import tempfile
import pypandoc
import csv
import time
import re
import uuid
from textwrap import wrap

# Load environment variables from .env file
load_dotenv() 

# ------ OPENAI -------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# ------ DEEPSEEK -------
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
deep_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

# ------ ANTHROPIC -------
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
claude = anthropic.Anthropic()

# ------ GEMINI -------
GEMINI_API_KEY = os.getenv('GOOGLE_API_KEY')
gemini_via_openai_client = OpenAI(
    api_key=GEMINI_API_KEY, 
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


def run_ocr(files, ocr_engine):
    """
    Process multiple PDF files and return their summaries.

    Args:
        files (list): List of PDF files as file objects
        ocr_engine (str): OCR engine to use (Mistral)

    Returns:
        str: Summaries of the PDF files
    """
    if not files:
        return "No files uploaded."

    ocr_response = ""  # Initialize empty string to concatenate all OCR responses
    markdown_response = ""

    for file in files:
        pdf_path = file.name
        print(f"Processing {pdf_path} with {ocr_engine} OCR...")

        # Perform OCR
        if ocr_engine == "Mistral":
            file_ocr_response, file_markdown_response = mistral_ocr(pdf_path)
        elif ocr_engine == "Textract":
            file_ocr_response, file_markdown_response = textract_ocr(pdf_path)

        if not file_ocr_response:
            return f"OCR failed for {pdf_path}."
        
        ocr_response += "\n\n" + file_ocr_response  # Concatenate the OCR result
        markdown_response += "\n\n" + file_markdown_response

        print(f"Processing {pdf_path} with {ocr_engine} OCR completed!")
        ocr_response += "\n\n" + file_ocr_response  # Concatenate the OCR result
        
    yield "OCR COMPLETED", ocr_response, markdown_response

def gpt_extract(ocr_response: str) -> str:
    system_prompt = "You are an assistant that specializes in filling json forms with OCR data. Please fill accurate entries in the fields provided and output a json file only!"
    user_prompt = f"This is a pdf's OCR in markdown:\n\n{ocr_response}\n.\n" + "Convert this into a sensible structured json response containing doc_id,  Title, Language, Subject, Format, Genre, Administration, People and Organizations, Time Span, Date, Summary"

    chat_response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.5,
    )

    # Parse and return JSON response
    response_dict = json.loads(chat_response.choices[0].message.content, strict=False)

    return json.dumps(response_dict, indent=4)

def split_by_char_count(text: str, max_chars: int = 3000):
    chunks = []
    current = 0
    while current < len(text):
        end = min(current + max_chars, len(text))
        
        # Optional: Try to split at the last newline before max_chars
        split_point = text.rfind('\n', current, end)
        if split_point == -1 or split_point <= current:
            split_point = end
        
        chunks.append(text[current:split_point].strip())
        current = split_point
    return chunks


def gpt_extract_to_csv(ocr_response: str, csv_path: str = "output.csv"):
    system_prompt = (
        "You are an assistant that extracts structured data from OCR text.\n"
        "From each page, extract all relevant people entries as CSV rows with exactly 7 fields in this order:\n\n"
        "1. Name of Person\n"
        "2. Person color (use \"c\" if indicated, otherwise \"null\")\n"
        "3. Address\n"
        "4. Residence type or room count (e.g., 'h', 'r', '3 rms', 'b', 'bds')\n"
        "5. Job title\n"
        "6. Job name (e.g., company or organization)\n"
        "7. Job address\n\n"
        "⚠️ Format rules:\n"
        "- Output only raw CSV rows, one per line\n"
        "- Each row must have exactly 7 fields\n"
        "- If a field is missing or unknown, write \"null\"\n"
        "- Do NOT include column headers, explanations, or extra commas inside fields\n"
        "- Output must be valid CSV (use double quotes for all fields)\n\n"
        "Example row:\n"
        "\"John Doe\",\"c\",\"123 Main St\",\"3 rms\",\"Manager\",\"ABC Co.\",\"456 Elm St\""
    )


    # Prepare CSV and write headers
    with open(csv_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Name", "(c)", "Address", "h/r/rms/b/bds", "Job", "Job Name", "Job Address"])

    # Split OCR response by page
    pages = split_by_char_count(ocr_response, max_chars=7000)

    for i, page_text in enumerate(pages):
        user_prompt = f"Page {i+1} OCR content:\n\n{page_text}\n\nExtract the data as a CSV row."

        try:
            chat_response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.4,
            )

            csv_output = chat_response.choices[0].message.content.strip()
            print(csv_output)  # Log full response
            rows = list(csv.reader(csv_output.splitlines()))
            valid_rows = [row for row in rows if len(row) == 7]

            if not valid_rows:
                print(f"⚠️ Page {i+1} returned no valid rows.")
            else:
                with open(csv_path, mode='a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(valid_rows)
                print(f"✅ Processed page {i+1} with {len(valid_rows)} row(s).")

        except Exception as e:
            print(f"Error processing page {i+1}: {e}")

        time.sleep(5)  # Delay between requests

    return json.dumps({"status": "CSV extraction completed", "file": csv_path}, indent=4)

def deepseek_extract(ocr_response, file_id=None, csv_path="deepseek_benchmark.csv", chunk_size=10000):
    system_prompt = (
        "You are an assistant that fills structured JSON forms based on OCR text input. "
        "You must return a single, valid JSON object with only the specified fields. "
        "Do not include any explanation, markdown, or formatting like triple backticks. "
        "If a field is missing or unknown, set its value to null."
    )

    # Define all expected fields
    expected_fields = [
        "Collection(s)", "Published URL", "Title", "People and Organizations",
        "Description", "Language", "Publisher", "Date", "Source",
        "Subject", "Format Genre", "Accessibility Summary"
    ]

    # Fields that should have only one entry (not appended)
    single_value_fields = [
        "Published URL", "Title", "Publisher", "Date", 
        "Source", "Format Genre"
    ]

    # Initialize response dictionary
    combined_response = {field: None for field in expected_fields}

    # Split OCR text into chunks while preserving paragraphs
    paragraphs = [p.strip() for p in ocr_response.split('\n\n') if p.strip()]
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 <= chunk_size:
            current_chunk += f"\n\n{para}" if current_chunk else para
        else:
            chunks.append(current_chunk)
            current_chunk = para
    if current_chunk:
        chunks.append(current_chunk)

    # Process each chunk
    for i, chunk in enumerate(chunks):
        user_prompt = (
            f"This is part {i+1}/{len(chunks)} of OCR content:\n\n{chunk}\n\n"
            "Extract or update these fields:\n" +
            "\n".join([f" - {field}" for field in expected_fields]) +
            "\nFor Title, Published URL, Publisher, Date, Source, and Format Genre - "
            "keep only the most relevant single value.\n"
            "For other fields, you may append multiple values separated by commas.\n"
            "⚠️ Return ONLY JSON with updated fields. Keep null for missing data."
        )

        try:
            chat_response = deep_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )

            content = chat_response.choices[0].message.content
            chunk_data = json.loads(content.strip('` \n'))

            # Merge results intelligently
            for field in expected_fields:
                if field in chunk_data and chunk_data[field]:
                    # For single-value fields, keep the first valid value
                    if field in single_value_fields:
                        if not combined_response[field]:
                            combined_response[field] = chunk_data[field]
                    # For list-like fields, append new values
                    else:
                        if combined_response[field]:
                            if isinstance(combined_response[field], str):
                                combined_response[field] += ", " + str(chunk_data[field])
                            elif isinstance(combined_response[field], list):
                                combined_response[field].extend(chunk_data[field])
                        else:
                            combined_response[field] = str(chunk_data[field])

        except Exception as e:
            print(f"⚠️ Error processing chunk {i+1}: {str(e)}")
            continue

    # Clean up the response
    for field in combined_response:
        if combined_response[field] is None:
            combined_response[field] = "Null"
        elif isinstance(combined_response[field], str) and combined_response[field].startswith("['") and combined_response[field].endswith("']"):
            # Remove list formatting if present
            combined_response[field] = combined_response[field][2:-2]

    # CSV handling
    unique_id = file_id or str(uuid.uuid4())
    fields = ["Unique ID"] + expected_fields

    row = [unique_id] + [
        combined_response[field] if combined_response[field] != "Null" else "Null"
        for field in expected_fields
    ]

    # Write to CSV
    write_headers = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    prepend_newline = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0

    with open(csv_path, mode="a", newline='', encoding="utf-8") as f:
        if prepend_newline:
            f.write("\n")
        writer = csv.writer(f, lineterminator="\n")
        if write_headers:
            writer.writerow(fields)
        writer.writerow(row)

    print(f"✅ Appended entry to {csv_path} with ID: {unique_id}")
    return json.dumps(combined_response, indent=4)

def deepseek_extract_to_csv(ocr_response: str, csv_path: str = "output.csv"):
    system_prompt = (
        "You are an assistant that extracts structured data from OCR text.\n"
        "From each page, extract all relevant individual person records as CSV rows with exactly 7 fields in this order:\n\n"
        "1. Full Name of Person\n"
        "2. Person color (use \"c\" if indicated, otherwise \"null\")\n"
        "3. Address\n"
        "4. Residence type or room count (e.g., 'h', 'r', '3 rms', 'b', 'bds')\n"
        "5. Job title\n"
        "6. Job name (e.g., company or organization)\n"
        "7. Job address\n\n"
        "⚠️ Format rules:\n"
        "- Output only raw CSV rows, one per line\n"
        "- Each row must have exactly 7 fields\n"
        "- If a field is missing or unknown, write \"null\"\n"
        "- Do NOT include column headers, explanations, or extra commas inside fields\n"
        "- Output must be valid CSV (use double quotes for all fields)\n\n"
        "Example row:\n"
        "\"John Doe\",\"c\",\"123 Main St\",\"3 rms\",\"Manager\",\"ABC Co.\",\"456 Elm St\""
    )

    # Prepare CSV and write headers
    with open(csv_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Name", "(c)", "Address", "h/r/rms/b/bds", "Job", "Job Name", "Job Address"])

    # Split OCR response by page
    pages = split_by_char_count(ocr_response, max_chars=4000)

    for i, page_text in enumerate(pages):
        user_prompt = f"Page {i+1} OCR content:\n\n{page_text}\n\nExtract the data as a CSV row."

        try:
            chat_response = deep_client.chat.completions.create(
                model="deepseek-chat",  # or another model name
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,  # Lower for more deterministic output
            )

            csv_output = chat_response.choices[0].message.content.strip()
            print(csv_output)  # Log full response
            rows = list(csv.reader(csv_output.splitlines()))
            valid_rows = [row for row in rows if len(row) == 7]

            if not valid_rows:
                print(f"⚠️ Page {i+1} returned no valid rows.")
            else:
                with open(csv_path, mode='a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(valid_rows)
                print(f"✅ Processed page {i+1} with {len(valid_rows)} row(s).")

        except Exception as e:
            print(f"Error processing page {i+1}: {e}")

        time.sleep(5)  # Delay between requests

    return json.dumps({"status": "CSV extraction completed", "file": csv_path}, indent=4)

def claude_extract(ocr_response):
    system_prompt = "You are an assistant that specializes in filling json forms with OCR data. Please fill accurate entries in the fields provided and output a json file only!"
    user_prompt = f"This is a pdf's OCR in markdown:\n\n{ocr_response}\n.\n" + "Convert this into a sensible structured json response containing doc_id,  Title, Language, Subject, Format, Genre, Administration, People and Organizations, Time Span, Date, Summary"
    
    chat_response = claude.messages.create(
        model="claude-3-5-sonnet-latest",
        max_tokens=1000,
        temperature=0.7,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_prompt},
        ]
    )
    
    # Parse and return JSON response
    response_dict = json.loads(chat_response.content[0].text)

    return json.dumps(response_dict, indent=4)


def itemize_with_gemini(ocr_response):
    print("Itemization has started!")
    system_prompt = """You are an expert document structuring assistant. Your task is to analyze long OCR-scanned text and intelligently group related content into logical sections. 
    First identify logical sections or documents, then group related pages/content together, finally label those sections intelligently. For each section, extract structured information and present it in markdown format. Only return in markdown format. Please make sure no information is lost, everything from the ocr version should be included."""
    user_prompt = f"This is a pdf's OCR in markdown:\n\n{ocr_response}\n.\n" + "Convert this into a sensible itemized markdown version of the document."

    prompts = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    response = gemini_via_openai_client.chat.completions.create(
        model="gemini-2.0-flash-exp",
        messages=prompts
    )

    return response.choices[0].message.content

def gemini_extract(ocr_response, file_id=None, csv_path="gemini_benchmark.csv"):
    system_prompt = (
        "You are an assistant that fills structured JSON forms based on OCR text input. "
        "You must return a single, valid JSON object with only the specified fields. "
        "Do not include any explanation, markdown, or formatting like triple backticks. "
        "If a field is missing or unknown, set its value to null."
    )

    user_prompt = (
        f"This is OCR content extracted from a document:\n\n{ocr_response}\n\n"
        "Please convert this into one structured JSON object with exactly the following fields:\n"
        " - Collection(s)\n"
        " - Published URL\n"
        " - Title\n"
        " - People and Organizations\n"
        " - Description\n"
        " - Language\n"
        " - Publisher\n"
        " - Date\n"
        " - Source\n"
        " - Subject\n"
        " - Format Genre\n"
        " - Accessibility Summary\n"
        "⚠️ Output one valid JSON object only. No markdown, no backticks, no explanation. "
        "If a field is missing, use null."
    )

    prompts = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    chat_response = gemini_via_openai_client.chat.completions.create(
        model="gemini-2.0-flash-exp",
        messages=prompts
    )

    content = chat_response.choices[0].message.content
    print("🔍 Gemini raw response:\n", content)

    def strip_code_fence(text):
        lines = text.strip().splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()

    cleaned = strip_code_fence(content)

    try:
        response_dict = json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing failed: {e}")
        return json.dumps({"error": "Model returned invalid JSON", "raw_output": cleaned}, indent=4)

    unique_id = file_id or str(uuid.uuid4())

    # Define CSV column headers
    fields = [
        "Unique ID", "Collection(s)", "Published URL", "Title", "People and Organizations",
        "Description", "Language", "Publisher", "Date", "Source",
        "Subject", "Format Genre", "Accessibility Summary"
    ]

    # Prepare CSV row with "Null" if value is None or null
    row = [unique_id] + [
        value if value is not None and value != "null" else "Null"
        for field in fields[1:]
        for key, value in response_dict.items() if key.strip() == field
    ]

    # Fallback for any missing keys not matched in the above (ensures fixed ordering)
    if len(row) < len(fields):
        field_dict = {k.strip(): v for k, v in response_dict.items()}
        for field in fields[1 + len(row) - 1:]:
            row.append(field_dict.get(field, "Null") or "Null")

    # Check if file exists and has headers already
    write_headers = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0

    prepend_newline = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0

    with open(csv_path, mode="a", newline='', encoding="utf-8") as f:
        if prepend_newline:
            f.write("\n")
        writer = csv.writer(f, lineterminator="\n")
        if write_headers:
            writer.writerow(fields)
        writer.writerow(row)

    print(f"✅ Appended entry to {csv_path} with ID: {unique_id}")
    return json.dumps(response_dict, indent=4)


def markdown_to_pdf(markdown_text):
    # Convert markdown to HTML
    html = markdown2.markdown(markdown_text)
    print("Converting the markdown to PDF!")
    # Add PDF-specific formatting options
    extra_args = [
        '--pdf-engine=xelatex',  # Better unicode support
        '--variable', 'geometry:margin=1in',  # Reasonable margins
        '--variable', 'geometry:a4paper',  # Standard paper size
        '--variable', 'mainfont=Helvetica'  # Ensures proper font handling
    ]
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        pypandoc.convert_text(
            markdown_text,
            'pdf',
            format='md',
            outputfile=tmp_pdf.name,
            extra_args=extra_args
        )
        return tmp_pdf.name

def extract_metadata(ocr_response, llm_engine):
    """
    Extract metadata from the OCR response based on the selected LLM engine.

    Args:
        ocr_response (dict): OCR response from the OCR engine
        llm_engine (str): LLM engine to use (DeepSeek, GPT-4, or Claude)

    Returns:
        dict: Metadata extracted from the OCR response
    """
    if not ocr_response:
        return "No response provided."

    print(f"Extracting metadata using {llm_engine}...")

    if llm_engine == "DeepSeek":
        metadata = deepseek_extract(ocr_response)
    elif llm_engine == "GPT-4":
        metadata = gpt_extract(ocr_response)
    elif llm_engine == "Claude":
        metadata = claude_extract(ocr_response)
    elif llm_engine == "Gemini":
        metadata = gemini_extract(ocr_response)
    
    if not metadata:
        return f"Failed to extract metadata using {llm_engine}."
        
    print(f"Metadata extraction completed!")
    return metadata

def main():
    with gr.Blocks(css=".gr-button { font-size: 16px !important; } .gr-dropdown, .gr-textbox { font-size: 15px !important; }") as demo:
        gr.Markdown("""
        # 🧾 OCR & Metadata Extraction Tool
        
        Welcome! This tool helps you:
        1. **Extract text** from PDF files using a selected OCR engine
        2. **Convert** that text into structured metadata using an LLM
        
        ---
        """)

        ocr_done_state = gr.State()

        # --- Step 1: Upload PDFs and select OCR engine ---
        gr.Markdown("### 📄 Step 1: Upload PDF and Select OCR Engine")

        with gr.Row():
            files_input = gr.Files(
                label="📂 Upload PDF(s)", 
                type="filepath", 
                file_types=[".pdf"]
            )

            engine_dropdown = gr.Dropdown(
                choices=["Mistral", "Textract"],
                label="🧠 OCR Engine",
                info="Choose the engine for text extraction",
                interactive=True
            )

        run_ocr_btn = gr.Button("▶️ Run OCR", variant="primary")

        ocr_status = gr.Textbox(
            label="Status",
            interactive=False,
            visible=True,
            show_label=False
        )

        # Add markdown display for OCR results
        with gr.Accordion("📝 View OCR Results in Markdown", open=False):
            markdown_display = gr.Markdown(label="OCR Results")

        # --- Step 2A: Itemize Document with Gemini ---
        gr.Markdown("### 📊 Step 2A: Itemize Document with Gemini")

        itemize_btn = gr.Button("📋 Itemize with Gemini", interactive=False)
        download_pdf = gr.File(label="📥 Download Itemized PDF", visible=False)

        # --- Hidden: Only needed internally for markdown passing ---
        itemized_markdown = gr.Textbox(visible=False)

        # --- Step 2B: Metadata Extraction ---
        gr.Markdown("### 🧠 Step 2B: Extract Metadata with an LLM")

        llm_dropdown = gr.Dropdown(
            choices=["DeepSeek", "GPT-4", "Claude", "Gemini"],
            label="🤖 LLM Engine",
            interactive=False,
            info="Select a model to generate structured metadata"
        )

        extract_btn = gr.Button("📤 Extract Metadata (JSON)", interactive=False)

        with gr.Accordion("🧾 View Extracted Metadata", open=False):
            json_output = gr.JSON(label="Structured Metadata")

        # --- Wiring ---
        run_ocr_btn.click(
            fn=run_ocr,
            inputs=[files_input, engine_dropdown],
            outputs=[ocr_status, ocr_done_state, markdown_display]
        ).then(
            fn=lambda: gr.update(interactive=False),  # Disable the 'Run OCR' button as soon as it's clicked
            inputs=None,
            outputs=[run_ocr_btn]
        ).then(
            fn=lambda: (gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)),  # Enable the other buttons (LLM dropdown, Itemize button, Extract button)
            inputs=None,
            outputs=[llm_dropdown, extract_btn, itemize_btn]
        ).then(
            fn=lambda: gr.update(interactive=True),  # Re-enable the 'Run OCR' button once OCR is completed
            inputs=None,
            outputs=[run_ocr_btn]
        )

        itemize_btn.click(
            fn=itemize_with_gemini,
            inputs=[ocr_done_state],
            outputs=[itemized_markdown]
        ).then(
            fn=markdown_to_pdf,
            inputs=[itemized_markdown],
            outputs=[download_pdf]
        ).then(
            fn=lambda: gr.update(visible=True),  # Make the download button visible
            inputs=None,
            outputs=[download_pdf]
        ).then(
            fn=lambda: gr.update(interactive=True),  # Enable the itemize button after process completion
            inputs=None,
            outputs=[itemize_btn]
        )

        extract_btn.click(
            fn=extract_metadata,
            inputs=[ocr_done_state, llm_dropdown],
            outputs=[json_output]
        ).then(
            fn=lambda: gr.update(interactive=True),  # Enable the extract button once the extraction is completed
            inputs=None,
            outputs=[extract_btn]
        )
    demo.launch(share=True)

    
if __name__ == "__main__":
    main()
