from mistralai import Mistral, DocumentURLChunk, ImageURLChunk, TextChunk
from mistralai.models import OCRResponse
from typing import List
import base64
from pathlib import Path
import json
import tiktoken
from dotenv import load_dotenv
import os
import time

# Load environment variables from .env file
load_dotenv()  # By default looks for .env file in current directory

# ------ MISTRAL -------
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=MISTRAL_API_KEY)



def num_tokens_by_tiktoken(text: str) -> int:
    enc = tiktoken.encoding_for_model("gpt-4-turbo")
    return len(enc.encode(text))
  

def mistral_ocr(pdf_path) -> str:
  # load file
  uploaded_pdf = client.files.upload(
      file={
          "file_name": "uploaded_file.pdf",
          "content": open(pdf_path, "rb"),
      },
      purpose="ocr"
  )
  signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)

  # process pdf
  wait_time = 1
  it = 0
  while(it < 5):
    try:
      pdf_response = client.ocr.process(
          document=DocumentURLChunk(document_url=signed_url.url), 
          model="mistral-ocr-latest", 
          include_image_base64=True
      )
      break
    except:
      it+=1
      if wait_time == 1 :
        wait_time *= 2
      else:
        wait_time **2
      time.sleep(wait_time)
      continue
    return "failed"
    
  pdf_text = pdf_response.pages[0].markdown
  
  markdown, markdown_display = get_combined_markdown(pdf_response)
  return markdown, markdown_display
  # print("--------")
  # print("num input tokens: ")
  num_tokens = num_tokens_by_tiktoken(markdown)
  # print(num_tokens)
  # print("--------")

  if(num_tokens > 123000):
    print("num tokens too large")
    return


def replace_images_in_markdown(markdown_str: str, images_dict: dict) -> str:
  for img_name, base64_str in images_dict.items():
        # print("replacing images")
        header, encoded = base64_str.split(",", 1)
        base64_data_url = f"data:image/jpeg;base64,{encoded}"
        # Process image with OCR
        # process pdf
        wait_time = 1
        it = 0
        while(it < 5):
          try:
            image_response = client.ocr.process(
                document=ImageURLChunk(image_url=base64_data_url),
                model="mistral-ocr-latest"
            )
            break
          except:
            it+=1
            if wait_time == 1 :
              wait_time *= 2
            else:
              wait_time **2
            time.sleep(wait_time)
            continue
          return "failed"
        # Combine text from image and markdown and extract JSON metadata
        image_ocr_markdown = image_response.pages[0].markdown
        empty_str = ""
        markdown_str = markdown_str.replace(f"![{img_name}]({img_name})", f"![{empty_str}]({image_ocr_markdown})")
        # images_dict[img_name] = im
  # print("done replacing images")
  # print(markdown_str)
  return markdown_str

def get_combined_markdown(ocr_response: OCRResponse) -> str:
  markdowns: list[str] = []
  markdownToDisplay: list[str] = []
  for page in ocr_response.pages:
    image_data = {}
    for img in page.images:
      image_data[img.id] = img.image_base64
    markdowns.append(replace_images_in_markdown(page.markdown, image_data))
    markdownToDisplay.append(replace_images_in_markdown_to_display(page.markdown, image_data))
  # print("done getting combined markdown")
  # print(markdowns)

  return "\n\n".join(markdowns), "\n\n".join(markdownToDisplay)


"""
FUNCTIONS FOR DISPLAYING OCR'ED TEXT
"""


def replace_images_in_markdown_to_display(markdown_str: str, images_dict: dict) -> str:
    """
    Replace image placeholders in markdown with base64-encoded images.

    Args:
        markdown_str: Markdown text containing image placeholders
        images_dict: Dictionary mapping image IDs to base64 strings

    Returns:
        Markdown text with images replaced by base64 data
    """
    for img_name, base64_str in images_dict.items():
        markdown_str = markdown_str.replace(
            f"![{img_name}]({img_name})", f"![{img_name}]({base64_str})"
        )
    return markdown_str

def get_combined_markdown_to_display(ocr_response: OCRResponse) -> str:
    """
    Combine OCR text and images into a single markdown document.

    Args:
        ocr_response: Response from OCR processing containing text and images

    Returns:
        Combined markdown string with embedded images
    """
    markdowns: list[str] = []
    # Extract images from page
    for page in ocr_response.pages:
        image_data = {}
        for img in page.images:
            image_data[img.id] = img.image_base64
        # Replace image placeholders with actual images
        markdowns.append(replace_images_in_markdown_to_display(page.markdown, image_data))

    return "\n\n".join(markdowns)

def mistral_ocr_markdown(pdf_path) -> str:
  """
  pdf path - path to a valid pdf
  """
  # load file
  uploaded_pdf = client.files.upload(
      file={
          "file_name": "uploaded_file.pdf",
          "content": open(pdf_path, "rb"),
      },
      purpose="ocr"
  )
  signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)

  # process pdf
  wait_time = 1
  it = 0
  while(it < 5):
    try:
      pdf_response = client.ocr.process(
          document=DocumentURLChunk(document_url=signed_url.url),
          model="mistral-ocr-latest",
          include_image_base64=True
      )
      break
    except:
      it+=1
      if wait_time == 1 :
        wait_time *= 2
      else:
        wait_time **2
      time.sleep(wait_time)
      continue
    return "failed"

  pdf_text = pdf_response.pages[0].markdown

  # CHANGE THIS
  display(Markdown(get_combined_markdown_to_display(pdf_response)))
