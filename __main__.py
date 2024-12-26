#!/usr/bin/env python3

import argparse
import os
import sys
import tempfile
from typing import List, Tuple
from argparse import Namespace

try:
    import pypeln as pl
except ImportError:
    raise ImportError("Please install pypeln: pip install pypeln")

try:
    import google.generativeai as genai
except ImportError:
    raise ImportError(
        "Please install google-generativeai to use Gemini: pip install google-generativeai"
    )

try:
    from pdf2image import convert_from_path
except ImportError:
    raise ImportError("Please install pdf2image: pip install pdf2image")

try:
    from tqdm import tqdm
except ImportError:
    raise ImportError("Please install tqdm: pip install tqdm")

try:
    from PIL import Image
except ImportError:
    raise ImportError("Please install PIL: pip install Pillow")


###############################################################################
# Helper Function: Upload a file to Gemini
###############################################################################


def upload_to_gemini(path: str, mime_type: str = None):
    """
    Uploads the given file to Gemini.
    See https://ai.google.dev/gemini-api/docs/prompting_with_media

    :param path: Path to the local file you want to upload.
    :param mime_type: MIME type (e.g., "image/png", "image/jpg").
    :return: A Gemini File object (with .uri, .display_name, etc.).
    """
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file


###############################################################################
# PDF to Image Conversion
###############################################################################


def pdf_to_images(pdf_path: str, dpi: int = 300) -> List[Image.Image]:
    """
    Convert a PDF to a list of PIL Images, one per page.
    Adjust the DPI higher or lower depending on clarity/performance trade-offs.

    :param pdf_path: Path to the PDF file.
    :param dpi: Resolution (dots-per-inch) for the conversion.
    :return: A list of PIL Image objects, one per page.
    """
    return convert_from_path(pdf_path, dpi=dpi)


###############################################################################
# Main Transcription Logic
###############################################################################


def transcribe_pdf_pages(pdf_path: str, model: genai.GenerativeModel, debug: bool = False) -> List[str]:
    """
    Converts the given PDF to images (one per page), then calls the Gemini API
    for each page to transcribe to Markdown (with special handling for figures, etc.).

    :param pdf_path: The path to the PDF file.
    :return: A list of strings, each containing the Markdown transcription of one page.
    """
    # Convert PDF to images
    images = pdf_to_images(pdf_path)
    transcribed_pages = []

    def save_temp(args: Tuple[int, Image.Image]) -> Tuple[int, str]:
        """
        Saves a page image to a temporary PNG file.
        Returns (page_index, tmp_path).
        """
        page_index, image = args
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            tmp_path = tmp_file.name
        image.save(tmp_path, "PNG")
        return page_index, tmp_path

    def send_gemini(args: Tuple[int, str]) -> Tuple[int, str]:
        """
        Uploads the temporary PNG file to Gemini, gets transcription.
        Cleans up the temporary file. Returns (page_index, page_text).
        """
        page_index, tmp_path = args
        gemini_file = upload_to_gemini(tmp_path, mime_type="image/png")
        user_instructions = (
            "transcribe the pdf page to markdown, including transcribing charts to tables, "
            "but only label complex figures as [FIGURE NOT INCLUDED]."
        )
        chat_session = model.start_chat(
            history=[{"role": "user", "parts": [gemini_file]}]
        )
        response = chat_session.send_message(user_instructions)
        page_text = response.text

        try:
            os.remove(tmp_path)
        except OSError:
            pass

        return page_index, page_text

    # If debug mode is ON, process only the first page, synchronously.
    if debug and images:
        page_index = 0
        # Save temp
        print(f"Saving temp file for page {page_index}")
        _, tmp_path = save_temp((page_index, images[page_index]))
        print(f"Saved temp file to {tmp_path}")

        # Upload and transcribe
        print(f"Uploading temp file to Gemini for page {page_index}")
        gemini_file = upload_to_gemini(tmp_path, mime_type="image/png")
        print(f"Uploaded temp file to Gemini for page {page_index}")
        user_instructions = (
            "transcribe the pdf page to markdown, including transcribing charts to tables, "
            "but only label complex figures as [FIGURE NOT INCLUDED]."
        )
        chat_session = model.start_chat(
            history=[{"role": "user", "parts": [gemini_file]}]
        )
        print(f"Sending user instructions to Gemini for page {page_index}")
        response = chat_session.send_message(user_instructions)
        print(f"Received response from Gemini for page {page_index}")
        page_text = response.text
        transcribed_pages.append(page_text)

        # Cleanup
        try:
            os.remove(tmp_path)
        except OSError:
            pass

        return transcribed_pages

    # 1) Save each PDF page to a temp file in a background thread.
    stage1 = pl.thread.map(
        save_temp,
        enumerate(images),
        workers=1,
        maxsize=len(images),
    )

    # 2) Use multithreading to upload the temp files to Gemini and process them.
    stage2 = pl.thread.map(
        send_gemini,
        stage1,
        workers=4,
        maxsize=len(images),
    )

    # Collect results and sort them by page_index before returning.
    results = list(tqdm(stage2, desc="Transcribing pages", total=len(images)))
    results.sort(key=lambda x: x[0])
    transcribed_pages = [x[1] for x in results]

    return transcribed_pages


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe a PDF file with Gemini.")
    parser.add_argument("pdf_path", help="Path to the PDF file to be transcribed.")
    parser.add_argument(
        "--api_key",
        help="Gemini API key. If not provided, uses GEMINI_API_KEY environment variable.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="If set, only process the first PDF page, synchronously (no threading)."
    )

    args = parser.parse_args()

    # Prefer the command-line API key if provided, otherwise fall back to environment.
    if args.api_key:
        api_key = args.api_key
    elif os.path.exists(".api_key"):
        with open(".api_key", "r") as f:
            api_key = f.read()
    else:
        api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "No API key provided. Either pass --api_key or set GEMINI_API_KEY."
        )

    # Configure gemini here, after parsing API key:
    genai.configure(api_key=api_key)

    # Create the Gemini model configuration
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash-thinking-exp-1219",
        generation_config=generation_config,
    )

    pdf_path = args.pdf_path
    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    print(f"Transcribing PDF: {pdf_path}")
    pages_md = transcribe_pdf_pages(pdf_path, model, debug=args.debug)

    print("\n--- Transcription Results ---\n")
    for i, page_md in enumerate(pages_md, start=1):
        print(f"# Page {i}\n")
        print(page_md)
        print("\n\n")


if __name__ == "__main__":
    main()
