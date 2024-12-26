#!/usr/bin/env python3

import argparse
import logging
import os
import sys
import tempfile
from typing import List, Tuple

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

try:
    from pypeln.thread import map as tmap
except ImportError:
    raise ImportError("Please install pypeln: pip install pypeln")


logger = logging.getLogger(__name__)

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
    logger.info(f"Uploaded file '{file.display_name}' as: {file.uri}")
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


def process_page(
    page_index: int, image: Image.Image, model: genai.GenerativeModel
) -> str:
    """
    Saves one PDF page to a temp file, uploads it to Gemini, and returns the transcription text.
    """
    tmp_path = _save_temp_image(page_index, image)
    gemini_file = upload_to_gemini(tmp_path, mime_type="image/png")
    user_instructions = (
        "transcribe the pdf page to markdown, including transcribing charts to tables, "
        "but only label complex figures as [FIGURE NOT INCLUDED]. "
        "Respond with an undecorated markdown string, appropriate for pasting into a markdown file."
    )
    chat_session = model.start_chat(history=[{"role": "user", "parts": [gemini_file]}])
    response = chat_session.send_message(user_instructions)
    txts_after_thinking = [part.text for part in response.parts[1:] if part.text]
    page_text = "\n".join(txts_after_thinking)
    try:
        os.remove(tmp_path)
    except OSError:
        logger.warning(f"Failed to remove temporary file {tmp_path}.")
    return page_text


def _save_temp_image(page_index: int, image: Image.Image) -> str:
    """
    Saves a page image to a temporary PNG file and returns the path.
    """
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        tmp_path = tmp_file.name
    image.save(tmp_path, "PNG")
    logger.info(f"Saved page {page_index} to temp file {tmp_path}")
    return tmp_path


def transcribe_pdf_pages(
    pdf_path: str,
    model: genai.GenerativeModel,
    debug: bool = False,
    workers_save: int = 4,
    workers_upload: int = 4,
    workers_chat: int = 4,
) -> List[str]:
    """
    Converts the given PDF to images (one per page), then calls the Gemini API
    for each page to transcribe to Markdown (with special handling for figures, etc.).

    :param pdf_path: The path to the PDF file.
    :return: A list of strings, each containing the Markdown transcription of one page.
    """
    images = pdf_to_images(pdf_path)
    transcribed_pages = []

    # If debug mode is ON, just process one page synchronously (save -> upload -> chat).
    if debug and images:
        logger.info("Debug mode enabled: only processing the first page (no threading).")
        page_index = 0
        tmp_path = _save_temp_image(page_index, images[page_index])
        gemini_file = upload_to_gemini(tmp_path, mime_type="image/png")
        user_instructions = (
            "transcribe the pdf page to markdown, including transcribing charts to tables, "
            "but only label complex figures as [FIGURE NOT INCLUDED]. "
            "Respond with an undecorated markdown string, appropriate for pasting into a markdown file."
        )
        chat_session = model.start_chat(history=[{"role": "user", "parts": [gemini_file]}])
        response = chat_session.send_message(user_instructions)
        txts_after_thinking = [part.text for part in response.parts[1:] if part.text]
        page_text = "\n".join(txts_after_thinking)
        transcribed_pages.append(page_text)
        try:
            os.remove(tmp_path)
        except OSError:
            logger.warning(f"Failed to remove temporary file {tmp_path}.")
        return transcribed_pages

    # Otherwise, do this in three stages:
    #  1) save_temp  -> returns (page_index, tmp_path)
    #  2) upload_gemini -> returns (page_index, gemini_file, tmp_path)
    #  3) chat_gemini   -> returns (page_index, page_text)

    def save_temp(args: Tuple[int, Image.Image]) -> Tuple[int, str]:
        page_index, img = args
        logger.info(f"Saving page {page_index} to temp file.")
        path = _save_temp_image(page_index, img)
        return page_index, path

    def upload_gemini_stage(args: Tuple[int, str]) -> Tuple[int, genai.types.File, str]:
        page_index, tmp_path = args
        logger.info(f"Uploading page {page_index} to Gemini.")
        gem_file = upload_to_gemini(tmp_path, mime_type="image/png")
        return page_index, gem_file, tmp_path

    def chat_gemini_stage(args: Tuple[int, genai.types.File, str]) -> Tuple[int, str]:
        page_index, gem_file, tmp_path = args
        logger.info(f"Chatting page {page_index} with Gemini.")
        user_instructions = (
            "transcribe the pdf page to markdown, including transcribing charts to tables, "
            "but only label complex figures as [FIGURE NOT INCLUDED]. "
            "Respond with an undecorated markdown string, appropriate for pasting into a markdown file."
        )
        chat_session = model.start_chat(history=[{"role": "user", "parts": [gem_file]}])
        response = chat_session.send_message(user_instructions)
        txts_after_thinking = [part.text for part in response.parts[1:] if part.text]
        page_text = "\n".join(txts_after_thinking)
        # Remove temp file after chat
        try:
            os.remove(tmp_path)
        except OSError:
            logger.warning(f"Failed to remove temporary file {tmp_path}.")
        return page_index, page_text

    # Create pipeline for stage 1: saving images
    stage1 = tmap(
        save_temp,
        enumerate(images),
        workers=workers_save,
        maxsize=len(images),
    )
    results_stage1 = list(tqdm(stage1, desc="Saving pages", total=len(images)))
    results_stage1.sort(key=lambda x: x[0])

    # Stage 2: uploading
    stage2 = tmap(
        upload_gemini_stage,
        results_stage1,
        workers=workers_upload,
        maxsize=len(images),
    )
    results_stage2 = list(tqdm(stage2, desc="Uploading pages", total=len(images)))
    results_stage2.sort(key=lambda x: x[0])

    # Stage 3: chat
    stage3 = tmap(
        chat_gemini_stage,
        results_stage2,
        workers=workers_chat,
        maxsize=len(images),
    )
    results_stage3 = list(tqdm(stage3, desc="Chatting pages", total=len(images)))
    results_stage3.sort(key=lambda x: x[0])

    transcribed_pages = [x[1] for x in results_stage3]
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
        help="If set, only process the first PDF page, synchronously (no threading).",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Path to an output file. If not specified, the transcription will be printed to stdout.",
    )
    parser.add_argument(
        "--workers_save",
        type=int,
        default=8,
        help="Number of workers for saving PDFs to temp images (default=8).",
    )
    parser.add_argument(
        "--workers_upload",
        type=int,
        default=1,
        help="Number of workers for uploading images to Gemini (default=2).",
    )
    parser.add_argument(
        "--workers_chat",
        type=int,
        default=8,
        help="Number of workers for chatting with Gemini (default=8).",
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
    pages_md = transcribe_pdf_pages(
        pdf_path=pdf_path,
        model=model,
        debug=args.debug,
        workers_save=args.workers_save,
        workers_upload=args.workers_upload,
        workers_chat=args.workers_chat,
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as out_file:
            for i, page_md in enumerate(pages_md, start=1):
                out_file.write(f"# Page {i}\n\n{page_md}\n\n\n")
        print(f"Transcription results saved to {args.output}")
    else:
        print("\n--- Transcription Results ---\n")
        for i, page_md in enumerate(pages_md, start=1):
            print(f"# Page {i}\n")
            print(page_md)
            print("\n\n")


if __name__ == "__main__":
    main()
