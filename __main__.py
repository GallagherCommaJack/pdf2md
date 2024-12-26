#!/usr/bin/env python3

import argparse
import atexit
import logging
import os
import sys
import tempfile
import time
from typing import List, Tuple

# Third-party imports
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

# Set up logging (you can change the level as needed)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

###############################################################################
# Global list for temporary files and atexit cleanup
###############################################################################
tmp_files = []


def _cleanup_tmp_files():
    """Remove all temporary files registered in tmp_files."""
    for path in tmp_files:
        try:
            os.remove(path)
            logger.info(f"Removed temporary file: {path}")
        except OSError:
            logger.warning(f"Failed to remove temporary file: {path}")


# Register the cleanup to run at script exit
atexit.register(_cleanup_tmp_files)

###############################################################################
# Argument parsing and main setup
###############################################################################


def parse_args():
    parser = argparse.ArgumentParser(description="Transcribe a PDF file with Gemini.")

    parser.add_argument("pdf_path", help="Path to the PDF file to be transcribed.")

    parser.add_argument(
        "--api_key",
        help="Gemini API key. If not provided, uses GEMINI_API_KEY environment variable or .api_key file.",
    )
    parser.add_argument(
        "--page_limit",
        type=int,
        default=0,
        help="If set, only process the first N pages. Use 0 for no limit.",
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
        default=4,
        help="Number of workers for chatting with Gemini (default=8).",
    )
    parser.add_argument(
        "--no_overlap",
        action="store_true",
        default=False,
        help="If specified, fully finishes each stage before starting the next.",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        default=False,
        help="If true, uses gemini-2.0-flash-thinking-exp-1219 and ignores the first response part. Otherwise uses gemini-1.5-flash and uses all parts.",
    )
    parser.add_argument(
        "--thinking_rate_limit",
        type=int,
        default=10,
        help="Max pages/min when thinking is True (default=10)."
    )

    return parser.parse_args()


def main():
    # -------------------------------------------------------------------------
    # 1) Parse command line arguments
    # -------------------------------------------------------------------------
    args = parse_args()

    # -------------------------------------------------------------------------
    # 2) Resolve API key
    # -------------------------------------------------------------------------
    if args.api_key:
        api_key = args.api_key
    elif os.path.exists(".api_key"):
        with open(".api_key", "r") as f:
            api_key = f.read().strip()
    else:
        api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "No API key provided. Either pass --api_key or set GEMINI_API_KEY."
        )

    # -------------------------------------------------------------------------
    # 3) Configure the Gemini client
    # -------------------------------------------------------------------------
    genai.configure(api_key=api_key)

    # Decide which model name to use based on --thinking
    # NOTE: thinking only supports 10 pages / minute
    if args.thinking:
        model_name = "gemini-2.0-flash-thinking-exp-1219"
    else:
        model_name = "gemini-1.5-flash"

    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 32,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config=generation_config,
    )

    # -------------------------------------------------------------------------
    # 4) Verify the PDF path
    # -------------------------------------------------------------------------
    pdf_path = args.pdf_path
    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 5) Run transcription pipeline
    # -------------------------------------------------------------------------
    print(f"Transcribing PDF: {pdf_path}")
    pages_md = transcribe_pdf_pages(
        pdf_path=pdf_path,
        model=model,
        page_limit=args.page_limit,
        workers_save=args.workers_save,
        workers_upload=args.workers_upload,
        workers_chat=args.workers_chat,
        no_overlap=args.no_overlap,
        thinking=args.thinking,
        thinking_rate_limit=args.thinking_rate_limit,
    )

    # -------------------------------------------------------------------------
    # 6) Output results
    # -------------------------------------------------------------------------
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


###############################################################################
# Helper functions (defined AFTER argument parsing & setup)
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


def pdf_to_images(pdf_path: str, dpi: int = 300) -> List[Image.Image]:
    """
    Convert a PDF to a list of PIL Images, one per page.
    Adjust the DPI higher or lower depending on clarity/performance trade-offs.

    :param pdf_path: Path to the PDF file.
    :param dpi: Resolution (dots-per-inch) for the conversion.
    :return: A list of PIL Image objects, one per page.
    """
    return convert_from_path(pdf_path, dpi=dpi)


def _save_temp_image(page_index: int, image: Image.Image) -> str:
    """
    Saves a page image to a temporary PNG file and returns the path.
    """
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        tmp_path = tmp_file.name
    image.save(tmp_path, "PNG")
    tmp_files.append(tmp_path)
    logger.info(f"Saved page {page_index} to temp file {tmp_path}")
    return tmp_path


def process_page(
    page_index: int,
    image: Image.Image,
    model: genai.GenerativeModel,
    thinking: bool,
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

    # If we're "thinking", skip the first chunk (as in the original code).
    # Otherwise, use all response parts.
    if thinking:
        relevant_parts = response.parts[1:]
    else:
        relevant_parts = response.parts

    txts = [part.text for part in relevant_parts if part.text]
    page_text = "\n".join(txts)

    return page_text


def _throttle_iter(iterable, items_per_minute: int):
    """
    Enforce a maximum rate of items_per_minute for the given iterable.
    """
    delay = 60.0 / items_per_minute
    for item in iterable:
        yield item
        time.sleep(delay)


def transcribe_pdf_pages(
    pdf_path: str,
    model: genai.GenerativeModel,
    page_limit: int = 0,
    workers_save: int = 4,
    workers_upload: int = 4,
    workers_chat: int = 4,
    no_overlap: bool = False,
    thinking: bool = False,
    thinking_rate_limit: int = 10,
) -> List[str]:
    """
    Converts the given PDF to images (one per page), then calls the Gemini API
    for each page to transcribe to Markdown.

    This version can either pipeline the 3 stages (save image, upload to gemini, chat)
    or run them sequentially if --no_overlap is specified.
    """
    images = pdf_to_images(pdf_path)
    if page_limit > 0:
        logger.info(f"Limiting to first {page_limit} pages.")
        images = images[:page_limit]

    # If no_overlap is True, do each stage sequentially
    logger.info("Running pipeline with overlap using pypeln.")

    # Stage 1: save images
    def save_temp(args: Tuple[int, Image.Image]) -> Tuple[int, str]:
        page_index, img = args
        return page_index, _save_temp_image(page_index, img)

    # Stage 2: upload
    def upload_gemini_stage(args: Tuple[int, str]) -> Tuple[int, genai.types.File]:
        page_index, tmp_path = args
        gem_file = upload_to_gemini(tmp_path, mime_type="image/png")
        return page_index, gem_file

    # Stage 3: chat
    def chat_gemini_stage(args: Tuple[int, genai.types.File]) -> Tuple[int, str]:
        page_index, gem_file = args
        user_instructions = (
            "transcribe the pdf page to markdown, including transcribing charts to tables, "
            "but only label complex figures as [FIGURE NOT INCLUDED]. "
            "Respond with an undecorated markdown string, appropriate for pasting into a markdown file."
        )
        chat_session = model.start_chat(history=[{"role": "user", "parts": [gem_file]}])
        response = chat_session.send_message(user_instructions)
        if thinking:
            relevant_parts = response.parts[1:]
        else:
            relevant_parts = response.parts
        txts = [part.text for part in relevant_parts if part.text]
        page_text = "\n".join(txts)

        return page_index, page_text

    # Create pipeline for stage 1
    stage1 = pl.thread.map(
        save_temp,
        enumerate(images),
        workers=workers_save,
        maxsize=len(images),
    )
    if no_overlap:
        stage1 = list(tqdm(stage1, desc="Saving pages", total=len(images)))
        stage1.sort(key=lambda x: x[0])

    # Stage 2
    stage2 = pl.thread.map(
        upload_gemini_stage,
        stage1,
        workers=workers_upload,
        maxsize=len(images),
    )
    if no_overlap:
        stage2 = list(tqdm(stage2, desc="Uploading pages", total=len(images)))
        stage2.sort(key=lambda x: x[0])

    # Enforce the thinking rate limit by throttling the iterator
    if thinking and thinking_rate_limit > 0:
        stage2 = list(stage2)  # collect results to avoid partial consumption
        stage2 = _throttle_iter(stage2, thinking_rate_limit)

    # Stage 3
    stage3 = pl.thread.map(
        chat_gemini_stage,
        stage2,
        workers=workers_chat,
        maxsize=len(images),
    )
    results_stage3 = list(tqdm(stage3, desc="Transcribing pages", total=len(images)))
    results_stage3.sort(key=lambda x: x[0])

    transcribed_pages = [x[1] for x in results_stage3]
    return transcribed_pages


if __name__ == "__main__":
    main()
