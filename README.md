# PDF Transcription with Gemini

This project provides a command-line application for transcribing PDFs using Google's Gemini model.  
[Example output here.](examples/deepseek.txt)
It converts PDF pages into images, uploads them to Gemini, and returns transcriptions in Markdown format.
Dependencies handled by uv, see pyproject.toml.

## Usage

Run the script with the path to the PDF and any additional arguments you need:

```bash
./main.py --api_key=YOUR_GEMINI_API_KEY path/to/input.pdf \
    --output=transcription.md \
    --page_limit=5
```

Command-line arguments include:
- `--api_key`: Your Gemini API key (optional if set in the environment or in a `.api_key` file).
- `--page_limit`: Process up to N pages (0 = no limit).
- `--output` / `-o`: Output file path (if not provided, prints to stdout).
- `--workers_save`, `--workers_upload`, `--workers_chat`: Number of parallel workers for each stage.
- `--no_overlap`: Run each stage sequentially instead of pipelined.
- `--thinking`: Use the “thinking” model variant with a reduced rate limit.

## Contributing

If you have any fixes or improvements, feel free to open a pull request or submit an issue.

## License

MIT License. See LICENSE file for details.