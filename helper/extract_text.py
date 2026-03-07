from pathlib import Path


DEFAULT_TRANSCRIPT_CANDIDATES = ("data/transcript.txt", "data/book.pdf")


def resolve_transcript_source(transcript_path=None):
    if transcript_path:
        return Path(transcript_path)

    for candidate in DEFAULT_TRANSCRIPT_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            return path

    return Path(DEFAULT_TRANSCRIPT_CANDIDATES[0])


def prepare_transcript(transcript_path=None, output_path="generated/extracted_text.txt"):
    source_path = resolve_transcript_source(transcript_path)
    output_file = Path(output_path)

    if not source_path.exists():
        print(f"{source_path} not found")
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists():
        print(f"{output_file} already exists")
        return

    if source_path.suffix.lower() == ".txt":
        text = source_path.read_text(encoding="utf-8")
    elif source_path.suffix.lower() == ".pdf":
        import pypdf

        with source_path.open("rb") as file:
            reader = pypdf.PdfReader(file)
            text = "".join(page.extract_text() or "" for page in reader.pages)
    else:
        print(f"Unsupported transcript format: {source_path.suffix}")
        return

    output_file.write_text(text, encoding="utf-8")
    print(f"Transcript prepared from {source_path} and saved to {output_file}")


def extract_text_from_pdf(pdf_path=None, output_path="generated/extracted_text.txt"):
    prepare_transcript(transcript_path=pdf_path, output_path=output_path)
