import argparse
import csv
import json
import re
import shutil
from pathlib import Path

from helper import align_audio, prepare_transcript, prepare_voice_audio, preprocess_data


AUDIO_EXTENSIONS = {".wav", ".mp3", ".mp4", ".m4a", ".flac", ".ogg", ".aac"}
TRANSCRIPT_EXTENSIONS = {".txt", ".pdf"}
GENERATED_DIR = Path("generated")
ITEMS_DIR = GENERATED_DIR / "items"
WAVS_DIR = GENERATED_DIR / "wavs"
DATASET_PATH = GENERATED_DIR / "dataset.csv"
PROGRESS_PATH = GENERATED_DIR / "progress.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a resumable voice dataset from folders of recordings and matching transcripts."
    )
    parser.add_argument(
        "--voice-path",
        default="data/voices",
        help="Folder containing the recordings you want to fine-tune on.",
    )
    parser.add_argument(
        "--transcript-path",
        default="data/transcripts",
        help="Folder containing matching transcript files.",
    )
    parser.add_argument(
        "--gentle-url",
        default="http://localhost:8765/transcriptions?async=false",
        help="Gentle alignment endpoint.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing generated dataset artifacts before rebuilding them from the beginning.",
    )
    return parser.parse_args()


def slugify(value):
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "item"


def clear_generated_outputs():
    if GENERATED_DIR.exists():
        shutil.rmtree(GENERATED_DIR)


def list_supported_files(folder_path, extensions):
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"{folder} not found")
    if not folder.is_dir():
        raise NotADirectoryError(f"{folder} is not a folder")

    files = [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in extensions]
    return sorted(files, key=lambda path: path.name.lower())


def build_pairs(audio_dir, transcript_dir):
    audio_files = list_supported_files(audio_dir, AUDIO_EXTENSIONS)
    transcript_files = list_supported_files(transcript_dir, TRANSCRIPT_EXTENSIONS)

    if not audio_files:
        raise RuntimeError(f"No supported recordings found in {audio_dir}")
    if not transcript_files:
        raise RuntimeError(f"No supported transcripts found in {transcript_dir}")
    if len(audio_files) != len(transcript_files):
        raise RuntimeError(
            f"Recording count ({len(audio_files)}) does not match transcript count ({len(transcript_files)})."
        )

    audio_by_stem = {path.stem: path for path in audio_files}
    transcript_by_stem = {path.stem: path for path in transcript_files}

    pairs = []
    if len(audio_by_stem) == len(audio_files) and len(transcript_by_stem) == len(transcript_files):
        if set(audio_by_stem) == set(transcript_by_stem):
            for stem in sorted(audio_by_stem):
                item_id = slugify(stem)
                pairs.append(
                    {
                        "item_id": item_id,
                        "audio": audio_by_stem[stem],
                        "transcript": transcript_by_stem[stem],
                    }
                )
            return pairs

    print("Warning: file names do not match cleanly, so setup.py will pair files by sorted order.")
    for index, (audio_file, transcript_file) in enumerate(zip(audio_files, transcript_files), start=1):
        item_id = f"{index:04d}_{slugify(audio_file.stem)}"
        pairs.append({"item_id": item_id, "audio": audio_file, "transcript": transcript_file})
    return pairs


def load_progress(audio_dir, transcript_dir):
    if not PROGRESS_PATH.exists():
        return {
            "audio_dir": str(Path(audio_dir).resolve()),
            "transcript_dir": str(Path(transcript_dir).resolve()),
            "completed_items": [],
            "last_completed_item": None,
            "current_item": None,
            "next_item": None,
            "total_items": 0,
        }

    with PROGRESS_PATH.open("r", encoding="utf-8") as file:
        progress = json.load(file)

    current_audio_dir = str(Path(audio_dir).resolve())
    current_transcript_dir = str(Path(transcript_dir).resolve())
    if progress.get("audio_dir") != current_audio_dir or progress.get("transcript_dir") != current_transcript_dir:
        raise RuntimeError("Existing generated progress belongs to different input folders. Use --force to restart.")

    progress.setdefault("completed_items", [])
    progress.setdefault("last_completed_item", None)
    progress.setdefault("current_item", None)
    progress.setdefault("next_item", None)
    progress.setdefault("total_items", 0)
    return progress


def save_progress(progress):
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    with PROGRESS_PATH.open("w", encoding="utf-8") as file:
        json.dump(progress, file, indent=2)


def rebuild_aggregate_dataset(completed_items):
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DATASET_PATH.open("w", newline="", encoding="utf-8") as dataset_file:
        writer = csv.writer(dataset_file, delimiter="|")
        for item_id in completed_items:
            item_dataset = ITEMS_DIR / item_id / "dataset.csv"
            if not item_dataset.exists():
                continue
            with item_dataset.open("r", newline="", encoding="utf-8") as item_file:
                reader = csv.reader(item_file, delimiter="|")
                writer.writerows(reader)


def ensure_step_output(path, step_name):
    if not Path(path).exists():
        raise RuntimeError(f"{step_name} did not produce {path}")


def process_pair(pair, gentle_url):
    item_dir = ITEMS_DIR / pair["item_id"]
    transcript_output = item_dir / "extracted_text.txt"
    audio_output = item_dir / "recording.wav"
    alignment_output = item_dir / "alignment.json"
    item_dataset_output = item_dir / "dataset.csv"

    item_dir.mkdir(parents=True, exist_ok=True)

    prepare_transcript(str(pair["transcript"]), output_path=str(transcript_output))
    ensure_step_output(transcript_output, "Transcript preparation")

    prepare_voice_audio(str(pair["audio"]), output_path=str(audio_output))
    ensure_step_output(audio_output, "Audio preparation")

    align_audio(
        gentle_url=gentle_url,
        transcript_path=str(transcript_output),
        audio_path=str(audio_output),
        output_path=str(alignment_output),
    )
    ensure_step_output(alignment_output, "Audio alignment")

    preprocess_data(
        alignment_path=str(alignment_output),
        audio_path=str(audio_output),
        dataset_path=str(item_dataset_output),
        wavs_dir=str(WAVS_DIR),
        item_prefix=pair["item_id"],
        append=False,
    )
    ensure_step_output(item_dataset_output, "Dataset preprocessing")


def main():
    args = parse_args()

    if args.force:
        clear_generated_outputs()

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    ITEMS_DIR.mkdir(parents=True, exist_ok=True)
    WAVS_DIR.mkdir(parents=True, exist_ok=True)

    pairs = build_pairs(args.voice_path, args.transcript_path)
    pair_ids = {pair["item_id"] for pair in pairs}

    progress = load_progress(args.voice_path, args.transcript_path)
    completed_items = [item_id for item_id in progress["completed_items"] if item_id in pair_ids]
    progress["completed_items"] = completed_items
    progress["total_items"] = len(pairs)
    progress["next_item"] = next((pair["item_id"] for pair in pairs if pair["item_id"] not in completed_items), None)
    save_progress(progress)
    rebuild_aggregate_dataset(completed_items)

    print(f"Found {len(pairs)} recordings and {len(pairs)} transcripts.")
    print(f"Resuming with {len(completed_items)} completed and {len(pairs) - len(completed_items)} remaining.")

    for pair in pairs:
        item_id = pair["item_id"]
        if item_id in completed_items:
            print(f"Skipping completed item: {item_id}")
            continue

        progress["current_item"] = item_id
        progress["next_item"] = item_id
        save_progress(progress)

        print(f"Processing {item_id}")
        process_pair(pair, args.gentle_url)

        completed_items.append(item_id)
        progress["completed_items"] = completed_items
        progress["last_completed_item"] = item_id
        progress["current_item"] = None
        progress["next_item"] = next(
            (next_pair["item_id"] for next_pair in pairs if next_pair["item_id"] not in completed_items),
            None,
        )
        save_progress(progress)
        rebuild_aggregate_dataset(completed_items)

    print("Dataset prep complete.")
    print(f"Audio folder: {args.voice_path}")
    print(f"Transcript folder: {args.transcript_path}")
    print(f"Progress file: {PROGRESS_PATH}")
    print(f"Combined metadata: {DATASET_PATH}")
    print(f"Combined wavs: {WAVS_DIR}")


if __name__ == "__main__":
    main()
