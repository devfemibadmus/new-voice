import argparse
import shutil
from pathlib import Path

from helper import align_audio, prepare_transcript, prepare_voice_audio, preprocess_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a voice dataset from a WAV recording and matching transcript."
    )
    parser.add_argument(
        "--voice-path",
        default="data/voice.wav",
        help="Path to the voice recording you want to fine-tune on.",
    )
    parser.add_argument(
        "--transcript-path",
        default="data/transcript.txt",
        help="Path to the transcript that matches the recording.",
    )
    parser.add_argument(
        "--gentle-url",
        default="http://localhost:8765/transcriptions?async=false",
        help="Gentle alignment endpoint.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing generated dataset artifacts before rebuilding them.",
    )
    return parser.parse_args()


def clear_generated_outputs():
    paths_to_remove = [
        Path("generated/recording.wav"),
        Path("generated/extracted_text.txt"),
        Path("generated/alignment.json"),
        Path("generated/dataset.csv"),
        Path("generated/wavs"),
    ]

    for path in paths_to_remove:
        if not path.exists():
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def main():
    args = parse_args()

    if args.force:
        clear_generated_outputs()

    prepare_transcript(args.transcript_path)
    prepare_voice_audio(args.voice_path)
    align_audio(gentle_url=args.gentle_url)
    preprocess_data()

    print("Dataset prep complete.")
    print("Input voice file:", args.voice_path)
    print("Input transcript file:", args.transcript_path)
    print("Prepared audio:", "generated/recording.wav")
    print("Prepared transcript:", "generated/extracted_text.txt")
    print("Metadata:", "generated/dataset.csv")


if __name__ == "__main__":
    main()
