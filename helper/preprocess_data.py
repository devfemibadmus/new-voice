import csv
import json
from pathlib import Path

from pydub import AudioSegment


def preprocess_data(
    alignment_path="generated/alignment.json",
    audio_path="generated/recording.wav",
    dataset_path="generated/dataset.csv",
    wavs_dir="generated/wavs",
    item_prefix=None,
    append=False,
):
    alignment_file = Path(alignment_path)
    audio_file = Path(audio_path)
    dataset_file = Path(dataset_path)
    wavs_path = Path(wavs_dir)

    if not alignment_file.exists():
        raise FileNotFoundError(f"{alignment_file} not found")
    if not audio_file.exists():
        raise FileNotFoundError(f"{audio_file} not found")

    wavs_path.mkdir(parents=True, exist_ok=True)
    dataset_file.parent.mkdir(parents=True, exist_ok=True)

    prefix = f"{item_prefix}_" if item_prefix else ""
    if item_prefix:
        for existing_file in wavs_path.glob(f"{prefix}segment_*.wav"):
            existing_file.unlink()

    with alignment_file.open("r", encoding="utf-8") as file:
        alignment = json.load(file)

    audio = AudioSegment.from_wav(audio_file)
    dataset_rows = []

    for word in alignment.get("words", []):
        if "start" not in word or "end" not in word:
            continue

        start_time = int(word["start"] * 1000)
        end_time = int(word["end"] * 1000)
        text_segment = str(word.get("word", "")).strip()
        if not text_segment:
            continue

        clip_id = f"{prefix}segment_{start_time}_{end_time}"
        segment_path = wavs_path / f"{clip_id}.wav"
        audio_segment = audio[start_time:end_time]
        audio_segment.export(segment_path, format="wav")
        dataset_rows.append((clip_id, "speaker1", text_segment))

    write_mode = "a" if append else "w"
    with dataset_file.open(write_mode, newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter="|")
        writer.writerows(dataset_rows)

    print(f"Prepared {len(dataset_rows)} segments into {wavs_path} and wrote metadata to {dataset_file}")
    return len(dataset_rows)
