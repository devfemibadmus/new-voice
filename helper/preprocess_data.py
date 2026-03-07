import csv
import json
import os
from pathlib import Path

from pydub import AudioSegment


def preprocess_data():
    if not os.path.exists("generated/alignment.json"):
        print("generated/alignment.json not found")
        return
    if os.path.exists("generated/dataset.csv"):
        print("generated/dataset.csv already exists")
        return

    Path("generated/wavs").mkdir(parents=True, exist_ok=True)

    with open("generated/alignment.json", "r", encoding="utf-8") as file:
        alignment = json.load(file)

    audio = AudioSegment.from_wav("generated/recording.wav")
    dataset = []

    for word in alignment["words"]:
        if "start" in word and "end" in word:
            start_time = int(word["start"] * 1000)
            end_time = int(word["end"] * 1000)
            text_segment = word["word"]
            audio_segment = audio[start_time:end_time]
            segment_path = f"generated/wavs/segment_{start_time}_{end_time}.wav"
            audio_segment.export(segment_path, format="wav")
            dataset.append((f"segment_{start_time}_{end_time}", "speaker1", text_segment))

    with open("generated/dataset.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter="|")
        writer.writerows(dataset)

    print("Dataset created and saved to generated/dataset.csv")
