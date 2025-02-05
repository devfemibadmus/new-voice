import os
import csv
import json
from pydub import AudioSegment

def preprocess_data():
    if not os.path.exists("generated/alignment.json"):
        print("generated/alignment.json not found")
        return
    if os.path.exists("generated/dataset.csv"):
        print("generated/dataset.csv already exists")
        return
    
    with open("generated/alignment.json", "r", encoding="utf-8") as f:
        alignment = json.load(f)
        
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
            dataset.append((f"segment_{start_time}_{end_time}", "speaker1", text_segment))  # Added speaker_id column
            
    with open("generated/dataset.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="|")
        # writer.writerow(["audio_path", "speaker_id", "text"])  # Updated header
        writer.writerows(dataset)
        
    print("Dataset created and saved to generated/dataset.csv")


preprocess_data()
