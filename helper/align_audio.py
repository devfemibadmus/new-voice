from pathlib import Path

import requests


def align_audio(
    gentle_url="http://localhost:8765/transcriptions?async=false",
    transcript_path="generated/extracted_text.txt",
    audio_path="generated/recording.wav",
    output_path="generated/alignment.json",
):
    transcript_file = Path(transcript_path)
    audio_file = Path(audio_path)
    output_file = Path(output_path)

    if not transcript_file.exists():
        print(f"{transcript_file} not found")
        return
    if not audio_file.exists():
        print(f"{audio_file} not found")
        return
    if output_file.exists():
        print(f"{output_file} already exists")
        return

    transcript = transcript_file.read_text(encoding="utf-8")
    with audio_file.open("rb") as audio_stream:
        response = requests.post(
            gentle_url,
            files={"audio": audio_stream},
            data={"transcript": transcript},
            timeout=300,
        )

    try:
        response.raise_for_status()
        response.json()
    except ValueError:
        print("Error: Server returned an invalid JSON response. Check the Gentle server.")
        print(response.text)
        return
    except requests.RequestException as exc:
        print(f"Error talking to Gentle: {exc}")
        return

    output_file.write_text(response.text, encoding="utf-8")
    print(f"Alignment created and saved to {output_file}")
