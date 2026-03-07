from pathlib import Path

from pydub import AudioSegment


DEFAULT_AUDIO_CANDIDATES = ("data/voice.wav", "data/recording.mp4")


def resolve_audio_source(audio_path=None):
    if audio_path:
        return Path(audio_path)

    for candidate in DEFAULT_AUDIO_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            return path

    return Path(DEFAULT_AUDIO_CANDIDATES[0])


def prepare_voice_audio(audio_path=None, output_path="generated/recording.wav"):
    source_path = resolve_audio_source(audio_path)
    output_file = Path(output_path)

    if not source_path.exists():
        print(f"{source_path} not found")
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists():
        print(f"{output_file} already exists")
        return

    audio = AudioSegment.from_file(source_path)
    audio.export(output_file, format="wav")
    print(f"Voice audio prepared from {source_path} and saved to {output_file}")


def convert_mp4_wav(audio_path=None, output_path="generated/recording.wav"):
    prepare_voice_audio(audio_path=audio_path, output_path=output_path)
