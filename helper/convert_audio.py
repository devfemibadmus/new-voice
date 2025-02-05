from pydub import AudioSegment
import os

def convert_mp4_wav(audio_path="data/recording.mp4"):
    if not os.path.exists(audio_path):
        print(f"{audio_path} not found")
        return
    if os.path.exists("generated/recording.wav"):
        print("generated/recording.wav already exists")
        return
    audio = AudioSegment.from_file(audio_path, format="mp4")
    wav_path = "generated/recording.wav"
    audio.export(wav_path, format="wav")
    print("Wav recording created and saved to generated/recording.wav")


convert_mp4_wav()