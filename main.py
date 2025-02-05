from TTS.api import TTS

files = sorted(os.listdir("output"))
tts = TTS(model_path=files[-1], config_path="coqui_tts_config.json")

text = "Hello, this is my custom voice!"
output_path = "output.wav"
tts.tts_to_file(text=text, file_path=output_path)

print(f"Speech saved to {output_path}")

