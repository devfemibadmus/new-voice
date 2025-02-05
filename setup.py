from helper import extract_text_from_pdf, convert_mp4_wav, align_audio, preprocess_data

extract_text_from_pdf()
convert_mp4_wav()
align_audio()
preprocess_data()

with open("generated/dataset.csv", "r", encoding="utf-8") as file:
    data = file.read()

# Replace 'â' with "'ll"
data = data.replace("â", "'ll")

with open("dataset_fixed.csv", "w", encoding="utf-8") as file:
    file.write(data)


