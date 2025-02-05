import requests, os

def align_audio(gentle_url="http://localhost:8765/transcriptions?async=false"):
    if not os.path.exists("generated/extracted_text.txt"):
        print("generated/extracted_text.txt not found")
        return
    if os.path.exists("generated/alignment.json"):
        print("generated/alignment.json already exists")
        return
    with open("generated/extracted_text.txt", "r") as f:
        text = f.read()
    files = {"audio": open("generated/recording.wav", "rb")}
    data = {"transcript": text}
    response = requests.post(gentle_url, files=files, data=data)
    try:
        response.json()
    except ValueError:
        print("Error: Server returned an invalid JSON response. Check the Gentle server.")
        print(response.text)
        return
    with open("generated/alignment.json", "w") as f:
        f.write(response.text)
    
    print("Alignment created and saved to generated/alignment.json")

align_audio()