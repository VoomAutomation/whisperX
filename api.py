import requests
import json
import time
#tiny 1.408 seconds
#small  4.129 seconds
#base 1.676 seconds   8.566 seconds  Elapsed time: 1.752 seconds
def transcribe_and_time(file_path, url="http://127.0.0.1:8080/transcribe"):
    """Send audio file to API and return response and elapsed time in seconds."""
    with open(file_path, "rb") as f:
        files = {"file": open(r"data/ar.wav", "rb")}
        data = {"dialect": "ar", "language": "ar"}
        start_time = time.time()  # Start timer
        response = requests.post(url, files=files, data=data)
        elapsed_time = time.time() - start_time  # End timer

    # Save response JSON to file
    with open("arabic_test.json", "w", encoding="utf-8") as f_out:
        json.dump(response.json(), f_out, ensure_ascii=False, indent=2)

    return response.json(), elapsed_time


# Usage
result, duration = transcribe_and_time(r"data/ar.wav")
print(f"Text: {result.get('text', 'N/A')}")
print(f"Phonemes: {len(result.get('phonemes', []))}")
print(f"Detected: {result.get('detected_language')}, Used: {result.get('used_language')}")

