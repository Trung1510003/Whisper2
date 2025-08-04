import whisper

def transcribe(audio_path, lang=None):
    model = whisper.load_model("base")
    print(whisper.__file__)
    print("Đang nhận diện tiếng nói...")
    result = model.transcribe(audio_path, language=lang)
    print(f"Văn bản: {result['text']}")
    return result['text']

if __name__ == "__main__":
    transcribe("output.wav", lang="vi") 
