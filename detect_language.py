import whisper

def detect_language(audio_path):
    model = whisper.load_model("base")
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)
    lang = max(probs, key=probs.get)
    print(f"Ngôn ngữ phát hiện: {lang} ({probs[lang]*100:.2f}%)")
    return lang

if __name__ == "__main__":
    detect_language("output.wav")
