import whisper
import sounddevice as sd
from scipy.io.wavfile import write

def record_audio(filename='output.wav', duration=5, fs=16000):
    print("🎤 Bắt đầu ghi âm...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, audio)
    print(f"✅ Đã ghi âm xong: {filename}")

def detect_language(audio_path, model):
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)
    lang = max(probs, key=probs.get)
    print(f"Ngôn ngữ: {lang} ({probs[lang]*100:.2f}%)")
    return lang

def transcribe(audio_path, model, lang):
    print("Nhận diện tiếng nói...")
    result = model.transcribe(audio_path, language=lang)
    print(f"Văn bản: {result['text']}")
    return result['text']

if __name__ == "__main__":
    model = whisper.load_model("base")
    audio_path = "output.wav"
    
    record_audio(audio_path)
    lang = detect_language(audio_path, model)
    transcribe(audio_path, model, lang)
