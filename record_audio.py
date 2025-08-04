import sounddevice as sd
from scipy.io.wavfile import write

def record_audio(filename='output.wav', duration=5, fs=16000):
    print("Bắt đầu ghi âm...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, audio)
    print(f"Đã ghi âm xong và lưu tại: {filename}")

if __name__ == "__main__":
    record_audio()
