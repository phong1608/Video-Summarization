import subprocess
import os
import librosa
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch

class Transcript:
    
    def __init__(self, model_name: str = "vinai/PhoWhisper-base", audio_file: str = "data/audio.wav") -> None:
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.audio_file = audio_file

    def transcribe(self, audio_path: str) -> str:
        
        # Load file âm thanh với tần số mẫu 16kHz
        speech_array, sampling_rate = librosa.load(audio_path, sr=16000)
        speech_array = torch.tensor(speech_array)

        # Tokenize đầu vào
        inputs = self.processor(speech_array, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Sinh output mà không cần tính gradient
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs)

        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return transcription

    def extract_audio(self, video_path: str) -> str:
        
        command = [
            "ffmpeg",
            "-i", video_path,
            "-ab", "16k",
            "-ac", "2",
            "-ar", "44100",
            "-vn",
            self.audio_file
        ]
        subprocess.run(command, check=True)
        try:
            transcription = self.transcribe(self.audio_file)
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}") from e
        finally:
            self.delete_audio()
        return transcription

    def delete_audio(self) -> None:
        
        try:
            os.remove(self.audio_file)
        except OSError as e:
            print(f"Warning: Failed to delete audio file: {e}")
