import subprocess
from transformers import pipeline
import os
class Transcript:
    """
    This class is used to extract the audio from the video and then transcribe it to text.
    
    """
    
    def __init__(self):
        self.transcriber = pipeline(
            task="automatic-speech-recognition",
            model="vinai/PhoWhisper-medium",
            device="cuda"
        )
    def extract_audio(self, video_path):
        command = ["ffmpeg", "-i", video_path, "-ab", "16k", "-ac", "2", "-ar", "44100", "-vn", "../data/audio.wav"]
        subprocess.run(command, check=True)
        try:
            output = self.transcriber('../data/audio.wav', return_timestamps=True)['text']
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {str(e)}") from e
        finally:
            self.delete_audio()
        return output

    def delete_audio(self):
        try:
            os.remove('../data/audio.wav')
        except OSError as e:
            print(f"Warning: Failed to delete audio file: {str(e)}")
