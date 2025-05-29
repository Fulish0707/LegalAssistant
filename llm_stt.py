import requests
import json
import time
import sounddevice as sd
import soundfile as sf
import os
import hmac
import hashlib
import base64
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get credentials from .env
NAVER_CLIENT_ID = os.getenv('client_id')
NAVER_CLIENT_SECRET = os.getenv('client_secret')

def record_audio_to_file(filename="recorded_audio.wav", duration=5, samplerate=16000):
    """
    Records audio from the microphone and saves it to a WAV file.
    
    Args:
        filename (str): The path to save the recorded WAV file.
        duration (int): The duration of the recording in seconds.
        samplerate (int): The sample rate of the audio (Hz).
                          Commonly 16000 for speech.
    Returns:
        str or None: The path to the saved audio file if successful, None otherwise.
    """
    print(f"Recording for {duration} seconds... Speak now!")
    try:
        # Ensure the samplerate is supported by your microphone
        # You might need to check sd.query_devices() for available rates
        recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
        sd.wait()  # Wait until recording is finished
        sf.write(filename, recording, samplerate)
        print(f"Recording saved to {filename}")
        return filename
    except Exception as e:
        print(f"Error during audio recording: {e}")
        print("Please ensure your microphone is connected and working.")
        print("You might need to install PortAudio for sounddevice:")
        print("  macOS: `brew install portaudio`")
        print("  Debian/Ubuntu: `sudo apt-get install libportaudio2`")
        return None

def transcribe_audio_file(audio_file_path, language="Kor"):
    """
    Transcribes audio using Naver Clova Speech Recognition.
    
    Args:
        audio_file_path (str): Path to the audio file
        language (str): Language code (default: ko-KR)
    """
    url = f"https://naveropenapi.apigw.ntruss.com/recog/v1/stt?lang={language}"
    
    with open(audio_file_path, 'rb') as data:
        headers = {
            "X-NCP-APIGW-API-KEY-ID": NAVER_CLIENT_ID,
            "X-NCP-APIGW-API-KEY": NAVER_CLIENT_SECRET,
            "Content-Type": "application/octet-stream"
        }
        
        response = requests.post(url, data=data, headers=headers)
        rescode = response.status_code
        
        if rescode == 200:
            result = response.json()
            return result.get('text', '')
        else:
            print("Error: " + response.text)
            return None

# You can add a combined function for convenience
def get_speech_input(duration=7, language="Kor", temp_audio_file="temp_recorded_audio.wav"):
    """
    Records audio and transcribes it using Naver Clova Speech Recognition.
    
    Args:
        duration (int): Duration of recording in seconds
        language (str): Language code
        temp_audio_file (str): Temporary file to save audio
    """
    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        print("Error: Naver Cloud Platform credentials not found in .env file")
        return None
        
    audio_path = record_audio_to_file(temp_audio_file, duration=duration)
    if audio_path:
        transcript = transcribe_audio_file(audio_path, language)
        # Clean up the temporary audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)
        return transcript
    return None

def get_speech_input_from_microphone(duration=7, language_code="ko-KR", temp_audio_file="temp_recorded_audio.wav"):
    """
    Records audio from the microphone and transcribes it, returning the text.
    
    Args:
        duration (int): Duration of recording in seconds.
        language_code (str): Language for transcription.
        temp_audio_file (str): Temporary file to save audio.
    Returns:
        str or None: Transcribed text or None if failed.
    """
    audio_path = record_audio_to_file(temp_audio_file, duration=duration)
    if audio_path:
        transcript = transcribe_audio_file(audio_path, language_code=language_code)
        if os.path.exists(audio_path):
            os.remove(audio_path)
        return transcript
    return None
        

if __name__ == "__main__":
    # This block will only execute when the script is run directly,
    # not when it's imported as a module.
    print("Running llm_stt.py as a standalone script for testing...")
    
    transcribed_text = get_speech_input(duration=5, language="Kor")
    
    if transcribed_text:
        print(f"\nFinal Transcribed Text: '{transcribed_text}'")
    else:
        print("\nSpeech-to-Text failed.")
