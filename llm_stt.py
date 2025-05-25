# stt_utils.py

from google.cloud import speech
import sounddevice as sd
import soundfile as sf
import os
import numpy as np
import time

# Ensure you have authenticated with Google Cloud:
# gcloud auth application-default login
# And enabled the Cloud Speech-to-Text API in your GCP project.

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

def transcribe_audio_file(audio_file_path, language_code="ko-KR", sample_rate_hertz=16000):
    """
    Transcribes audio from a file using Google Cloud Speech-to-Text.
    
    Args:
        audio_file_path (str): The path to the audio file to transcribe.
        language_code (str): The BCP-47 language tag for the audio (e.g., "en-US", "ko-KR").
        sample_rate_hertz (int): The sample rate of the audio file in Hertz.
                                 Must match the rate used during recording.
    Returns:
        str or None: The transcribed text if successful, None otherwise.
    """
    client = speech.SpeechClient()

    if not os.path.exists(audio_file_path):
        print(f"Error: Audio file not found at {audio_file_path}")
        return None

    with open(audio_file_path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sample_rate_hertz,
        language_code=language_code,
        # enable_automatic_punctuation=True, # Uncomment for automatic punctuation
    )

    print("Transcribing audio...")
    try:
        response = client.recognize(config=config, audio=audio)
        transcript = ""
        for result in response.results:
            transcript += result.alternatives[0].transcript
        print(f"Transcription: {transcript}")
        return transcript
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

# You can add a combined function for convenience
def get_speech_input(duration=7, language_code="ko-KR", temp_audio_file="temp_recorded_audio.wav"):
    """
    Records audio and transcribes it, returning the text.
    
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
        

transcribed_text = get_speech_input(duration=5, language_code="ko-KR")
if transcribed_text:
    print(f"\nFinal Transcribed Text: '{transcribed_text}'")
else:
    print("\nSpeech-to-Text failed.")