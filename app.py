import streamlit as st
from transformers import pipeline
from moviepy.editor import AudioFileClip
import torch
import os
import yt_dlp

# Set Streamlit page configuration
st.set_page_config(page_title="Video-to-Text Transcription", layout="centered", initial_sidebar_state="auto")

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# User input for chunk length and stride length
st.sidebar.header("Model Configuration")
chunk_length_s = st.sidebar.number_input("Chunk Length (seconds):", min_value=5, max_value=120, value=10, step=1)
stride_length_s = st.sidebar.number_input("Stride Length (seconds):", min_value=1, max_value=10, value=2, step=1)

# Load the Whisper model pipeline
@st.cache_resource
def load_model(chunk_length, stride_length):
    return pipeline(
        "automatic-speech-recognition",
        "openai/whisper-small",
        chunk_length_s=chunk_length,
        stride_length_s=stride_length,
        return_timestamps=True,
        device=device
    )

pipe = load_model(chunk_length_s, stride_length_s)

# Function to download audio using yt-dlp
def download_audio_youtube(video_url, output_path="temp_audio.mp4"):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])
    return output_path

# App title
st.title("YouTube Video to Text Transcription")

# Input for YouTube video link
video_url = st.text_input("Enter YouTube video link:")

if video_url:
    try:
        # Download audio
        st.info("Downloading audio...")
        audio_file = download_audio_youtube(video_url)

        # Extract audio as WAV
        st.info("Extracting audio...")
        audio_clip = AudioFileClip(audio_file)
        wav_file = "temp_audio.wav"
        audio_clip.write_audiofile(wav_file, codec='pcm_s16le')
        audio_clip.close()

        # Perform transcription
        st.info("Transcribing audio...")
        transcription = pipe(wav_file)

        # Display transcription
        st.subheader("Transcription Result:")
        st.text_area("Transcribed Text:", transcription['text'], height=400)

        # Cleanup temporary files
        os.remove(audio_file)
        os.remove(wav_file)
    except Exception as e:
        st.error(f"Error: {e}")
