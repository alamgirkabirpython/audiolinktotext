import streamlit as st
from transformers import pipeline
from pydub import AudioSegment
import torch
import os
import time
import yt_dlp
import tempfile

# Set Streamlit page config
st.set_page_config(page_title="Audio/Video-to-Text Transcription", layout="centered", initial_sidebar_state="auto")

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the Whisper model pipeline
@st.cache_resource
def load_model():
    return pipeline(
        "automatic-speech-recognition",
        "openai/whisper-large-v2",
        chunk_length_s=30,
        stride_length_s=3,
        return_timestamps=True,
        device=device,
    )

pipe = load_model()

# Function to download audio using yt-dlp
def download_audio_youtube(video_url, output_path="temp_audio.mp4"):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'quiet': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        return output_path
    except yt_dlp.utils.DownloadError as e:
        st.error(f"Error downloading audio from YouTube: {e}")
        return None

# Function to format transcription with timestamps and combine text
def format_transcription(transcription):
    formatted_text = ""
    full_text = ""
    previous_text = ""

    for chunk in transcription.get('chunks', []):
        text = chunk["text"]
        timestamps = chunk.get("timestamp", None)

        # Handle missing timestamps
        if timestamps:
            start_time, end_time = timestamps
            formatted_text += f"[{start_time:.2f} - {end_time:.2f}] {text.strip()}\n"
        else:
            formatted_text += f"[No Timestamp] {text.strip()}\n"

        # Avoid duplicate consecutive text
        if text.strip() != previous_text:
            full_text += text.strip() + " "
            previous_text = text.strip()

    return formatted_text.strip(), full_text.strip()

# Function to remove repeated words
def remove_repeated_words(text):
    words = text.split()
    cleaned_words = []
    for i, word in enumerate(words):
        if i == 0 or word != words[i - 1]:
            cleaned_words.append(word)
    return ' '.join(cleaned_words)

# Function to convert audio to WAV format using pydub
def convert_to_wav(input_file, output_file="temp_audio.wav"):
    audio = AudioSegment.from_file(input_file)
    audio.export(output_file, format="wav")
    return output_file

# Main app function
def main():
    st.markdown("<h1 style='color: #00bfff;'>Audio/Video-to-Text Transcription App</h1>", unsafe_allow_html=True)

    # Tabs for different functionalities
    tab1, tab2 = st.tabs(["Audio File", "YouTube Video"])

    # Tab for audio file upload
    with tab1:
        st.subheader("Transcribe Audio File")
        uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "ogg"])
        st.audio(uploaded_file)

        if uploaded_file:
            if st.button("Transcribe Audio"):
                with st.spinner("Processing..."):
                    start_time = time.time()

                    # Use tempfile for temporary file storage
                    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                        temp_file.write(uploaded_file.getbuffer())
                        temp_file_path = temp_file.name

                    # Convert to WAV using pydub
                    wav_file = convert_to_wav(temp_file_path)

                    # Perform transcription
                    transcription = pipe(wav_file)
                    st.write(transcription)  # Debugging: Check the transcription output
                    formatted_transcription, full_transcription = format_transcription(transcription)

                    st.success("Transcription completed!")
                    st.subheader("Formatted Transcription with Timestamps")
                    st.text_area("Formatted Output", value=formatted_transcription, height=400)

                    st.subheader("Full Combined Transcription")
                    st.text_area("Combined Text Output", value=full_transcription, height=400)

                    # Download options
                    st.download_button("Download Formatted Transcription", formatted_transcription, file_name="formatted_transcription.txt")
                    st.download_button("Download Full Transcription", full_transcription, file_name="full_transcription.txt")

                    end_time = time.time()
                    st.write(f"Time taken: {round(end_time - start_time, 2)} seconds")

                    # Clean up
                    os.remove(temp_file_path)
                    os.remove(wav_file)

    # Tab for YouTube video transcription
    with tab2:
        st.subheader("Transcribe YouTube Video")
        video_url = st.text_input("Enter YouTube video link:")

        if video_url:
            if st.button("Transcribe Video"):
                with st.spinner("Processing..."):
                    try:
                        # Download audio
                        st.info("Downloading audio...")
                        audio_file = download_audio_youtube(video_url)

                        if audio_file:
                            # Convert to WAV using pydub
                            st.info("Extracting audio...")
                            wav_file = convert_to_wav(audio_file)

                            # Perform transcription
                            st.info("Transcribing audio...")
                            transcription = pipe(wav_file)
                            st.write(transcription)  # Debugging: Check the transcription output
                            formatted_transcription, full_transcription = format_transcription(transcription)

                            st.success("Transcription completed!")
                            st.subheader("Formatted Transcription with Timestamps")
                            st.text_area("Formatted Output", value=formatted_transcription, height=400)

                            st.subheader("Full Combined Transcription")
                            st.text_area("Combined Text Output", value=full_transcription, height=400)

                            # Download options
                            st.download_button("Download Formatted Transcription", formatted_transcription, file_name="formatted_transcription.txt")
                            st.download_button("Download Full Transcription", full_transcription, file_name="full_transcription.txt")

                            # Clean up
                            os.remove(audio_file)
                            os.remove(wav_file)

                    except Exception as e:
                        st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
