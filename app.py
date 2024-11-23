import streamlit as st
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pydub import AudioSegment
import torch
import os
import time
import yt_dlp
import tempfile

# Set Streamlit page config
st.set_page_config(page_title="Audio/Video-to-Text Transcription", layout="centered", initial_sidebar_state="auto")

# Check if GPU is available
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load the Whisper model and pipeline
@st.cache_resource
def load_model():
    model_id = "openai/whisper-large-v3"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)

    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        batch_size=16,  # Adjust based on device
        torch_dtype=torch_dtype,
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

# Function to format transcription
def format_transcription(transcription):
    formatted_text = transcription["text"]
    return formatted_text.strip()

# Function to convert audio to WAV format using pydub
def convert_to_wav(input_file, output_file="temp_audio.wav"):
    audio = AudioSegment.from_file(input_file)
    audio.export(output_file, format="wav")
    return output_file

# Main app function
def main():
    st.markdown("<h1 style='color: #00bfff;'>Audio/Video-to-Text Transcription App</h1>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Audio File", "YouTube Video"])

    with tab1:
        st.subheader("Transcribe Audio File")
        uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "ogg"])
        if uploaded_file:
            st.audio(uploaded_file)

            if st.button("Transcribe Audio"):
                with st.spinner("Processing..."):
                    start_time = time.time()

                    # Temporary file handling
                    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                        temp_file.write(uploaded_file.getbuffer())
                        temp_file_path = temp_file.name

                    try:
                        # Convert to WAV and transcribe
                        wav_file = convert_to_wav(temp_file_path)
                        transcription = pipe(wav_file)
                        formatted_transcription = format_transcription(transcription)

                        # Display results
                        st.success("Transcription completed!")
                        st.subheader("Transcription")
                        st.text_area("Output", value=formatted_transcription, height=400)

                        # Download options
                        st.download_button("Download Transcription", formatted_transcription, file_name="transcription.txt")

                    except Exception as e:
                        st.error(f"Error processing audio: {e}")

                    finally:
                        end_time = time.time()
                        st.write(f"Time taken: {round(end_time - start_time, 2)} seconds")
                        os.remove(temp_file_path)

    with tab2:
        st.subheader("Transcribe YouTube Video")
        video_url = st.text_input("Enter YouTube video link:")
        if video_url:
            if st.button("Transcribe Video"):
                with st.spinner("Processing..."):
                    try:
                        audio_file = download_audio_youtube(video_url)

                        if audio_file:
                            wav_file = convert_to_wav(audio_file)
                            transcription = pipe(wav_file)
                            formatted_transcription = format_transcription(transcription)

                            # Display results
                            st.success("Transcription completed!")
                            st.subheader("Transcription")
                            st.text_area("Output", value=formatted_transcription, height=400)

                            # Download options
                            st.download_button("Download Transcription", formatted_transcription, file_name="transcription.txt")

                            os.remove(audio_file)
                            os.remove(wav_file)

                    except Exception as e:
                        st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
