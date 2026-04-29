import os
import json
import pandas as pd
import streamlit as st
from moviepy import VideoFileClip
from transformers import pipeline

# =========================
# FFmpeg Path
# =========================

os.environ["PATH"] += os.pathsep + r"C:\Users\MCC\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin"

# =========================
# Streamlit UI
# =========================

st.title("Video Named Entity Recognition")

st.write("Upload a video and extract entities using Hugging Face models.")

uploaded_file = st.file_uploader(
    "Upload Video",
    type=["mp4", "mov", "avi"]
)

# =========================
# Functions
# =========================

def extract_audio(video_path, audio_path):
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path)
    clip.close()

def transcribe_audio(audio_path):
    asr = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small"
    )

    result = asr(
        audio_path,
        return_timestamps=True
    )

    return result["text"]

def extract_entities(text):
    ner = pipeline(
        "ner",
        model="dslim/bert-base-NER",
        aggregation_strategy="simple"
    )

    results = ner(text)

    entities = []

    bad_entities = ["Sa", "ki", "CO", "Global"]

    for item in results:
        entity_text = item.get("word", "").replace("##", "").strip()
        entity_label = item.get("entity_group", "")
        score = round(float(item.get("score", 0)), 4)

        if len(entity_text) < 2:
            continue

        if score < 0.70:
            continue

        if entity_text in bad_entities:
            continue

        entities.append({
            "Entity": entity_text,
            "Label": entity_label,
            "Score": score
        })

    return pd.DataFrame(entities)

# =========================
# Main Process
# =========================

if uploaded_file is not None:

    os.makedirs("temp", exist_ok=True)

    video_path = os.path.join("temp", uploaded_file.name)
    audio_path = "temp/audio.wav"

    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())

    st.video(video_path)

    if st.button("Run NLP Pipeline"):

        with st.spinner("Extracting audio..."):
            extract_audio(video_path, audio_path)

        with st.spinner("Converting speech to text..."):
            transcript = transcribe_audio(audio_path)

        st.subheader("Transcript")
        st.write(transcript)

        with st.spinner("Extracting entities..."):
            entities_df = extract_entities(transcript)

        st.subheader("Entities")
        st.dataframe(entities_df)

        csv = entities_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Entities CSV",
            csv,
            "entities.csv",
            "text/csv"
        )