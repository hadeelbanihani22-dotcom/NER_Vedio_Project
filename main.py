import os
import json
import pandas as pd
from moviepy import VideoFileClip
from transformers import pipeline


# =========================
# FFmpeg Path
# =========================

os.environ["PATH"] += os.pathsep + r"C:\Users\MCC\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin"

# =========================
# Settings
# =========================

VIDEO_PATH = "input/video.mp4"
AUDIO_PATH = "input/audio.wav"

OUTPUT_DIR = "output"

TRANSCRIPT_PATH = f"{OUTPUT_DIR}/transcript.txt"
ENTITIES_JSON_PATH = f"{OUTPUT_DIR}/entities.json"
ENTITIES_CSV_PATH = f"{OUTPUT_DIR}/entities.csv"
ENTITIES_SUMMARY_PATH = f"{OUTPUT_DIR}/entities_summary.csv"

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

# =========================
# 1) Extract audio from video
# =========================

def extract_audio_from_video(video_path, audio_path):
    clip = VideoFileClip(video_path)

    if clip.audio is None:
        raise ValueError("No audio found in the video.")

    clip.audio.write_audiofile(audio_path)
    clip.close()

    return audio_path

# =========================
# 2) Speech to Text
# =========================

def transcribe_audio(audio_path):
    asr = pipeline(
        task="automatic-speech-recognition",
        model="openai/whisper-small"
    )

    result = asr(
        audio_path,
        return_timestamps=True
    )

    return result["text"]

# =========================
# 3) Extract Entities
# =========================

def extract_entities(text):
    ner = pipeline(
        task="ner",
        model="dslim/bert-base-NER",
        aggregation_strategy="simple"
    )

    results = ner(text)

    entities = []
    bad_entities = ["Sa", "ki", "CO", "Global", "r Starmer"]
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
            "entity_text": entity_text,
            "entity_label": entity_label,
            "score": score,
            "start": item.get("start", None),
            "end": item.get("end", None)
        })

    return entities
# =========================
# 4) Save Results
# =========================

def save_results(transcript, entities):
    with open(TRANSCRIPT_PATH, "w", encoding="utf-8") as f:
        f.write(transcript)

    with open(ENTITIES_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(entities, f, ensure_ascii=False, indent=2)

    df = pd.DataFrame(entities)
    df.to_csv(ENTITIES_CSV_PATH, index=False, encoding="utf-8-sig")

    if not df.empty:
        summary = (
            df.groupby(["entity_label", "entity_text"])
            .size()
            .reset_index(name="count")
            .sort_values(by="count", ascending=False)
        )
        summary.to_csv(ENTITIES_SUMMARY_PATH, index=False, encoding="utf-8-sig")

# =========================
# 5) Main
# =========================

def main():
    print("Extracting audio from video...")
    audio_path = extract_audio_from_video(VIDEO_PATH, AUDIO_PATH)

    print("Converting speech to text...")
    transcript = transcribe_audio(audio_path)

    print("\nTranscript:")
    print(transcript)

    print("\nExtracting entities...")
    entities = extract_entities(transcript)

    print("\nEntities:")
    for entity in entities:
        print(entity)

    print("\nSaving results...")
    save_results(transcript, entities)

    print("\nDone!")
    print("Files saved in output folder.")

if __name__ == "__main__":
    main()