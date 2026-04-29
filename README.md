# Video Named Entity Recognition (NER) Project

## Project Overview

This project is an NLP pipeline that processes a video file and extracts useful information from it using Hugging Face pre-trained models.

The system performs the following steps:

1. Extract audio from the video
2. Convert speech into text using Speech-to-Text
3. Extract and classify Named Entities from the transcript
4. Save the results into multiple output formats

---

# Technologies Used

- Python
- Hugging Face Transformers
- Whisper Speech-to-Text Model
- Named Entity Recognition (NER)
- MoviePy
- Pandas
- VS Code

---

# Pre-trained Models Used

## Speech-to-Text Model

```python
openai/whisper-small
```

Used to convert audio into text.

## Named Entity Recognition Model

```python
dslim/bert-base-NER
```

Used to extract and classify entities such as:

- PER → Person
- ORG → Organization
- LOC → Location
- MISC → Miscellaneous

---

# Project Structure

```text
NER_Vedio_Project/
│
├── input/
│   └── video.mp4
    └── audio.wav
│
├── output/
│   ├── transcript.txt
│   ├── entities.json
│   ├── entities.csv
│   └── entities_summary.csv
│
├── main.py
└── README.md
└── requirements.txt
```

---

# Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

# Requirements

```txt
transformers==5.6.2
torch==2.11.0
pandas==3.0.2
moviepy==2.2.1
accelerate==1.13.0
imageio-ffmpeg==0.6.0
```

---

# How to Run

1. Place the video file inside the `input` folder.

Example:

```text
input/video.mp4
```

2. Run the project:

```bash
py main.py
```

---

# Output Files

After running the project, the following files will be generated inside the `output` folder:

| File | Description |
|------|-------------|
| transcript.txt | Full transcript extracted from the video |
| entities.json | Extracted entities in JSON format |
| entities.csv | Extracted entities in CSV format |
| entities_summary.csv | Entity frequency summary |

---

# Example Workflow

```text
Video
   ↓
Audio Extraction
   ↓
Speech-to-Text
   ↓
Named Entity Recognition
   ↓
Save Results
```

---

# Features

- Video to text conversion
- Automatic speech recognition
- Named Entity Recognition (NER)
- Entity classification
- Export results to TXT, JSON, and CSV
- Easy to extend and improve

---

# Code Explanation

## Step 1: Extract Audio from Video

The project uses MoviePy to read the video file and extract the audio into a WAV file.

Function used:

```python
def extract_audio_from_video(video_path, audio_path)
```

---

## Step 2: Convert Audio to Text

The project uses the Whisper model from Hugging Face:

```python
openai/whisper-small
```

This model converts speech into text.

Function used:

```python
def transcribe_audio(audio_path)
```

---

## Step 3: Named Entity Recognition (NER)

The project uses the following Hugging Face NER model:

```python
dslim/bert-base-NER
```

The model extracts entities from the transcript such as:

- People
- Organizations
- Locations
- Miscellaneous entities

Function used:

```python
def extract_entities(text)
```

---

## Step 4: Save Results

The extracted data is saved into:

- TXT
- JSON
- CSV

Function used:

```python
def save_results(transcript, entities)
```

---

## Step 5: Main Pipeline

The `main()` function runs the complete NLP workflow:

1. Extract audio
2. Convert speech to text
3. Extract entities
4. Save results

---

# Author

Hadeel

