---
layout: page
title: IntelliView
description: An AI interview assistant that reads facial expression + voice, then generates a personalized feedback report.
img: assets/img/intelliview_viz.jpg
importance: 1
category: work
related_publications: false
references:
  - title: "Code (GitHub)"
    url: "https://github.com/PedroTajia/IntelliView"
  - title: "AISC slides"
    url: "/assets/pdf/AISC COMPETITIVE.pdf"
---

## IntelliView — Your AI Interview Assistant

**IntelliView** is a project I built with Lauren Gallego and Alfonso Mayoral
for the **Artificial Intelligence Student Collective (AISC)** as an interview-practice tool.
It analyzes **your facial expressions** (computer vision) and **your voice** (speech-to-text), then uses an **LLM** to generate a structured report with feedback and concrete ways to improve for interviews.

## What it does

- **Face detection (YOLO):** draws a bounding box around the user’s face.
- **Emotion detection (YOLO):** classifies emotion from facial expressions using a second fine-tuned YOLO model.

- **Speech-to-text (Whisper):** transcribes real-time audio into text to use in the report.
- **LLM report generation:** combines the transcript + logged emotions and generates a formatted interview report.
- **Low-latency analysis:** designed to run in real time with minimal latency.

## System pipeline

1. User starts recording (video + audio)
2. YOLO detects face → YOLO predicts emotion (per frame)
3. Whisper transcribes speech to text (timestamped)
4. LLM generates a report from transcript + emotion log

<div class="row justify-content-sm-center">
  <div class="col-sm-10 mt-3 mt-md-0" style="max-width: 500px;">
    {% include figure.liquid path="assets/img/intelliview_viz.png" title="Pipeline: video/audio → emotion + transcript → LLM report" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">
   pipeline diagram 
</div>

## Whisper (speech-to-text)

We used **OpenAI Whisper**, an open-source Transformer trained on **680,000 hours** of multilingual audio that can transcribe and translate speech, and provides fine-grained timestamps.

## LLM report agent

The report generator follows a simple agent workflow: import transcript → clean/timestamp → extract candidate info → draft template → LLM refines → assemble report → export (TXT/MD/HTML).

## Web app flow (user experience)

- **Step 1:** user accounts with hashed passwords and basic password validation.
- **Step 2:** a personal home page to view / modify / delete generated reports. {index=15}
- **Step 3:** press **Start Recording**, run the interview, then **Stop Recording** and wait a few seconds for the report.

## Engineering iteration (what I learned)

Early on, we tried training a **single YOLO model** on both face detection + facial expression datasets, but ran into issues like **catastrophic forgetting** and dataset mismatch. We fixed this by moving to a **two-model YOLO setup** (one for face detection, one for emotion).

<div class="row justify-content-sm-center">
  <div class="col-sm-10 mt-3 mt-md-0" style="max-width: 500px;">
    {% include figure.liquid path="assets/img/first_iteration.png" title="catastrophic forgetting" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">
  This is my first iteration on creating a single yolo model that classify and detect images given two different datasets 
</div>

We also iterated on the LLM strategy to reduce cost and latency, moving toward a fast on-device setup with a small model + keyword search and tight prompting (final phase mentions ~20s end-to-end and laptop-friendly deployment).
