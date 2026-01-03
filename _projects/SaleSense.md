---
layout: page
title: SaleSense
description: A tool that scores resale listings and generates clear, higher-converting product descriptions.
img: assets/img/salesense_ui.png
importance: 1
category: work
related_publications: false
references:
  - title: "GitHub: AISC_SaleSense"
    url: "https://github.com/PedroTajia/AISC_SaleSense"
  - title: "Streamlit"
    url: "https://streamlit.io/"
  - title: "Tailwind CSS"
    url: "https://tailwindcss.com/"
---

**SaleSense** helps everyday sellers improve second-hand clothing listings by scoring the description and generating actionable rewrite suggestions. It’s built to be fast, simple, and practical for real resale workflows.

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/salesense_ui.png" title="SaleSense UI" class="img-fluid rounded z-depth-1" %}
  </div>
</div>

<div class="caption">
SaleSense interface (replace this with your best screenshot/GIF of the input → output flow).
</div>

## Problem

Most resale platforms depend heavily on listing quality, but many users don’t know what details to include or how to write descriptions that sell. SaleSense makes that improvement automatic: score → feedback → improved description.

## Dataset imbalance

The dataset was extremely imbalanced (sold vs. not sold), which made a naive binary classifier approach unreliable.

<div class="row">
  <div class="col-sm mt-3 mt-md-0" style="max-width: 500px;">
    {% include figure.liquid loading="eager" path="assets/img/salesense_imbalance.png" title="Class imbalance (sold vs. not sold)" class="img-fluid rounded z-depth-1" %}
  </div>
</div>

<div class="caption">
Distribution of “sold” status and the initial baseline attempt.
</div>

## Making the dataset workable

To train and validate models more reliably, we downsampled the negative class to create a more balanced working set.

<div class="row">
  <div class="col-sm mt-3 mt-md-0" style="max-width: 350px;">
    {% include figure.liquid loading="eager" path="assets/img/salesense_downsampling.png" title="Downsampling strategy" class="img-fluid rounded z-depth-1" %}
  </div>
</div>

<div class="caption">
Downsampling strategy used to reduce imbalance and stabilize training/evaluation.
</div>

## LLM feedback loop

For rewriting and feedback, we used an LLM-driven feedback loop to generate clearer, more complete descriptions and concrete suggestions.

<div class="row">
  <div class="col-sm mt-3 mt-md-0" style="max-width: 700px;">
    {% include figure.liquid loading="eager" path="assets/img/salesense_llm_feedback_loop.png" title="LLM feedback loop" class="img-fluid rounded z-depth-1" %}
  </div>
</div>

<div class="caption">
LLM feedback loop used to produce improvements and guidance.
</div>

## Design + build

The interface was designed to be minimal and fast: paste a description, generate feedback, and get an improved version immediately. The project is implemented as a lightweight web app (Streamlit) so it’s easy to run and demo.

## Repository

Code: **https://github.com/PedroTajia/AISC_SaleSense**
