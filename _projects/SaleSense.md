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
  - title: "Slides used to present"
    url: "/assets/pdf/AISC Project Demo.pdf"
---

**SaleSense** helps sellers to improve second-hand products by scoring the description and generating a new better description. It’s built to be fast, simple, and practical for real resale workflows.

## Problem

Most resale platforms depend heavily on listing quality, but many users don’t know what details to include or how to write descriptions that sell. SaleSense makes that improvement automatic: score → feedback → improved description.

## Dataset imbalance

The dataset was extremely imbalanced (sold vs. not sold), which made the classifier approach unreliable.

<div class="row">
  <div class="col-sm mt-3 mt-md-0" style="max-width: 500px;">
    {% include figure.liquid loading="eager" path="assets/img/salesense_imbalance.png" title="Class imbalance (sold vs. not sold)" class="img-fluid rounded z-depth-1" %}
  </div>
</div>

<div class="caption">
Distribution of “sold” status and the initial baseline attempt.
</div>

## Making the dataset workable

To train and validate models more reliably, we down sampled the negative class to create a more balanced working set.

<div class="row">
  <div class="col-sm mt-3 mt-md-0" style="max-width: 350px;">
    {% include figure.liquid loading="eager" path="assets/img/salesense_downsampling.png" title="Downsampling strategy" class="img-fluid rounded z-depth-1" %}
  </div>
</div>

<div class="caption">
Downsampling strategy used to reduce imbalance and stabilize training/evaluation.
</div>

## LLM feedback loop

For rewriting and feedback, we use an LLM-driven feedback loop to produce clearer and more complete descriptions. A LightGBM classifier scores the inputed description and provides a target signal, guiding the LLM to iteratively rewrite the text to maximize the predicted “sellability” score.

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
