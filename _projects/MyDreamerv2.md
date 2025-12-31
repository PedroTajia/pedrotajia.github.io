---
layout: page
title: MyDreamerv2
description: A reimplementation and extension of DreamerV2 with exploration via Plan2Explore and selected improvements from DreamerV3
img: assets/img/sketch_dv2.png
importance: 1
category: work
related_publications: true
---

**MyDreamerV2** is a PyTorch reimplementation of _DreamerV2_ that focuses on understanding and extending world-model–based reinforcement learning. The project includes a faithful reproduction of the original DreamerV2 pipeline, an explicit implementation of **Plan2Explore** for intrinsic motivation, and selected architectural and training improvements inspired by **DreamerV3**.

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/rssm_viz.png" title="RSSM latent visualization" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/viz.png" title="Environment visualization" class="img-fluid rounded z-depth-1" %}
  </div>
</div>

<div class="caption">
Left: a visualization of latent states learned by the Recurrent State-Space Model (RSSM), illustrating how the world model compresses observations into a compact predictive representation.  
Right: the corresponding environment used for training, from which the observations are collected.
</div>

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/dv3_return.png" title="Return curve" class="img-fluid rounded z-depth-1" %}
  </div>
</div>

<div class="caption">
Episode return over training, showing a stable asymptotic improvement as the agent learns and plans within the learned world model.
</div>

Beyond reproducing the baseline DreamerV2 results, this implementation emphasizes **exploration through uncertainty**. Plan2Explore is implemented on top of the learned latent dynamics, encouraging the agent to seek trajectories where the world model is uncertain rather than relying solely on extrinsic reward.

<div class="row justify-content-sm-center">
  <div class="col-sm-4 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/dv3.png" title="DreamerV2 architecture" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm-6 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/plan2explore.png" title="Plan2Explore overview" class="img-fluid rounded z-depth-1" %}
  </div>
</div>

<div class="caption">
Left: the DreamerV2 architecture, consisting of a learned world model (RSSM), an actor, and a critic trained entirely in latent space.  
Right: the Plan2Explore module, which adds an intrinsic reward based on ensemble disagreement in latent dynamics to drive efficient exploration. While grounded in DreamerV2, parts of this implementation incorporate stability and training ideas inspired by DreamerV3.
</div>

For implementation details, design choices, and experimental results, see the full repository:  
**https://github.com/PedroTajia/MyDreamerV2**

<!-- The code is simple.
Just wrap your images with `<div class="col-sm">` and place them inside `<div class="row">` (read more about the <a href="https://getbootstrap.com/docs/4.4/layout/grid/">Bootstrap Grid</a> system).
To make images responsive, add `img-fluid` class to each; for rounded corners and shadows use `rounded` and `z-depth-1` classes.
Here's the code for the last row of images above: -->
<!--
{% raw %}

```html
<div class="row justify-content-sm-center">
  <div class="col-sm-8 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/6.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm-4 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/11.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
```

{% endraw %} -->
